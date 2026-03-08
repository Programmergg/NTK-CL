import copy
import math
import torch
import numpy as np
from torch import nn
from easydict import EasyDict
from models import prompt_vit
from torch.nn import functional as F
from models.fusion_nets import AttentionFusion

def get_backbone(args):
    name = args.backbone_type.lower()
    tuning_config = EasyDict(
        ffn=True,
        ffn_scalar="0.1",
        ffn_num=args.ffn_num,
        d_model=768,
        num_prompt_tokens=args.num_prompt_tokens,
        hidden_size=args.hidden_size,
        frozen_prompt_list_num=args.frozen_prompt_list_num,
        prompt_mode=args.model_name,
        _device=args.device[0],
        dataset=args.dataset,
    )
    if name == "vit_base_patch16_224_in1k":
        model = prompt_vit.vit_base_patch16_224_in1k(num_classes=0, drop_path_rate=0.0, tuning_config=tuning_config)
        model.out_dim = 768
    elif name == "vit_base_patch16_224_in21k":
        model = prompt_vit.vit_base_patch16_224_in21k(num_classes=0, drop_path_rate=0.0, tuning_config=tuning_config)
        model.out_dim = 768
    else:
        raise NotImplementedError("Unknown type {}".format(name))
    return model.eval()

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, frozen_prompt_list_num):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.frozen_prompt_list_num = frozen_prompt_list_num
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        self.sigma = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.sigma.data.fill_(1)

    def reset_parameters_to_zero(self):
        self.weight.data.fill_(0)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        out = self.sigma * out
        return {'logits': out, 'features': input}

    def forward_reweight(self, input, alpha=0.1, beta=0.0, gamma=1.0, out_dim=768, use_init_ptm=False):
        task_out = 0.0
        num_sections = self.frozen_prompt_list_num + 2 if use_init_ptm else self.frozen_prompt_list_num + 1
        # Define the multipliers for each section based on whether it's the initial PTM or not
        multipliers = [beta] + [gamma] * (num_sections - 2) + [alpha]
        if not use_init_ptm:
            multipliers = [gamma] * (num_sections - 1) + [alpha]
        # Process each section
        for i in range(num_sections):
            start_dim = i * out_dim
            end_dim = (i + 1) * out_dim
            # Normalize input and weight
            input_norm = F.normalize(input[:, start_dim:end_dim], p=2, dim=1)
            weight_norm = F.normalize(self.weight[:, start_dim:end_dim], p=2, dim=1)
            # Linear transformation and scaling
            linear_out = F.linear(input_norm, weight_norm) * multipliers[i]
            task_out += linear_out
        # Apply final scaling with sigma
        out_all = self.sigma * task_out
        return {'logits': out_all, 'features': input}

class MutualInformationLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super(MutualInformationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, targets):
        batch_size = features.size(0)
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positive_mask = targets.unsqueeze(1) == targets.unsqueeze(0)
        positive_mask[mask] = False
        positive_similarity = similarity_matrix[positive_mask]
        negative_mask = ~mask
        negative_similarity = similarity_matrix[negative_mask].view(batch_size, batch_size - 1)
        positive_loss = -torch.log(torch.exp(positive_similarity).sum())
        negative_loss = torch.logsumexp(negative_similarity, dim=1).mean()
        mi_loss = (positive_loss + negative_loss).mean()
        return mi_loss

class PromptNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc = None
        self.args = args
        self.backbone = get_backbone(args)
        self.out_dim = self.backbone.out_dim
        if self.args.use_init_ptm:
            self.fusion_model = AttentionFusion(self.out_dim * 3)
        else:
            self.fusion_model = AttentionFusion(self.out_dim * 2)
        self.nce_loss = MutualInformationLoss(temperature=0.1)

        self.reg_channel_prompt_list = copy.deepcopy(self.backbone.channel_prompt_list)
        for param in self.reg_channel_prompt_list.parameters():
            param.requires_grad = False
        self.reg_patch_prompt_list = copy.deepcopy(self.backbone.patch_prompt_list)
        for param in self.reg_patch_prompt_list.parameters():
            param.requires_grad = False
        self.reg_fusion_model = copy.deepcopy(self.fusion_model)
        for param in self.reg_fusion_model.parameters():
            param.requires_grad = False

    def initialize_fc(self, nb_classes, _cur_task):
        out_dim = self.out_dim * (self.args.frozen_prompt_list_num + 2) if self.args.use_init_ptm else self.out_dim * (self.args.frozen_prompt_list_num + 1)
        cls_num = self.args.init_cls if _cur_task == 0 else self.args.increment
        self.proxy_fc = CosineLinear(out_dim, cls_num, self.args.frozen_prompt_list_num).to(self.args.device[0])
        fc = CosineLinear(out_dim, nb_classes, self.args.frozen_prompt_list_num).to(self.args.device[0])
        fc.reset_parameters_to_zero()
        if self.fc is not None:
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[:self.fc.out_features, :] = nn.Parameter(weight)
        del self.fc
        self.fc = fc
        for param in self.fc.parameters():
            param.requires_grad = False

    def replace_fc(self, train_dataset, train_loader):
        self.eval()
        with torch.no_grad():
            embeddings, labels = [], []
            for _, data, label in train_loader:
                data, label = data.to(self.args.device[0]), label.to(self.args.device[0])
                if self.args.use_init_ptm:
                    channel_x, patch_x, _, _ = self.backbone(data, use_init_ptm=True)
                else:
                    channel_x, patch_x, _, _ = self.backbone(data, use_init_ptm=False)
                if self.args.model_name == 'patch':
                    embedding = patch_x
                elif self.args.model_name == 'channel':
                    embedding = channel_x
                elif self.args.model_name == 'both':
                    embedding = self.fusion_model(channel_x, patch_x)
                embeddings.append(embedding.cpu())
                labels.append(label.cpu())
            embeddings = torch.cat(embeddings, dim=0)
            labels = torch.cat(labels, dim=0)
            class_list = np.unique(train_dataset.labels)
            for class_index in class_list:
                data_index = (labels == class_index).nonzero().squeeze(-1)
                proto = embeddings[data_index].mean(0)
                self.fc.weight.data[class_index, :] = proto

    def orthogonalize_with_truncated_svd(self, channel_emb, patch_emb, mixed_emb, k=40):
        pre_emb_normalized = F.normalize(self.fc.weight.to(self.args.device[0]).transpose(0, 1), p=2, dim=0)
        u, s, v = torch.linalg.svd(pre_emb_normalized, full_matrices=True)
        basis = u[:, :k]
        def project_and_remove(matrix):
            projection = matrix @ basis @ basis.T
            return matrix - projection
        channel_emb_normalized = F.normalize(channel_emb, p=2, dim=-1)
        patch_emb_normalized = F.normalize(patch_emb, p=2, dim=-1)
        mixed_emb_normalized = F.normalize(mixed_emb, p=2, dim=-1)
        orthogonalized_channel_emb = project_and_remove(channel_emb_normalized)
        orthogonalized_patch_emb = project_and_remove(patch_emb_normalized)
        orthogonalized_mixed_emb = project_and_remove(mixed_emb_normalized)
        dis_loss = torch.sum(orthogonalized_channel_emb) + torch.sum(orthogonalized_patch_emb)
        return dis_loss

    def forward(self, x, test=False):
        channel_x, patch_x, channel_input_list, patch_input_list = self.backbone(x, use_init_ptm=self.args.use_init_ptm)
        mixed_x = self.fusion_model(channel_x, patch_x)
        orthogonal_loss = self.orthogonalize_with_truncated_svd(channel_x, patch_x, mixed_x)

        if test == False:
            channel_out = self.proxy_fc(channel_x)
            patch_out = self.proxy_fc(patch_x)
            mixed_out = self.proxy_fc(mixed_x)
        else:
            channel_out = self.fc.forward_reweight(channel_x, alpha=self.args.alpha, beta=self.args.beta, gamma=self.args.gamma, use_init_ptm=self.args.use_init_ptm)
            patch_out = self.fc.forward_reweight(patch_x, alpha=self.args.alpha, beta=self.args.beta, gamma=self.args.gamma, use_init_ptm=self.args.use_init_ptm)
            mixed_out = self.fc.forward_reweight(mixed_x, alpha=self.args.alpha, beta=self.args.beta, gamma=self.args.gamma, use_init_ptm=self.args.use_init_ptm)
        return channel_out, patch_out, mixed_out, orthogonal_loss