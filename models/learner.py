import torch
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from models.inc_net import PromptNet
from torch.utils.data import DataLoader

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data Length Error."
    all_acc = {"total": np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)}

    def compute_acc(mask):
        if len(mask) == 0: return 0
        return np.around((y_pred[mask] == y_true[mask]).sum() * 100 / len(mask), decimals=2)

    for class_id in range(0, np.max(y_true) + 1, increment):
        mask = np.where((y_true >= class_id) & (y_true < class_id + increment))[0]
        label = f"{class_id:02}-{class_id + increment - 1:02}"
        all_acc[label] = compute_acc(mask)
    all_acc["old"] = compute_acc(np.where(y_true < nb_old)[0])
    all_acc["new"] = compute_acc(np.where(y_true >= nb_old)[0])
    return all_acc

def update_ratios(task_num):
    if task_num == 0:
        return [0, 1]
    else:
        cur_ratio = 1 / (task_num + 1)
        pre_ratio = cur_ratio / update_ratios(task_num - 1)[-1]
        return [pre_ratio, cur_ratio]

class PromptLearner(object):
    def __init__(self, args):
        super().__init__()
        if args.dataset == 'eurosat' or args.dataset == 'eurosat_in21k' or args.dataset == 'plantvillage' or args.dataset == 'plantvillage_in21k' or args.dataset == 'kvasir' or args.dataset == 'kvasir_in21k':
            self.topk = 1
        else:
            self.topk = 5
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0

        self.args = args
        self._network = PromptNet(args)

        self.beta = args.beta
        self.alpha = args.alpha  # forward_reweight is divide by _cur_task
        self.min_lr = args.min_lr if args.min_lr is not None else 1e-8
        self.weight_decay = args.weight_decay if args.weight_decay is not None else 0.0005
        
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.initialize_fc(self._total_classes, self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        # self._network.show_trainable_params()
        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        if len(self.args.device) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self.args.device)
        self._train(self.train_loader)
        if len(self.args.device) > 1:
            self._network = self._network.module
        # Calculate the index for which "pre" prefix set to use
        if self._cur_task != self.args.nb_tasks - 1:
            index = self.args.frozen_prompt_list_num - self._cur_task // (self.args.nb_tasks // self.args.frozen_prompt_list_num)
            self._network.backbone.update_ema(self._network.backbone.channel_prompt_list, getattr(self._network.backbone, f"{'pre_' * (index)}channel_prompt_list"), update_ratios(self._cur_task))
            self._network.backbone.update_ema(self._network.backbone.patch_prompt_list, getattr(self._network.backbone, f"{'pre_' * (index)}patch_prompt_list"), update_ratios(self._cur_task))
        self._network.replace_fc(self.train_dataset_for_protonet, self.train_loader_for_protonet)

    def _train(self, train_loader):
        self._network.to(self.args.device[0])
        if self._cur_task == 0 or self.args.init_cls == self.args.increment:
            optimizer = self.get_optimizer(lr=self.args.init_lr)
            scheduler = self.get_scheduler(optimizer, self.args.init_epochs)
        else:
            # for base 0 setting, the later_lr and later_epochs are not used
            # for base N setting, the later_lr and later_epochs are used
            if "later_lr" not in self.args or self.args.later_lr == 0:
                self.args.later_lr = self.args.init_lr
            if "later_epochs" not in self.args or self.args.later_epochs == 0:
                self.args.later_epochs = self.args.init_epochs
            optimizer = self.get_optimizer(lr=self.args.later_lr)
            scheduler = self.get_scheduler(optimizer, self.args.later_epochs)
        self._init_train(train_loader, optimizer, scheduler)
    
    def get_optimizer(self, lr):
        if self.args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()),  momentum=0.9,  lr=lr, weight_decay=self.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()), lr=lr,  weight_decay=self.weight_decay)
        elif self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self._network.parameters()), lr=lr,  weight_decay=self.weight_decay)
        return optimizer
    
    def get_scheduler(self, optimizer, epoch):
        if self.args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args.scheduler == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args.init_milestones, gamma=self.args.init_lr_decay)
        elif self.args.scheduler == 'constant':
            scheduler = None
        return scheduler

    def _init_train(self, train_loader, optimizer, scheduler):
        # if self._cur_task == 0 or self.args.init_cls == self.args.increment:
        if self._cur_task == 0:
            epochs = self.args.init_epochs
        else:
            epochs = self.args.later_epochs
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.args.device[0]), targets.to(self.args.device[0])
                proxy_channel_out, proxy_patch_out, proxy_mixed_out, orthogonal_loss = self._network(inputs, test=False)
                proxy_channel_logits, proxy_patch_logits, proxy_mixed_logits = proxy_channel_out["logits"], proxy_patch_out["logits"], proxy_mixed_out["logits"]
                aux_targets = targets.clone()
                aux_targets = torch.where(aux_targets - self._known_classes >= 0, aux_targets - self._known_classes, -1)
                proxy_channel_loss = F.cross_entropy(proxy_channel_logits, aux_targets.long())
                proxy_patch_loss = F.cross_entropy(proxy_patch_logits, aux_targets.long())
                proxy_mixed_loss = F.cross_entropy(proxy_mixed_logits, aux_targets.long())

                pre_embeddings = self._network.fc.weight[:-self.args.increment, :].to(self.args.device[0])
                selected_labels = torch.randperm(pre_embeddings.shape[0])[:self.args.batch_size].to(self.args.device[0])
                selected_embeddings = pre_embeddings[selected_labels]
                channel_x, patch_x, mix_x = proxy_channel_out["features"], proxy_patch_out["features"], proxy_mixed_out["features"]
                orth_embeddings = torch.cat((mix_x, selected_embeddings), dim=0)
                orth_labels = torch.cat((targets, selected_labels), dim=0)
                mi_loss = -self._network.nce_loss(orth_embeddings, orth_labels)

                # Calculate L2 regularization
                reg_loss = torch.tensor(0.).to(self.args.device[0])
                for idx, (pre_module, module) in enumerate(zip(self._network.reg_channel_prompt_list, self._network.backbone.channel_prompt_list)):
                    for ((pre_name, pre_param), (name, param)) in zip(pre_module.named_parameters(), module.named_parameters()):
                        if param.requires_grad and 'weight' in name:
                            reg_loss += (param - pre_param).pow(2).sum()
                for idx, (pre_module, module) in enumerate(zip(self._network.reg_patch_prompt_list, self._network.backbone.patch_prompt_list)):
                    for ((pre_name, pre_param), (name, param)) in zip(pre_module.named_parameters(), module.named_parameters()):
                        if param.requires_grad and 'weight' in name:
                            reg_loss += (param - pre_param).pow(2).sum()
                params = [[self._network.reg_fusion_model.key, self._network.fusion_model.key], [self._network.reg_fusion_model.value, self._network.fusion_model.value], [self._network.reg_fusion_model.query, self._network.fusion_model.query]]
                for idx, (pre_module, module) in enumerate(params):
                    for ((pre_name, pre_param), (name, param)) in zip(pre_module.named_parameters(), module.named_parameters()):
                        if param.requires_grad and 'weight' in name:
                            reg_loss += (param - pre_param).pow(2).sum()

                if torch.isnan(mi_loss).any() or torch.isinf(mi_loss).any():
                    mi_loss = torch.nan_to_num(mi_loss, nan=0.0, posinf=0.0, neginf=0.0)
                if torch.isnan(orthogonal_loss).any() or torch.isinf(orthogonal_loss).any():
                    orthogonal_loss = torch.nan_to_num(orthogonal_loss, nan=0.0, posinf=0.0, neginf=0.0)
                if torch.isnan(reg_loss).any() or torch.isinf(reg_loss).any():
                    reg_loss = torch.nan_to_num(reg_loss, nan=0.0, posinf=0.0, neginf=0.0)
                
                if self.args.model_name == 'channel':
                    loss = proxy_channel_loss
                elif self.args.model_name == 'patch':
                    loss = proxy_patch_loss
                elif self.args.model_name == 'both':
                    loss = proxy_channel_loss + proxy_patch_loss + proxy_mixed_loss + mi_loss * self.args.nce_temp + orthogonal_loss * self.args.dis_temp + reg_loss * self.args.reg_temp

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                if self.args.model_name == 'channel':
                    _, preds = torch.max(proxy_channel_logits, dim=1)
                elif self.args.model_name == 'patch':
                    _, preds = torch.max(proxy_patch_logits, dim=1)
                elif self.args.model_name == 'both':
                    _, preds = torch.max(proxy_mixed_logits, dim=1)
                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)
            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)
        logging.info(info)

    def after_task(self):
        self._known_classes = self._total_classes
        self._network.reg_patch_prompt_list.load_state_dict(self._network.backbone.patch_prompt_list.state_dict())
        self._network.reg_channel_prompt_list.load_state_dict(self._network.backbone.channel_prompt_list.state_dict())
        self._network.reg_fusion_model.load_state_dict(self._network.fusion_model.state_dict())

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred[:, 0], y_true, self._known_classes, self.args.increment)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true), decimals=2)
        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_results(self.test_loader)
        # print(y_pred.shape, y_true.shape) # (1000, 5) (1000,)
        accy = self._evaluate(y_pred, y_true)
        return accy

    def _eval_results(self, loader):
        task_correct, task_acc, total = 0, 0, 0
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.args.device[0])
            with torch.no_grad():
                channel_out, patch_out, mixed_out, _ = self._network(inputs, test=True)
                if self.args.model_name == 'channel':
                    outputs = channel_out["logits"]
                elif self.args.model_name == 'patch':
                    outputs = patch_out["logits"]
                elif self.args.model_name == 'both':
                    outputs = mixed_out["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1] # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            task_ids = (targets - self.args.init_cls) // self.args.increment + 1
            task_logits = torch.zeros(outputs.shape).to(self.args.device[0])
            for i, task_id in enumerate(task_ids):
                if task_id == 0:
                    start_cls = 0
                    end_cls = self.args.init_cls
                else:
                    start_cls = self.args.init_cls + (task_id - 1) * self.args.increment
                    end_cls = self.args.init_cls + task_id * self.args.increment
                task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]
            # calculate the accuracy of task_id
            pred_task_ids = (torch.max(outputs, dim=1)[1] - self.args.init_cls) // self.args.increment + 1
            task_correct += (pred_task_ids.cpu() == task_ids).sum()
            pred_task_y = torch.max(task_logits, dim=1)[1]
            task_acc += (pred_task_y.cpu() == targets).sum()
            total += len(targets)
        logging.info("Task ID Acc: {}".format(tensor2numpy(task_correct) * 100 / total))
        logging.info("Class ID Acc in This Task: {}".format(tensor2numpy(task_acc) * 100 / total))
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]