import timm
import math
import torch
import torch.nn as nn
from functools import partial
from torch.nn import LayerNorm
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed

class Channel_Prompt(nn.Module):
    def __init__(self, config=None, d_model=None, bottleneck=None, dropout=0.0, scalar="1.0"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck
        self.scale = float(scalar)
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout

        with torch.no_grad():
            # kaiming initialization
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            # nn.init.orthogonal_(self.down_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale
        return up

class Patch_Prompt(nn.Module):
    def __init__(self, config=None, scalar="1.0"):
        super(Patch_Prompt, self).__init__()
        self.config = config
        self.scale = float(scalar)
        self.global_patch_prompt = nn.Linear(197, self.config.num_prompt_tokens)
        self.prompt_norm = LayerNorm(self.config.hidden_size, eps=1e-6)
        with torch.no_grad():
            nn.init.zeros_(self.global_patch_prompt.weight)
            # nn.init.kaiming_uniform_(self.global_patch_prompt.weight, a=math.sqrt(5))
            nn.init.zeros_(self.global_patch_prompt.bias)
            nn.init.zeros_(self.prompt_norm.weight)
            nn.init.zeros_(self.prompt_norm.bias)

    def forward(self, x):
        residual_prompt = None
        if x.shape[1] == 197 + self.config.num_prompt_tokens:
            residual_prompt = x[:, 1: 1 + self.config.num_prompt_tokens, :]
            x = torch.cat((x[:, :1, :], x[:, (1 + self.config.num_prompt_tokens):, :]), dim=1)
        patch_in = torch.transpose(x, 2, 1)
        patch_in_prompts = self.global_patch_prompt(patch_in)
        patch_in_prompts = torch.transpose(patch_in_prompts, 2, 1)
        patch_prompts = self.prompt_norm(patch_in_prompts)
        if residual_prompt is not None:
            patch_prompts = patch_prompts + self.scale * residual_prompt
            x = torch.cat((x[:, :1, :], patch_prompts, x[:, 1:, :]), dim=1)
        else:
            x = torch.cat((x[:, :1, :], patch_prompts, x[:, 1:, :]), dim=1)
        return x, patch_prompts

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(self.q_proj(x), N, B).view(B * self.num_heads, -1, self.head_dim)
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)
        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)
        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_prompt_tokens=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.num_prompt_tokens = num_prompt_tokens
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x, channel_prompt=None, patch_prompt=None, mode=None):
        channel_x, patch_x = None, None
        x = x + self.drop_path(self.attn(self.norm1(x)))
        residual = x
        if mode == 'channel':
            if channel_prompt is not None:
                channel_x = channel_prompt(x)
            channel_out = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
            channel_out = self.drop_path(self.mlp_drop(self.fc2(channel_out)))
            if channel_prompt is not None:
                channel_out = channel_out + residual + channel_x
            else:
                channel_out = channel_out + residual
            return channel_out, channel_x
        elif mode == 'patch':
            if patch_prompt is not None:
                x, patch_x = patch_prompt(x)
            patch_out = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
            patch_out = self.drop_path(self.mlp_drop(self.fc2(patch_out)))
            if patch_out.shape != residual.shape:
                patch_out[:, 0, :] += residual[:, 0, :]
                patch_out[:, 1 + self.num_prompt_tokens:, :] += residual[:, 1:, :]
            else:
                patch_out = patch_out + residual
            return patch_out, patch_x

class VisionTransformer_Prompt(nn.Module):
    """ Vision Transformer with support for global average pooling"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None, tuning_config=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                      drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, num_prompt_tokens=tuning_config.num_prompt_tokens)
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.tuning_config = tuning_config
        self.frozen_prompt_list_num = tuning_config.frozen_prompt_list_num
        self.initialize_prompt_lists()

    def initialize_prompt_lists(self):
        """Helper function to initialize prompt lists."""
        self.channel_prompt_list = self.construct_channel_prompt_list()
        for i in range(1, self.frozen_prompt_list_num + 1):
            setattr(self, f"{'pre_' * i}channel_prompt_list", self.construct_channel_prompt_list())
        self.patch_prompt_list = self.construct_patch_prompt_list()
        for i in range(1, self.frozen_prompt_list_num + 1):
            setattr(self, f"{'pre_' * i}patch_prompt_list", self.construct_patch_prompt_list())

    def construct_channel_prompt_list(self):
        channel_prompt_list = nn.ModuleList()
        if self.tuning_config.ffn:
            for i in range(len(self.blocks)):
                channel_prompt = Channel_Prompt(self.tuning_config, dropout=0.1, bottleneck=self.tuning_config.ffn_num, scalar=self.tuning_config.ffn_scalar).to(self.tuning_config._device)
                channel_prompt_list.append(channel_prompt)
        return channel_prompt_list

    def construct_patch_prompt_list(self):
        patch_prompt_list = nn.ModuleList()
        if self.tuning_config.ffn:
            for i in range(len(self.blocks)):
                patch_prompt = Patch_Prompt(self.tuning_config, scalar=self.tuning_config.ffn_scalar).to(self.tuning_config._device)
                patch_prompt_list.append(patch_prompt)
        return patch_prompt_list

    def update_ema(self, source_modulelist, target_modulelist, task_ratio):
        with torch.no_grad():
            for source_module, target_module in zip(source_modulelist, target_modulelist):
                if source_module is None or target_module is None:
                    continue
                source_params = dict(source_module.named_parameters())
                target_params = dict(target_module.named_parameters())
                for name in source_params:
                    target_params[name].data.mul_(task_ratio[0]).add_(source_params[name].data, alpha=task_ratio[1])

    def forward_blocks(self, x, channel_prompt_list=None, patch_prompt_list=None, mode=None):
        if mode == 'channel':
            channel_x_prompt = []
            for blk, channel_prompt in zip(self.blocks, channel_prompt_list):
                x, channel_x = blk(x, channel_prompt=channel_prompt, mode=mode)
                channel_x_prompt.append(channel_x)
            return self.norm(x), channel_x_prompt
        elif mode == 'patch':
            patch_x_prompt = []
            for blk, patch_prompt in zip(self.blocks, patch_prompt_list):
                x, patch_x = blk(x, patch_prompt=patch_prompt, mode=mode)
                patch_x_prompt.append(patch_x)
            return self.norm(x), patch_x_prompt

    def forward(self, x, use_init_ptm=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if use_init_ptm:
            ptm_x = self.forward_blocks(x, channel_prompt_list=[None] * len(self.blocks), mode='channel')[0]
            channel_outputs = [self.forward_blocks(x, channel_prompt_list=getattr(self, f"{'pre_' * i}channel_prompt_list"), mode='channel') for i in range(self.frozen_prompt_list_num, -1, -1)]
            patch_outputs = [self.forward_blocks(x, patch_prompt_list=getattr(self, f"{'pre_' * i}patch_prompt_list"), mode='patch') for i in range(self.frozen_prompt_list_num, -1, -1)]
            channel_x_list = [output[0] for output in channel_outputs]
            patch_x_list = [output[0] for output in patch_outputs]
            channel_x_combined = torch.cat([xp[:, 0] for xp in channel_x_list], dim=1)
            patch_x_combined = torch.cat([xp[:, 0] for xp in patch_x_list], dim=1)
            channel_x_combined = torch.cat([ptm_x[:, 0], channel_x_combined], dim=1)
            patch_x_combined = torch.cat([ptm_x[:, 0], patch_x_combined], dim=1)
            channel_input_list = [output[1] for output in channel_outputs]
            patch_input_list = [output[1] for output in patch_outputs]
            return channel_x_combined, patch_x_combined, channel_input_list, patch_input_list
        else:
            channel_outputs = [self.forward_blocks(x, channel_prompt_list=getattr(self, f"{'pre_' * i}channel_prompt_list"), mode='channel') for i in range(self.frozen_prompt_list_num, -1, -1)]
            patch_outputs = [self.forward_blocks(x, patch_prompt_list=getattr(self, f"{'pre_' * i}patch_prompt_list"), mode='patch') for i in range(self.frozen_prompt_list_num, -1, -1)]
            channel_x_list = [output[0] for output in channel_outputs]
            patch_x_list = [output[0] for output in patch_outputs]
            channel_x_combined = torch.cat([xp[:, 0] for xp in channel_x_list], dim=1)
            patch_x_combined = torch.cat([xp[:, 0] for xp in patch_x_list], dim=1)
            channel_input_list = [output[1] for output in channel_outputs]
            patch_input_list = [output[1] for output in patch_outputs]
            return channel_x_combined, patch_x_combined, channel_input_list, patch_input_list

def vit_base_patch16_224_in1k(**kwargs):
    model = VisionTransformer_Prompt(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight
    msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)
    # freeze all but the prompt components
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    for i in range(model.frozen_prompt_list_num, 0, -1):
        saved_patch_variable = getattr(model, f"{'pre_' * i}patch_prompt_list")
        if isinstance(saved_patch_variable, nn.ModuleList):
            saved_patch_variable.requires_grad_(False)
        saved_channel_variable = getattr(model, f"{'pre_' * i}channel_prompt_list")
        if isinstance(saved_channel_variable, nn.ModuleList):
            saved_channel_variable.requires_grad_(False)
    return model

def vit_base_patch16_224_in21k(**kwargs):
    model = VisionTransformer_Prompt(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight
    msg = model.load_state_dict(state_dict, strict=False)
    # import clip
    # clip_model, preprocess = clip.load("ViT-B/16", device=kwargs['tuning_config']['_device'])
    # state_dict = clip_model.visual.state_dict()
    # print(msg)
    # freeze all but the prompt components
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    for i in range(model.frozen_prompt_list_num, 0, -1):
        saved_patch_variable = getattr(model, f"{'pre_' * i}patch_prompt_list")
        if isinstance(saved_patch_variable, nn.ModuleList):
            saved_patch_variable.requires_grad_(False)
        saved_channel_variable = getattr(model, f"{'pre_' * i}channel_prompt_list")
        if isinstance(saved_channel_variable, nn.ModuleList):
            saved_channel_variable.requires_grad_(False)
    return model