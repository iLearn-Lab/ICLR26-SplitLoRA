import os

import torch
import torch.nn as nn

from typing import Dict
import numpy as np
from copy import deepcopy
from .layers import LoRALayer, PlainMultiheadAttentionLoRA, LinearLoRA

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'lora_' in name:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import math




class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, pos='',layer=0,):
        super().__init__()
        self.in_dim, self.out_dim, self.rank = in_dim, out_dim, rank
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.ModuleList([nn.Linear(in_dim, self.rank, bias=False)])
        self.lora_B = nn.ModuleList([nn.Linear(self.rank, out_dim, bias=False)])

        nn.init.kaiming_uniform_(self.lora_A[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[0].weight)
        self.task_id = 0
        self.old_weight = 0.
        self.pos = pos
        self.layer = layer
        self.cur_matrix = torch.zeros(in_dim ,in_dim)
        self.n_cur_matrix = 0
        
    def update(self, task_id):
        lora_A, lora_B = self.lora_A[self.task_id].weight, self.lora_B[self.task_id].weight

        self.old_weight += lora_B @ lora_A
        self.lora_A.append(nn.Linear(self.in_dim, self.rank, bias=False))
        nn.init.kaiming_uniform_(self.lora_A[task_id].weight, a=math.sqrt(5))
        self.lora_B.append(nn.Linear(self.rank, self.out_dim, bias=False))
        nn.init.zeros_(self.lora_B[task_id].weight)
        self.task_id = task_id
    def update_grad_before_train(self, task_id):

        with torch.no_grad():
            if self.task_id == 0:
                return 
            else:
                random_coeffs = torch.randn(self.feature_list.shape[1],self.rank).to(self.lora_A[task_id].weight.device)
                initialized_matrix = self.feature_list @ random_coeffs
                initialized_matrix = initialized_matrix / initialized_matrix.norm() * self.lora_A[task_id].weight.data.norm()
                self.lora_A[task_id].weight.data.copy_(initialized_matrix.T)                

    def update_grad_after_train(self, task_id):
        with torch.no_grad():
            activation = self.cur_matrix
            U,S,Vh = torch.linalg.svd(activation, full_matrices=False)
            sval_total = (S).sum()
            sval_ratio = (S)/sval_total
            k = torch.arange(1, self.in_dim+1).to(self.lora_A[0].weight.device)-1
            k = (self.in_dim-k.float()) / self.in_dim
            result = (self.task_id+1)*(1-torch.cumsum(sval_ratio,dim=-1)) - self.alpha*k
            r = torch.argmin(result)
            self.feature_list = U[:,max(r,1)+1:]
            Uf=torch.matmul(self.feature_list,self.feature_list.T)
            wandb.log({f'feature_matrix/{self.layer}_{self.pos}':self.feature_list.shape[1]})
            self.feature_mat = Uf


    def forward(self, x, get_cur_feat=False):

        if get_cur_feat:
            self.cur_matrix = self.cur_matrix.to(x.device)
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0))/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]

        lora_A, lora_B = self.lora_A[self.task_id].weight, self.lora_B[self.task_id].weight
        result = F.linear(x, self.old_weight+lora_B @ lora_A)

        return result



class QkvWithLoRA(torch.nn.Module):
    def __init__(self, qkv, rank, alpha, layer):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        # self.lora_q = LoRALayer(self.dim, self.dim, rank, alpha,rate,args)
        self.lora_k = LoRALayer(self.dim, self.dim, rank, alpha, pos='k',layer=layer)
        self.lora_v = LoRALayer(self.dim, self.dim, rank, alpha, pos='v',layer=layer)
        self.layer = layer

    def update(self, task_id):
        # self.lora_q.update(task_id)
        self.lora_k.update(task_id)
        self.lora_v.update(task_id)
    def update_grad(self, task_id, many):
        if 'before' in many:
            self.lora_k.update_grad_before_train(task_id)
            self.lora_v.update_grad_before_train(task_id)
        else:
            self.lora_k.update_grad_after_train(task_id)
            self.lora_v.update_grad_after_train(task_id)
    def forward(self, x, get_cur_feat=False):

        qkv = self.qkv(x)
        qkv[:, :, self.dim:2*self.dim] += self.lora_k(x, get_cur_feat=get_cur_feat)
        qkv[:, :, -self.dim:] += self.lora_v(x, get_cur_feat=get_cur_feat)

        return qkv

 
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha,rate,args):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha,rate,args
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    def update(self, task_id):
        self.lora.update(task_id)

from functools import partial
def apply_lora(args, model):
    rank = args.r
    list_lora_layers = []
    if args.task_id == 0:
        assign_lora = partial(QkvWithLoRA, rank=rank,alpha=args.alpha)
        for i,block in enumerate(model.blocks):
            block.attn.qkv = assign_lora(block.attn.qkv,layer=i)
            if 'o' in args.params:
                block.attn.proj = LinearWithLoRA(block.attn.proj, rank)
            if 'mlp' in args.params:
                block.mlp.fc1 = LinearWithLoRA(block.mlp.fc1, rank)
                block.mlp.fc2 = LinearWithLoRA(block.mlp.fc2, rank)
    else:
        for i,block in enumerate(model.blocks):
            block.attn.qkv.update(task_id=args.task_id)
            if 'o' in args.params:
                block.attn.proj.update(task_id=args.task_id)
            if 'mlp' in args.params:
                block.mlp.fc1.update(task_id=args.task_id)
                block.mlp.fc2.update(task_id=args.task_id)
    return list_lora_layers

def update_grad(taskid, model, many):

    for i,block in enumerate(model.blocks):
        block.attn.qkv.update_grad(task_id=taskid, many=many)

    return 

def save_lora(args, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if 'q' in args.params:
            layer_weights['q_proj'] = {
                'w_lora_A': layer.w_lora_A.data,
                'w_lora_B': layer.w_lora_B.data
            }
        if 'k' in args.params:
            layer_weights['k_proj'] = {
                'w_lora_A': layer.w_lora_A.data,
                'w_lora_B': layer.w_lora_B.data
            }
        if 'v' in args.params:
            layer_weights['v_proj'] = {
                'w_lora_A': layer.w_lora_A.data,
                'w_lora_B': layer.w_lora_B.data
            }
        if 'out' in args.params:
            layer_weights['out_proj'] = {
                'w_lora_A': layer.w_lora_A.data,
                'w_lora_B': layer.w_lora_B.data
            }

        if 'fc1' in args.params:
            layer_weights['fc1'] = {
                'w_lora_A': layer.w_lora_A.data,
                'w_lora_B': layer.w_lora_B.data
            }

        if 'fc2' in args.params:
            layer_weights['fc2'] = {
                'w_lora_A': layer.w_lora_A.data,
                'w_lora_B': layer.w_lora_B.data
            }

        weights[f'layer_{i}'] = layer_weights

    metadata = {
        'r': args.r,
        'alpha': args.alpha,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }

    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    save_dir = f'{args.save_path}/{backbone}/seed{args.seed}'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{args.filename}.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')


def load_lora(args, list_lora_layers):
    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    load_path = f'{args.save_path}/{backbone}/seed{args.seed}/{args.filename}.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['r'] != args.r:
        raise ValueError(
            f"r mismatch: expected {args.r}, found {metadata['r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(
            f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata['params'] != args.params:
        raise ValueError(
            f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}")

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in layer_weights:
            layer.w_lora_A.data.copy_(
                layer_weights['q_proj']['w_lora_A'])
            layer.w_lora_B.data.copy_(
                layer_weights['q_proj']['w_lora_B'])
        if 'k' in args.params and 'k_proj' in layer_weights:
            layer.w_lora_A.data.copy_(
                layer_weights['k_proj']['w_lora_A'])
            layer.w_lora_B.data.copy_(
                layer_weights['k_proj']['w_lora_B'])
        if 'v' in args.params and 'v_proj' in layer_weights:
            layer.w_lora_A.data.copy_(
                layer_weights['v_proj']['w_lora_A'])
            layer.w_lora_B.data.copy_(
                layer_weights['v_proj']['w_lora_B'])
        if 'out' in args.params and 'out' in layer_weights:
            layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
            layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])

    print(f'LoRA weights loaded from {load_path}')
