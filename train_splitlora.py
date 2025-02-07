import sys
import os
os.environ['TIMM_FUSED_ATTN'] = '0' if 'TIMM_FUSED_ATTN' not in os.environ else os.environ['TIMM_FUSED_ATTN']
import os.path as osp
from time import time as ttime
import argparse
import random
from collections import OrderedDict
import tqdm
from typing import Any, Literal
from copy import deepcopy
import warnings
import scipy.ndimage
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.cuda.amp.grad_scaler import GradScaler
import torch.utils.hooks
from torch.utils.data import DataLoader, TensorDataset
import torch.linalg
import torchvision
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from lora.utils import apply_lora,update_grad
from lora.utils import LoRALayer

import utils.vit_builder
from utils.vit_builder import VisionTransformer
from utils.dataset_builder import ImagePathDatasetClassManager, ImagePathDataset, Mixup, define_dataset
from utils.continual_manager import ClassIncrementalManager
from utils import misc
from einops import rearrange, reduce, repeat

torch.set_float32_matmul_precision("high")
import wandb 

class GlobalVarsManager:
    args: argparse.Namespace
    path_data_dict: dict[str, ImagePathDataset]
    cl_mngr: ClassIncrementalManager
    acc_mat_dict: OrderedDict[str, np.ndarray]
    cache_dict: dict
    param_dict: dict[Literal['base_params', 'task_params_'], OrderedDict[str, Tensor]]
    label_map_g2l: dict[int, tuple[int, int, int]]

    def init_from_args(self, args):
        self.args = args
        _dataset_class_manager = ImagePathDatasetClassManager(**{args.dataset: args.data_root})
        self.path_data_dict = {'train': _dataset_class_manager[args.dataset](train=True),
                               'eval': _dataset_class_manager[args.dataset](train=False)}
        self.cl_mngr = ClassIncrementalManager(self.path_data_dict['eval'].class_list, args.num_tasks, args.seed, shuffle=args.shuffle_classes)
        self.acc_mat_dict = OrderedDict(AccClassIncMat=np.zeros([_nt := self.cl_mngr.num_tasks, _nt]), AccClassIncList=np.zeros([_nt]))
        self.cache_dict = {}
        self.param_dict = {}
        self.label_map_g2l = {}

    def update_label_maps(self, taskid: int, task_classes: list[int]) -> tuple[dict[int, int], dict[str, int]]:
        _g2l_map = misc.make_label_maps(taskid, task_classes)
        if not all([_k not in self.label_map_g2l.keys() for _k in _g2l_map.keys()]):
            print("The global_to_local label map has been fully loaded, which is not expected.")
        self.label_map_g2l.update(_g2l_map)
        return _g2l_map


def get_args():
    parser = argparse.ArgumentParser(description='Class-incremental Learning')
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=('cifar100', 'imagenet_r', 'sdomainet'), help='use lowercase')
    parser.add_argument('-dr', '--data_root', type=str, default="")
    parser.add_argument('-t', '--num_tasks', type=int, default=10, choices=(1, 2, 5, 10, 20, 25, 50, 100))
    parser.add_argument('--shuffle_classes', type=misc.str2bool, default=True)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('-m', '--model', type=str, default='vit_base_patch16_clip_quickgelu_224.openai', help='vit_base_patch16_224.augreg_in21k, vit_base_patch16_clip_quickgelu_224.openai')
    parser.add_argument('--head_dim_type', type=str, choices=('task_classes', 'pretrained', 'text_dim'), default='pretrained')
    parser.add_argument('--logit_type', type=str, choices=('head_out', 'sim_imgtext'), default='sim_imgtext')
    parser.add_argument('--logit_scale', type=float, default=4.605170249938965, help='0 | 4.605170249938965')
    parser.add_argument('--logit_scale_trainable', type=misc.str2bool, default=False)
    parser.add_argument('--seperate_head', type=misc.str2bool, default=True)

    parser.add_argument('--ln_loss_lam', type=float, default=1.)
    parser.add_argument('--refine_head', type=misc.str2bool, default=False)
    parser.add_argument('--transform_type', type=str, choices=('timm', 'autoaug', 'prototype', 'clip'), default='autoaug')
    parser.add_argument('--prob_cutmixup', type=float, default=0)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-jt', '--workers', type=int, default=16)
    parser.add_argument('-je', '--eval_workers', type=int, default=2)
    parser.add_argument('-et', '--expand_times', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=28.)
    parser.add_argument('--use_amp', type=misc.str2bool, default=True)
    parser.add_argument('--sample_type', type=str, choices=('path', 'image'), default='image')
    parser.add_argument('--consecutive_training', type=misc.str2bool, default=True, help="")
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--persistent_workers', type=misc.str2bool, default=False)
    parser.add_argument('--training_string', type=str, nargs='+', default=('prompt',))
    parser.add_argument('-eb', '--eval_batch_size', type=int, default=100)

    parser.add_argument('--lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=20)

    parser.add_argument('--lr_scale', type=float, default=10.)
    parser.add_argument('--lr_scale_patterns', type=str, nargs='+')
    parser.add_argument('--optimizer', type=str, default='')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr_sch', type=str, default='multistep', choices=('cosine', 'step', 'multistep'))
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('-dm', '--decay_milestones', type=int, nargs='+', default=[5, 8])
    parser.add_argument('--decay_epochs', type=int, default=1000)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--show_bar', action='store_true')
    parser.add_argument('--print_model', action='store_true')

    args = parser.parse_args()

    return args


def seed_etc_options(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.set_printoptions(precision=4, linewidth=256)
    torch.set_printoptions(linewidth=256)
    torchvision.set_image_backend('accimage')


def set_model_mode(GVM: GlobalVarsManager, model: VisionTransformer, training: bool, to_gpu: bool = True, training_string: tuple[str] = ('prompt',),taskid=0) -> VisionTransformer:
    for name, p in model.named_parameters():
        p.requires_grad_(False)
        if training:
            if'head' in name:
                p.requires_grad_(True) 
            if 'lora_B' in name:
                id = name.split('lora_B.')[1]
                if str(taskid) in id:
                    p.requires_grad_(True)

    params_requires_grad = [n for n, p in model.named_parameters() if p.requires_grad]
    # print(f"Trainable params: {params_requires_grad}")
    model.eval()
    for n, m in model.named_modules():
        if training and any([(_s in n) and not isinstance(m, nn.Identity) for _s in training_string]):
            m.train()
        else:
            m.eval()
    modules_training = [n for n, m in model.named_modules() if m.training]
    params_requires_grad = [n for n, p in model.named_parameters() if p.requires_grad]
    # print(f"Trainable params: {params_requires_grad}")
    if to_gpu:
        model.cuda()

    return model


def set_learning_rates(GVM: GlobalVarsManager, model: VisionTransformer, base_lr: float, lr_scale: float, lr_scale_patterns: str) -> list[dict[str: Tensor | float]]:
    param_lr_groups = [{'params': [], 'lr': base_lr},
                       {'params': [], 'lr': 1e-2}]
    lr_param_dict = {_p['lr']: [] for _p in param_lr_groups}

    for name, p in model.named_parameters():
        if p.requires_grad:
            # import pdb;pdb.set_trace()
            _group_idx = 1 if 'head' in name else 0
            param_lr_groups[_group_idx]['params'].append(p)
            lr_param_dict[param_lr_groups[_group_idx]['lr']].append(name)
    return param_lr_groups


def train_one_epoch(GVM: GlobalVarsManager, curr_epoch: int, dataloader: DataLoader, model: VisionTransformer, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer,taskid: int) -> str:
    args = GVM.args
    temperature: float = args.temperature
    use_amp: bool = args.use_amp
    assert temperature > 0.

    _use_cutmixup = args.prob_cutmixup > 0

    if _use_cutmixup:
        cutmixup_fn = Mixup(mixup_alpha=1., cutmix_alpha=1., prob=args.prob_cutmixup, switch_prob=0.5, mode='batch', num_classes=len(GVM.cl_mngr.current_task_classes))

    amp_scalar = GradScaler(enabled=use_amp)
    scalar_meter = misc.ScalarMeter(loss="samp_avg:.4f", batch_time="step_sum:.3f", acc_top1="samp_avg:>6.2%")
    _btimer = ttime()

    for i_batch, (images, target) in tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)

        if _use_cutmixup:
            mix_img, mix_lbl = cutmixup_fn(images, target)
        else:
            mix_img = images
            mix_lbl = target

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits: Tensor = model(mix_img)

        if i_batch == 1:
            if args.seperate_head:
                assert logits.shape[1] == len(GVM.cl_mngr.current_task_classes)
            else:
                assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes)

        ce_loss = criterion(logits / temperature, mix_lbl)

        olora_loss = torch.zeros_like(ce_loss)
        loss: Tensor = ce_loss + olora_loss * args.ln_loss_lam

        optimizer.zero_grad()
        amp_scalar.scale(loss).backward()
        amp_scalar.step(optimizer)
        amp_scalar.update()

        acc_top1 = misc.calc_accuracy(logits, target, topk=(1,))[0]
        batch_time = ttime() - _btimer

        scalar_meter.add_step_value(len(images), loss=loss.item(), acc_top1=acc_top1, batch_time=batch_time)
        _btimer = ttime()

        wandb.log({f"train_loss/loss_{taskid}": loss.item(), f"train_acc/acc_top1_{taskid}": acc_top1})
    _epoch_scalar_str = scalar_meter.format_outout(scalar_meter.update_epoch_average_value())
    return _epoch_scalar_str


def cache_state(GVM: GlobalVarsManager, taskid: int, model: VisionTransformer):
    if taskid == 0:
        base_params = OrderedDict()
    task_params = OrderedDict()

    for n, p in model.named_parameters():
        if p.requires_grad:
            task_params[n] = p.clone()
            wandb.log({f"train_acc/task_params": torch.norm(p).item()})
        else:
            if taskid == 0:
                base_params[n] = p.clone()

    if taskid == 0:
        GVM.param_dict[f'base_params'] = base_params
    GVM.param_dict[f'task_params_{taskid}'] = task_params
    

def train_one_task(GVM: GlobalVarsManager, taskid: int, task_classes: list[int], model: VisionTransformer, args,**kwargs) -> VisionTransformer:
    args = GVM.args

    _ttimer = ttime()
    _ntstr = str(GVM.cl_mngr.num_tasks)

    model: VisionTransformer = set_model_mode(GVM, model, training=True, training_string=GVM.cache_dict['training_string'],taskid=taskid)
    model = modify_head(GVM, model, training=True, task_classes=task_classes)

    dataset = define_dataset(GVM, task_classes, training=True, transform_type=args.transform_type, target_map_to_local=args.seperate_head, expand_times=args.expand_times)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, timeout=args.timeout if args.workers > 0 else 0,
                            drop_last=args.prob_cutmixup > 0, persistent_workers=args.persistent_workers)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.lr_scale == 1:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())
    else:
        param_groups = set_learning_rates(GVM, model, args.lr, args.lr_scale, args.lr_scale_patterns)

    GVM.cache_dict['update_proj_dict'] = {}
    optimizer = create_optimizer_v2(param_groups, opt='adamW', lr=args.lr, weight_decay=args.weight_decay, foreach=True)

    scheduler, num_epochs = create_scheduler_v2(optimizer, sched=args.lr_sch, num_epochs=args.epochs, decay_epochs=args.decay_epochs, decay_milestones=args.decay_milestones,
                                                decay_rate=args.decay_rate, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs, warmup_lr=args.min_lr)
    assert num_epochs == args.epochs

    all_param = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rate = {
        "trainable_params": trainable_params,
        "orig_params": all_param,}
    print(f"Trainable params: {rate['trainable_params']}, Original params: {rate['orig_params']}, Trainable ratio: {rate['trainable_params'] / rate['orig_params']}")
    update_grad(taskid, model, 'before')
    torch.cuda.empty_cache()
    for epoch in range(0, args.epochs + 1):
        if epoch > 0:
            _epoch_scalar_str = train_one_epoch(GVM, epoch, dataloader, model, criterion, optimizer,taskid)
            print(f"Task [{taskid+1:>{len(_ntstr)}}/{_ntstr}] Epoch [{epoch:>{len(_nestr:=str(args.epochs))}}/{_nestr}]:: {_epoch_scalar_str}")
        scheduler.step(epoch)

    cache_state(GVM, taskid, model)

    with torch.no_grad():
        for i_batch, (images, target) in enumerate(dataloader, 1):
            # if i_batch %100 == 0:
            inputs: Tensor = images.cuda(non_blocking=True)
            model(inputs, get_cur_feat=True)

    update_grad(taskid, model, 'after')

    model.remove_text_features()
    print(f"Task [{taskid+1:>{len(_ntstr)}}/{_ntstr}]:: Training time = {misc.format_duration(ttime() - _ttimer)}")

    return model


def evaluate_one_task(GVM: GlobalVarsManager, train_taskid: int, eval_taskid: int, eval_task_classes: list[int], model: VisionTransformer) -> OrderedDict[str, float]:
    use_amp: bool = GVM.args.use_amp
    _ttimer = ttime()

    dataset = define_dataset(GVM, eval_task_classes, training=False, transform_type=GVM.args.transform_type, target_map_to_local=False)
    dataloader = DataLoader(dataset, batch_size=GVM.args.eval_batch_size, shuffle=False, num_workers=GVM.args.eval_workers, pin_memory=True, timeout=GVM.args.timeout if GVM.args.eval_workers > 0 else 0)

    set_model_mode(GVM, model, training=False)
    scalar_meter = misc.ScalarMeter(acc_class_inc="samp_avg:>6.2%")

    torch.cuda.empty_cache()
    for images, target in tqdm.tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)
        # import pdb;pdb.set_trace()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                logits: Tensor = model(images)

        assert logits.ndim == 2
        assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes), f"{logits.shape}, {len(GVM.cl_mngr.sofar_task_classes)}"

        _preds = logits.argmax(dim=1)

        acc_class_inc, _, _ = misc.calc_acc_topnn_dynamically(_preds, target)
        scalar_meter.add_step_value(target.shape[0], acc_class_inc=acc_class_inc)

    assert len(dataset) == len(scalar_meter)
    result_dict = scalar_meter.update_epoch_average_value()
    # import pdb;pdb.set_trace()
          ## Task [6/10]:: Eval [6/6]: eval_time=2.2s, acc_class_inc=93.32%
    print(f"Task [{train_taskid+1}/{GVM.cl_mngr.num_tasks}]:: Eval [{eval_taskid+1:>{len(_tt:=str(train_taskid+1))}}/{_tt}]: eval_time={ttime()-_ttimer:.1f}s, {scalar_meter.format_outout(result_dict)}")
    wandb.log({f"acc/{eval_taskid}":  result_dict['acc_class_inc']})
    result_dict['num_samples'] = len(dataset)

    return result_dict


def evaluate_tasks_sofar(GVM: GlobalVarsManager, train_taskid: int, model: VisionTransformer):
    model = modify_head(GVM, model, training=False)
    torch.cuda.empty_cache()
    average_acc_meter = misc.ScalarMeter(acc_class_inc="samp_avg:>6.2%")

    for eval_taskid in range(GVM.cl_mngr.current_taskid + 1):
        eval_task_classes = GVM.cl_mngr.get_classes(eval_taskid)
        one_result_dict = evaluate_one_task(GVM, train_taskid, eval_taskid, eval_task_classes, model)
        GVM.acc_mat_dict[f'AccClassIncMat'][train_taskid, eval_taskid] = one_result_dict['acc_class_inc']
        average_acc_meter.add_step_value(**one_result_dict)
    model.remove_text_features()

    avg_result_dict = average_acc_meter.update_epoch_average_value()
    GVM.acc_mat_dict[f'AccClassIncList'][train_taskid] = avg_result_dict['acc_class_inc']


def task_ending_info(GVM: GlobalVarsManager):
    current_taskid = GVM.cl_mngr.current_taskid

    acc_info_dict = {
        'class_inc_last_acc': float(GVM.acc_mat_dict['AccClassIncList'][current_taskid]),
        'class_inc_last_forg': misc.calc_forgetting(GVM.acc_mat_dict['AccClassIncMat'], current_taskid),
}
    wandb.log({f'eval/class_inc_last_acc': acc_info_dict['class_inc_last_acc'], 
              f'eval/class_inc_last_forg': acc_info_dict['class_inc_last_forg']})
    _formatter = misc.ScalarFormatter(sep=' | ', class_inc_last_acc=">6.2%", class_inc_last_forg=">6.2%")

    print(f":: ** Results of task [{current_taskid+1}]: [ {_formatter(**acc_info_dict)} ] **")
    print(f":: ** Time so far: {misc.format_duration(ttime() - GVM.cache_dict['exp_start_time'])} **")


def find_not_pretrained_params(model: VisionTransformer, pretrained: bool = True, pretrained_cfg: dict[str, str] = None, extra_pretrained_params: list[str] = []) -> list[str]:
    assert isinstance(extra_pretrained_params, (list, tuple))


    assert pretrained_cfg is not None

    if 'open_clip' in pretrained_cfg.get('hf_hub_filename', ''):
        _filename = timm.models._hub.HF_OPEN_CLIP_WEIGHTS_NAME
    else:
        _filename = timm.models._hub.HF_WEIGHTS_NAME
    pre_state_dict: OrderedDict[str, Tensor] = timm.models.load_state_dict_from_hf(pretrained_cfg['hf_hub_id'], _filename)

    if 'visual.class_embedding' in pre_state_dict.keys():
        pre_state_dict = timm.models.vision_transformer._convert_openai_clip(pre_state_dict, model)

    not_pretrained_params = []
    for n, p in model.named_parameters():
        if n not in pre_state_dict.keys() or not pretrained:
            not_pretrained_params.append(n)
        else:
            if p.shape != pre_state_dict[n].shape:
                not_pretrained_params.append(n)

    for n in deepcopy(not_pretrained_params):
        for _p in extra_pretrained_params:
            if _p in n:
                not_pretrained_params.remove(n)

    return not_pretrained_params


def get_param_id_dict(model: VisionTransformer, patterns: list[str]) -> dict[int, dict[Literal['name', 'shape'], str | list[int]]]:
    param_id_dict = {}
    for n, p in model.named_parameters():
        if p.requires_grad and any([_s in n for _s in patterns]):
            param_id_dict[id(p)] = {'name': n, 'shape': list(p.shape)}
    
    assert len(param_id_dict) > 0, f"{param_id_dict}"
    return param_id_dict


def get_head_dim_arg_dict(GVM: GlobalVarsManager, args: argparse.Namespace) -> dict[Literal['num_classes'], int]:
    head_dim_arg_dict = {}
    head_dim_type = args.head_dim_type

    match args.logit_type:
        case 'sim_imgtext':
            assert head_dim_type in ('pretrained', 'text_dim')
        case 'head_out':
            assert head_dim_type in ('task_classes')

    match head_dim_type:
        case 'task_classes':
            head_dim_arg_dict['num_classes'] = len(current_task_classes) if args.seperate_head else len(GVM.cl_mngr.sofar_task_classes)
        case 'pretrained':
            pass
        case 'text_dim':
            head_dim_arg_dict['num_classes'] = 512
        case _:
            raise ValueError(head_dim_type)
    return head_dim_arg_dict


def modify_head(GVM: GlobalVarsManager, model: VisionTransformer, training: bool, **kwargs):
    args: argparse.Namespace = GVM.args

    if training:
        _target_classes = kwargs['task_classes'] if args.seperate_head else GVM.cl_mngr.sofar_task_classes
    else:
        _target_classes = GVM.cl_mngr.sofar_task_classes


    if model.head.out_features != len(_target_classes):
        _mh = deepcopy(model.head)
        _mdevice = _mh.weight.device
        _mdtype = _mh.weight.dtype
        model.head = _mh.__class__(_mh.in_features, len(_target_classes), _mh.bias is not None, _mdevice, _mdtype)
        model.head.requires_grad_(_mh.weight.requires_grad)

        if training:
            assert model.head.weight.requires_grad
        else:
            assert _mh.out_features == len(GVM.cl_mngr.current_task_classes), f"{_mh.out_features}, {len(GVM.cl_mngr.current_task_classes)}"
            _hw = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.weight'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
            assert model.head.weight.data.shape == _hw.shape
            model.head.weight.data = _hw

            if _mh.bias is not None:
                _hb = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.bias'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                assert model.head.bias.data.shape == _hb.shape
                model.head.bias.data = _hb

    return model






if __name__ == "__main__":
    args = get_args()
    seed_etc_options(args.seed)

    GVM = GlobalVarsManager()
    GVM.init_from_args(args)

    GVM.cache_dict['exp_start_time'] = ttime()
    lr, rank, alpha = args.lr, args.rank, args.alpha
    wandb_name = f"{args.seed}"
    wandb.init(
        name=wandb_name,
        mode="offline",
        group="test",
        project=f"{args.dataset}_{args.num_tasks}",
    )
    for taskid, current_task_classes in GVM.cl_mngr:
        print(f"{'#'*90} Task: [{taskid+1}/{GVM.cl_mngr.num_tasks}] {'#'*90}")
        print(f"Current classes ({len(current_task_classes)}): {current_task_classes}")

        if not args.consecutive_training or taskid == 0:
            _other_args_dict = misc.get_specific_args_dict(args, 'logit_')
            _head_dim_arg_dict = get_head_dim_arg_dict(GVM, args)

            model: VisionTransformer = timm.create_model(args.model, pretrained=True, pretrained_strict=False, **_head_dim_arg_dict,
                                                         other_args_dict=_other_args_dict)
            GVM.cache_dict['pretrained_cfg'] = deepcopy(model.pretrained_cfg)

        model.num_tasks = args.num_tasks
        from argparse import Namespace
        param = ['k','v']

        r=  args.rank
        alpha = args.alpha
        apply_lora(Namespace(encoder='both', params=param, r=r, alpha=alpha, task_id=taskid, num_tasks=args.num_tasks), model) 
        args.training_string = ['lora','head']
        if args.consecutive_training and taskid > 0:
            pass
        GVM.update_label_maps(taskid, current_task_classes)
        GVM.cache_dict['training_string'] = args.training_string
        model = train_one_task(GVM, taskid, current_task_classes, model,args)
        evaluate_tasks_sofar(GVM, taskid, model)
        task_ending_info(GVM)

