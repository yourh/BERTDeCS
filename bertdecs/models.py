#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/21
@author yrh

"""

import contextlib
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from pathlib import Path
from collections import defaultdict
from itertools import chain
from dataclasses import dataclass, field
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from logzero import logger
from typing import Optional, Mapping, MutableMapping, Any

from bertdecs import is_master, barrier, highlight
from bertdecs.metrics import get_mif
from bertdecs.networks import BaseNet

__all__ = ['BaseModel']


class BaseModel(object):
    """

    """

    _models = {}

    def __init_subclass__(cls, model_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if model_name is not None:
            cls._models[model_name] = cls

    @classmethod
    def get_model(cls, model_name='default', *args, **kwargs):
        return cls._models.get(model_name, BaseModel)(*args, **kwargs)

    @dataclass
    class TrainState(object):
        """

        """

        cur_epoch: int = 0
        trained_steps: int = 0
        gradient_accumulated_steps: int = 0
        train_loss: list = field(default_factory=list)
        best: float = -np.inf
        early: int = 0

    def __init__(self, model_path: Path, device='cuda', enable_amp=False, local_rank=None, **kwargs):
        self.network = BaseNet.get_network(**kwargs).to(device)
        if local_rank is not None:
            self.network = nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.dp_network = DDP(self.network, device_ids=[local_rank], output_device=local_rank)
        else:
            self.dp_network = nn.DataParallel(self.network)
        self.model_path = model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.loss_fn = self.optimizer = self.lr_scheduler = None
        self.train_state: Optional[BaseModel.TrainState] = None
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        logger.info(f'Using AMP: {enable_amp}')
        self.enable_amp, self.scaler = enable_amp, torch.GradScaler(enabled=enable_amp)
        self.device = device
        self.train_batch_size = self.gradient_accumulation_steps = None
        self.loss_fn = self.optimizer = self.lr_scheduler = None
        self.train_state: BaseModel.TrainState | None = None

    def load_pretrained(self, pretrained: Path, strict_load=False, log_indent=''):
        if pretrained is not None and pretrained.exists() and pretrained.is_file():
            logger.info(f'{log_indent}Loading Pretrained Model from {highlight(pretrained)}:')
            log_indent += ' ' * 2
            pretrained_model = torch.load(pretrained, map_location='cpu', weights_only=True)
            for n, p in self.network.named_parameters():
                if n in pretrained_model and p.ndim == (p_:=pretrained_model[n]).ndim and \
                        (not strict_load or p.shape == p_.shape):
                    logger.info(f'{log_indent}Loading weights {n} of shape {tuple(p.shape)}')
                    if p.shape != p_.shape:
                        logger.warning(f'{log_indent}Loading weights {n} by weights of '
                                       f'a different shape {tuple(p_.shape)}')
                    p.data[tuple(slice(x) for x in p_.shape)] = p_[tuple(slice(x) for x in p.shape)]

    def get_optimizer(self, lr=1e-3, params_lr=(), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                      lr_scheduler=False, num_training_steps=0, num_warmup_steps=0.0, log_indent='', **kwargs):
        logger.info(f'{log_indent}AdamW Optimizer:')
        log_indent += ' ' * 2
        logger.info(f'{log_indent}lr={lr}')
        logger.info(f'{log_indent}eps={eps}')
        logger.info(f'{log_indent}weight_decay={weight_decay}')
        self.optimizer = torch.optim.AdamW(self.network.get_params_for_opt(lr, **dict(params_lr), log_indent=log_indent),
                                           betas=betas, eps=eps, weight_decay=weight_decay, **kwargs)
        if lr_scheduler:
            if num_warmup_steps < 1.0:
                num_warmup_steps = int(num_warmup_steps * num_training_steps)
            logger.info(f'{log_indent}Using linear lr scheduler:')
            log_indent += ' ' * 2
            logger.info(f'{log_indent}warmup_steps={num_warmup_steps}')
            logger.info(f'{log_indent}num_training_steps={num_training_steps}')
            self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)

    def move_inputs_to_device(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, tuple) or isinstance(obj, list):
            return type(obj)(self.move_inputs_to_device(x) for x in obj)
        if isinstance(obj, dict):
            return type(obj)({k: self.move_inputs_to_device(v) for k, v in obj.items()})
        return obj

    def get_outputs(self, inputs, **kwargs):
        with torch.autocast(self.device, enabled=self.enable_amp):
            outputs = self.dp_network(**self.move_inputs_to_device({**inputs, **kwargs}))
            if 'loss' in outputs:
                outputs['loss'] /= self.gradient_accumulation_steps
            return outputs

    def loss_and_backward(self, inputs: Mapping[str, Any], **kwargs):
        self.scaler.scale(loss := self.get_outputs(inputs, **kwargs)['loss']).backward()
        return loss

    def update_network(self):
        self.scaler.unscale_(self.optimizer)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    @property
    def update_network_step(self):
        return self.train_state.gradient_accumulated_steps == self.gradient_accumulation_steps

    @property
    def updated_network(self):
        return self.train_state.gradient_accumulated_steps == 0

    @property
    def grad_sync(self):
        return not dist.is_initialized() or self.update_network_step

    def train_step(self, inputs, **kwargs):
        self.dp_network.train()
        self.train_state.gradient_accumulated_steps += 1
        with ((contextlib.nullcontext if self.grad_sync else self.dp_network.no_sync)()):
            loss = self.loss_and_backward(inputs, **kwargs)
        if self.update_network_step:
            self.update_network()
            self.train_state.gradient_accumulated_steps = 0
            self.train_state.trained_steps += 1
        return loss.item()

    def init_train(self, train_loader: DataLoader, opt_options=(), num_epochs=10, gradient_accumulation_steps=1,
                   pretrained=None, strict_load=False, log_ident=''):
        self.train_state, self.gradient_accumulation_steps = self.TrainState(), gradient_accumulation_steps
        self.get_optimizer(num_training_steps=num_epochs * len(train_loader) // gradient_accumulation_steps,
                           **dict(opt_options), log_indent=log_ident)
        self.load_pretrained(pretrained, strict_load)

    def train(self, train_loader: DataLoader, valid_loader: Optional[DataLoader], opt_options=(), num_epochs=10,
              gradient_accumulation_steps=1, valid_steps=100, top_k=100, early=50, pretrained=None, strict_load=False,
              log_indent='', **kwargs):
        logger.info(f'{log_indent}Training:')
        log_indent += ' ' * 2
        logger.info(f'{log_indent}num_epochs={num_epochs}')
        self.init_train(train_loader, opt_options, num_epochs, gradient_accumulation_steps, pretrained, strict_load,
                        log_indent)
        while self.train_state.cur_epoch < num_epochs and (early is None or self.train_state.early <= early):
            self.train_state.cur_epoch += 1
            for inputs in tqdm(train_loader, desc=f'Epoch {self.train_state.cur_epoch}/{num_epochs}', leave=False,
                               dynamic_ncols=True, disable=not is_master(), unit='sample',
                               unit_scale=self.world_size * train_loader.batch_size):
                self.train_state.train_loss.append(self.train_step(inputs, **kwargs))
                if (self.updated_network and valid_steps is not None and
                        self.train_state.trained_steps % valid_steps == 0):
                    self.valid(valid_loader, top_k=top_k, log_indent=log_indent, **kwargs)
                    if early is not None and self.train_state.early > early:
                        break
        barrier()

    def valid(self, valid_loader: DataLoader, number=10, log_indent='', **kwargs):
        outputs, targets = self.predict(valid_loader, is_valid=True, **kwargs), valid_loader.dataset.targets
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mif_ = get_mif(outputs['labels'], targets, number)
        if mif_ > self.train_state.best:
            if is_master():
                self.save_network()
            self.train_state.best, self.train_state.early = mif_, 0
        else:
            self.train_state.early += 1
        training_log = (f'{log_indent}Epoch: {self.train_state.cur_epoch}, '
                        f'train loss: {np.mean(self.train_state.train_loss):.5f}, '
                        f'MiF: {mif_:.3f}, early stop: {self.train_state.early}')
        logger.info(training_log)
        self.train_state.train_loss = []

    @torch.no_grad()
    def predict_step(self, inputs: Mapping[str, Any], top_k: int, **kwargs):
        self.dp_network.eval()
        scores, labels = torch.topk(self.get_outputs(inputs, **kwargs)['outputs'], top_k)
        return {'scores': torch.sigmoid(scores).float(), 'labels': labels}

    def predict(self, data_loader: DataLoader, top_k=100, is_valid=False, **kwargs):
        if not is_valid:
            self.load_network()
        outputs = defaultdict(lambda: [[] for _ in range(self.world_size)])
        for inputs in tqdm(data_loader, desc='Predict', leave=False, dynamic_ncols=True, disable=not is_master(),
                           unit='sample', unit_scale=self.world_size * data_loader.batch_size):
            for k_, v_ in self.predict_step(inputs, top_k, **kwargs).items():
                if dist.is_initialized():
                    dist.all_gather(x_ := [torch.zeros_like(v_) for _ in range(self.world_size)], v_)
                    for i in range(self.world_size):
                        outputs[k_][i].append(x_[i].cpu().numpy())
                else:
                    outputs[k_][0].append(v_.cpu().numpy())
        barrier()
        return {k_: np.concatenate(list(chain(*outputs[k_])), axis=0)[:len(data_loader.dataset)] for k_ in outputs}

    def save_network(self):
        torch.save(self.network.state_dict(), self.model_path)

    def load_network(self):
        self.network.load_state_dict(torch.load(self.model_path, map_location='cpu', weights_only=True))


class CLModel(BaseModel, model_name='CLModel'):
    """

    """

    def loss_and_backward(self, inputs: MutableMapping[str, Any], **kwargs):
        inputs['inputs'] = torch.vstack([inputs.pop('pair_a'), inputs.pop('pair_b')])
        return super().loss_and_backward(inputs, **kwargs)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        torch.save(self.network.state_dict(), self.model_path)

    def valid(self, valid_loader, *args, **kwargs):
        logger.info(f'{self.train_state.cur_epoch} train loss: {np.mean(self.train_state.train_loss):.5f} ')
        self.train_state.train_loss = []
