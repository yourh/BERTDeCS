#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/21
@author yrh

"""

import warnings
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.utils.logging
import adapters
from collections import OrderedDict
from transformers import AutoModel
from logzero import logger

transformers.utils.logging.set_verbosity_error()

__all__ = ['BERTEncoder', 'MLAttention', 'MLLinear']


class BaseModule(nn.Module):
    """

    """

    def __init__(self, log_indent='', **kwargs):
        super().__init__()
        logger.info(f'{log_indent}{type(self).__name__:}:')
        self.log_indent = log_indent + ' ' * 2

    def get_params_for_opt(self, lr, *, module_lr=None, log_indent='', **kwargs):
        module_lr = module_lr or lr
        logger.info(f'{log_indent}Get parameters for {type(self).__name__}: module_lr={module_lr}')
        params = {'params': [], 'lr': module_lr}
        params_no_wd = {'params': [], 'lr': module_lr, 'weight_decay': 0.0}
        for n, p in self.named_parameters():
            (params if 'bias' not in n and 'norm' not in n else params_no_wd)['params'].append(p)
        return [params, params_no_wd]


class BERTEncoder(BaseModule):
    """

    """

    def __init__(self, *, model_name='AutoModel', bert_params=(), adapter_name='adapter', adapter_cls=None,
                 adapter_options=(), freeze=False, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.bert_model = getattr(transformers, model_name, AutoModel).from_pretrained(**dict(bert_params))
        self.num_layers = self.bert_model.config.num_hidden_layers
        self.hidden_size = self.bert_model.config.hidden_size
        self.pad_token_id = self.bert_model.config.pad_token_id
        logger.info(f'{self.log_indent}{type(self.bert_model).__name__}:')
        for k, v in dict(bert_params).items():
            logger.info(f'{self.log_indent}  {k}={v}')
        logger.info(f'{self.log_indent}  hidden_size={self.hidden_size}')
        logger.info(f'{self.log_indent}  num_layers={self.num_layers}')
        logger.info(f'{self.log_indent}dropout={dropout}')
        if adapter_cls is not None:
            logger.info(f'{self.log_indent}Using Adapter with:')
            for k, v in dict(adapter_options).items():
                logger.info(f'{self.log_indent}  {k}={v}')
            adapters.init(self.bert_model)
            self.bert_model.add_adapter(adapter_name, config=getattr(adapters, adapter_cls)(**dict(adapter_options)),
                                        set_active=True)
            if freeze:
                self.bert_model.train_adapter(adapter_name)
            logger.info('\n' + self.bert_model.adapter_summary())
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, masks=None, pooling=False, **kwargs):
        if masks is None:
            masks = inputs != self.pad_token_id
        bert_out = self.dropout(self.bert_model(inputs, masks.int()).last_hidden_state)
        if pooling:
            bert_out = bert_out.masked_fill(~masks[..., None], 0.0).sum(dim=-2) / masks.sum(dim=-1, keepdims=True)
        return {**kwargs, 'outputs': bert_out, 'masks': masks}


class MLAttention(BaseModule):
    """

    """

    def __init__(self, num_labels, hidden_size, att_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        logger.info(f'{self.log_indent}num_labels={num_labels}')
        logger.info(f'{self.log_indent}hidden_size={hidden_size}')
        logger.info(f'{self.log_indent}att_dropout={att_dropout}')
        self.attention = nn.Linear(hidden_size, num_labels, bias=False)
        self.att_dropout = nn.Dropout(att_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.attention.weight)

    def forward(self, inputs, masks=None, **kwargs):
        att_s = self.attention(inputs)
        if masks is not None:
            att_s = att_s.masked_fill(~masks[..., None], -np.inf)
        return {**kwargs, 'outputs': torch.einsum('nld,nlc->ncd', inputs, self.att_dropout(F.softmax(att_s, dim=1)))}


class MLLinear(BaseModule):
    """

    """

    def __init__(self, input_size, hidden_sizes=(), output_size=None, squeeze_finally=False, **kwargs):
        super().__init__(**kwargs)
        logger.info(f'{self.log_indent}input_size={input_size}')
        logger.info(f'{self.log_indent}hidden_sizes={hidden_sizes}')
        logger.info(f'{self.log_indent}output_size={output_size}')
        hidden_sizes = [input_size] + list(hidden_sizes)
        self.linear_and_act = nn.Sequential(
            *sum([[
                nn.Linear(in_s, out_s),
                nn.ReLU(),
            ] for in_s, out_s in zip(hidden_sizes[:-1], hidden_sizes[1:])], []),
            nn.Linear(hidden_sizes[-1], output_size) if output_size is not None else nn.Identity(),
            nn.Flatten(-2) if squeeze_finally else nn.Identity()
        )
        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if 'weight' in n and p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs, **kwargs):
        return {**kwargs, 'outputs': self.linear_and_act(inputs)}
