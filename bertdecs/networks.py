#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/21
@author yrh

"""

import torch
import torch.nn as nn
from logzero import logger

from bertdecs.modules import *
from bertdecs.losses import ContrastiveLoss

__all__ = ['BaseNet', 'MLANet', 'CLNet']


class BaseNet(nn.Module):
    """

    """

    _networks = {}

    def __init_subclass__(cls, network_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if network_name is not None:
            cls._networks[network_name] = cls

    def __init__(self, *, encoder_options=(), labels_list=None, log_indent=''):
        super().__init__()
        logger.info(f'{log_indent}{type(self).__name__}:')
        self.log_indent = log_indent + ' ' * 2
        self.encoder = BERTEncoder(**dict(encoder_options), log_indent=self.log_indent)
        self.hidden_size = self.encoder.hidden_size
        self.labels_list = labels_list
        self.num_labels = len(self.labels_list) if labels_list is not None else None

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    @classmethod
    def get_network(cls, network_name='default', **kwargs):
        return cls._networks.get(network_name, BaseNet)(**kwargs)

    def get_params_for_opt(self, lr, *, encoder_lr=None, pooling_lr=None, **kwargs):
        return self.encoder.get_params_for_opt(lr, module_lr=encoder_lr, **kwargs)


class MLANet(BaseNet, network_name='MLANet'):
    """

    """

    def __init__(self, *, att_options=(), classifier_options=(), loss_options=(), **kwargs):
        super().__init__(**kwargs)
        self.attention = MLAttention(self.num_labels, self.hidden_size, **dict(att_options), log_indent=self.log_indent)
        self.classifier = self.get_classifier(**dict(classifier_options), log_ident=self.log_indent)
        self.loss_fn = nn.BCEWithLogitsLoss(**dict(loss_options))

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        outputs = self.attention(outputs.pop('outputs'), **outputs)
        outputs = self.classifier(outputs.pop('outputs'), **outputs)
        if 'targets' in outputs:
            outputs['loss'] = self.loss_fn(outputs['outputs'], outputs['targets'])
        return outputs

    def get_classifier(self, **kwargs):
        return MLLinear(input_size=self.hidden_size, output_size=1, squeeze_finally=True, log_indent=self.log_indent,
                        **kwargs)

    def get_params_for_opt(self, lr, *, att_lr=None, cls_lr=None, **kwargs):
        return [*super().get_params_for_opt(lr, **kwargs),
                *self.attention.get_params_for_opt(lr, module_lr=att_lr, **kwargs),
                *self.classifier.get_params_for_opt(lr, module_lr=cls_lr, **kwargs)]


class CLNet(BaseNet, network_name='CLNet'):
    """

    """

    def __init__(self, *, ph_options=(), loss_options=(), **kwargs):
        super().__init__(**kwargs)
        self.projection_head = MLLinear(input_size=self.hidden_size, **dict(ph_options), log_indent=self.log_indent)
        self.loss_fn = ContrastiveLoss(**dict(loss_options), log_indent=self.log_indent)

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs, pooling=True)
        outputs = self.projection_head(outputs.pop('outputs'), **outputs)
        outputs['loss'] = self.loss_fn(outputs['outputs'])
        return outputs

    def get_params_for_opt(self, lr, *, att_lr=None, cls_lr=None, **kwargs):
        return [*super().get_params_for_opt(lr, **kwargs),
                *self.projection_head.get_params_for_opt(lr, module_lr=cls_lr, **kwargs)]
