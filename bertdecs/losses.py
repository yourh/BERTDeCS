#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/26
@author yrh

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from logzero import logger

__all__ = ['ContrastiveLoss']


class ContrastiveLoss(nn.Module):
    """

    """

    def __init__(self, n_views=2, temperature=0.1, log_indent='', **kwargs):
        super().__init__(**kwargs)
        logger.info(f'{log_indent}ContrastiveLoss:')
        self.log_indent = log_indent + ' ' * 2
        logger.info(f'{self.log_indent}n_views={n_views}')
        logger.info(f'{self.log_indent}temperature={temperature}')
        self.ce_loss, self.n_views, self.temperature = nn.CrossEntropyLoss(), n_views, temperature

    class GatherLayer(torch.autograd.Function):
        """
        """

        @staticmethod
        def forward(ctx, inputs):
            dist.all_gather(outputs := [torch.zeros_like(inputs) for _ in range(dist.get_world_size())], inputs)
            return tuple(outputs)

        @staticmethod
        def backward(ctx, *grads):
            return grads[dist.get_rank()] * dist.get_world_size()

    def forward(self, embeddings):
        embeddings = F.normalize(embeddings, dim=1)
        labels = torch.arange(n_:=embeddings.shape[0], device=embeddings.device) % (n_ // self.n_views)
        if dist.is_initialized():
            embeddings = torch.vstack(self.GatherLayer.apply(embeddings))
            labels = (labels[None].repeat(w_:=dist.get_world_size(), 1) +
                      torch.arange(w_, device=embeddings.device)[:, None] * n_).flatten()
        labels = labels[:, None] == labels[None]

        similarity_matrix = embeddings @ embeddings.T

        mask = torch.eye(*labels.shape, dtype=torch.bool, device=embeddings.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels].view(-1, 1)
        negatives = similarity_matrix[~labels].view(similarity_matrix.shape[0], -1)
        negatives = negatives[:, None].expand(-1, self.n_views - 1, -1).flatten(0, 1)

        logits = torch.hstack([positives, negatives]) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=embeddings.device)
        return self.ce_loss(logits, labels)
