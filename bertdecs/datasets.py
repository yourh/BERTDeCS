#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/21
@author yrh

"""

import numpy as np
import scipy.sparse as ssp
from torch.utils.data import Dataset
from typing import Optional, Dict

__all__ = ['MultiLabelDataset']

TDataX = np.ndarray
TDataC = Dict[str, ssp.csr_matrix]
TDataY = Optional[ssp.csr_matrix]


class MultiLabelDataset(Dataset):
    """

    """

    def __init__(self, features, targets=None, is_training=False):
        self.features, self.targets, self.is_training = features, targets, is_training

    def __getitem__(self, item):
        inputs = self.get_features(item)
        if self.is_training:
            inputs['targets'] = self.get_targets(item)
        return inputs

    def __len__(self):
        for f_ in self.features:
            return self.features[f_]['data'].shape[0]

    def get_features(self, item):
        features_ = {}
        for f_ in self.features:
            features_[f_] = self.features[f_]['data'][item]
        return features_

    def get_targets(self, item):
        if self.targets is None:
            return 0.0
        return (self.targets[item].A.squeeze(0) if ssp.issparse(self.targets) else self.targets[item]).astype(np.float32)
