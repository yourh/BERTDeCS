#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/21
@author yrh

"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Union, Optional, List, Iterable, Hashable

__all__ = ['get_mif']

TPredict = np.ndarray
TTarget = Union[Iterable[Iterable[Hashable]], csr_matrix]
TMlb = Optional[MultiLabelBinarizer]
TClass = Optional[List[Hashable]]


def get_mlb(classes: TClass = None, mlb: TMlb = None, targets: TTarget = None):
    if classes is not None:
        mlb = MultiLabelBinarizer(classes=classes, sparse_output=True)
        mlb.fit(None)
    if mlb is None and targets is not None:
        if isinstance(targets, csr_matrix):
            mlb = MultiLabelBinarizer(classes=range(targets.shape[1]), sparse_output=True)
            mlb.fit(None)
        else:
            mlb = MultiLabelBinarizer(sparse_output=True)
            mlb.fit(targets)
    return mlb


def get_mif(prediction: TPredict, targets: TTarget, number: None | Iterable | int = None, mlb: TMlb = None,
            classes: TClass = None):
    mlb = get_mlb(classes, mlb, targets if isinstance(targets, csr_matrix) else list(prediction) + list(targets))
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    if number is None:
        number = np.asarray(targets.sum(axis=-1)).squeeze()
    if isinstance(number, int):
        number = [number] * targets.shape[0]
    prediction = mlb.transform((p[:n] for p, n in zip(prediction, number)))
    if prediction.nnz == 0:
        return 0.0
    correct = prediction.multiply(targets)
    p, r = correct.nnz / prediction.nnz, correct.nnz / targets.nnz
    return 2 * p * r / (p + r) if p + r > 0.0 else 0.0
