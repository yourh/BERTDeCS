#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/21
@author yrh

"""

import warnings
import json
import numpy as np
import scipy.sparse as ssp
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from logzero import logger
from typing import Optional

from bertdecs.datasets import MultiLabelDataset

__all__ = ['get_dataset_from_cnf', 'get_labels', 'get_mlb', 'output_res', 'get_res']


def load_features(data_file: Path, inputs_name=None, item=None):
    logger.info(f'Loading inputs {inputs_name} from {data_file} by {item} of data_cnf')
    return np.load(data_file) if data_file.suffix == '.npy' else ssp.load_npz(data_file)


def load_targets(label_file: Optional[Path] = None, mlb: Optional[MultiLabelBinarizer] = None):
    logger.info(f'Loading targets from {label_file}')
    if label_file is not None and label_file.exists() and mlb is not None:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            targets = mlb.transform(get_labels(label_file))
    else:
        targets = None
    return targets


def get_dataset_from_cnf(data_cnf, features, mlb, **kwargs):
    features_ = {}
    for f_, fv_ in features.items():
        n_ = fv_.get('feature', f_)
        data_ = load_features(Path(data_cnf[n_]), f_, n_)
        features_[f_] = {'data': data_, **fv_}
    targets_ = load_targets(Path(p_), mlb) if (p_ := data_cnf['labels']) is not None else None
    return MultiLabelDataset(features_, targets_, **kwargs)


def get_bow_data(bow_file, label_file=None, th=0.01):
    logger.info(f'Getting bow data from bow_file={bow_file} and label_file={label_file}')
    f_ = ssp.load_npz(bow_file)
    if th is not None:
        f_.data[np.abs(f_.data) < th] = 0.0
        f_.eliminate_zeros()
    return f_, get_labels(label_file) if label_file is not None else None


def get_labels(label_file):
    with open(label_file) as fp:
        return [line.split() for line in fp]


def get_labels_tokens(labels_tokens_file, labels_list):
    with open(labels_tokens_file) as fp:
        labels_dict = json.load(fp)
    return [labels_dict[l_]['token'] for l_ in labels_list]


def get_mlb(labels_file: Path) -> Optional[MultiLabelBinarizer]:
    if labels_file is not None and labels_file.exists():
        with open(labels_file) as fp:
            labels = [line.strip() for line in fp]
        mlb = MultiLabelBinarizer(classes=labels, sparse_output=True)
        mlb.fit(None)
        return mlb
    else:
        return None


def output_res(res_path: Path, res, mlb=None):
    labels, scores = res['labels'], res['scores']
    res_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(res_path.with_stem(f'{res_path.stem}-labels_idx'), labels)
    labels_ = mlb.classes_[labels] if mlb is not None else labels
    np.save(res_path.with_stem(f'{res_path.stem}-labels'), labels_)
    np.savez(res_path, labels=labels_, scores=scores)
    if mlb is not None:
        r, c, v = [], [], []
        for i, (l_, s_) in enumerate(zip(labels, scores)):
            r += [i] * len(l_)
            c += list(l_)
            v += list(s_)
        ssp.save_npz(res_path.with_stem(f'{res_path.stem}-scores'),
                     ssp.csr_matrix((v, (r, c)), shape=(len(labels), len(mlb.classes_))))
