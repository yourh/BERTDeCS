#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/21
@author yrh

"""

import click
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from logzero import logger

from bertdecs.data_utils import get_labels

import warnings
warnings.filterwarnings('ignore')


def get_metrics(prediction, targets, labels_list=None):
    if labels_list is not None:
        with open(labels_list) as fp:
            labels_list = set([line.strip() for line in fp])
    mlb = MultiLabelBinarizer(sparse_output=True)
    if labels_list is not None:
        mlb.fit([list(set(x) & labels_list) for x in list(prediction) + list(targets)])
    else:
        mlb.fit(list(prediction) + list(targets))
    targets: csr_matrix = mlb.transform(targets)
    prediction: csr_matrix = mlb.transform(prediction)
    correct = prediction.multiply(targets)

    p, r = correct.nnz / prediction.nnz, correct.nnz / targets.nnz
    mif = np.asarray((2 * p * r / (p + r) if p + r > 0.0 else 0.0, p, r))

    def get_avg_f(dim):
        prediction_, targets_, correct_ = prediction.sum(1 - dim), targets.sum(1 - dim), correct.sum(1 - dim)
        p_, r_ = correct_ / prediction_, correct_ / targets_
        p_, r_ = np.asarray(p_).squeeze(), np.asarray(r_).squeeze()
        f_ = 2 * p_ * r_ / (p_ + r_)
        f_[np.isnan(f_)] = 0.0
        return np.asarray((f_[~np.isnan(r_)].mean(), p_[~np.isnan(p_)].mean(), r_[~np.isnan(r_)].mean()))

    return mif, get_avg_f(1), get_avg_f(0), mlb


def get_labels_with_number(labels, scores, targets, number):
    if number is not None:
        if Path(number).exists():
            labels = [l_[:n_] for l_, n_ in zip(labels, np.load(number))]
        elif number == 'true' or number == 'True':
            labels = [l_[:len(set(t_))] for l_, t_ in zip(labels, targets)]
        elif number.isdigit():
            number = int(number)
            labels = [l_[:number] for l_ in labels]
        else:
            number = float(number)
            labels = [[ll_ for ll_, ss_ in zip(l_, s_) if ss_ >= number] for l_, s_ in zip(labels, scores)]
    return labels


@click.command()
@click.option('-r', '--results', type=Path, help='Path of results.')
@click.option('-t', '--targets', type=Path, help='Path of targets.')
@click.option('-n', '--number', type=click.STRING, default=None, help='Predicted number.')
@click.option('--labels-list', type=Path, default=None, help='Path of labels list.')
def main(results, targets, number, labels_list):
    results, targets = np.load(results, allow_pickle=True), get_labels(targets)
    labels = get_labels_with_number(results['labels'], results.get('scores', None), targets, number)
    idx = np.asarray([i for i, t_ in enumerate(targets) if t_])
    logger.info(f'Size of evaluation testing set is {len(idx)}')
    mif, maf, ebf, mlb = get_metrics(np.asarray(labels, dtype=object)[idx], np.asarray(targets, dtype=object)[idx],
                                     labels_list)
    print('MiF: {:.3f} MiP: {:.3f} MiR: {:.3f}'.format(*mif))
    print('MaF: {:.3f} MaP: {:.3f} MaR: {:.3f}'.format(*maf))
    print('EBF: {:.3f} EBP: {:.3f} EBR: {:.3f}'.format(*ebf))


if __name__ == '__main__':
    main()
