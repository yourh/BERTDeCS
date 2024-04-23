#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2024/4/23
@author yrh

"""

import click
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from logzero import logger


def get_bert_tokenizer(bert_name='bert-base-multilingual-cased', do_lower_case=False, max_length=512, use_fast=True):
    bert_name = 'bert-base-multilingual-cased' if bert_name == 'mbert' else bert_name
    logger.info(f'Using Tokenizer: {bert_name} with do_lower_case={do_lower_case}')
    return AutoTokenizer.from_pretrained(bert_name, do_lower_case=do_lower_case,
                                         model_max_length=max_length, use_fast=use_fast)


@click.group()
def main():
    pass


@main.command()
@click.option('-j', '--journal-path', type=Path, default=None)
@click.option('-t', '--title-path', type=Path, default=None)
@click.option('-a', '--abstract-path', type=Path, default=None)
@click.option('-o', '--output-path', type=Path, default=None)
@click.option('--bert-name', type=click.STRING, default='mbert')
@click.option('--do-lower-case', is_flag=True)
@click.option('--max-length', type=click.INT, default=512)
@click.option('--total', type=click.INT, default=None)
def tokenize(journal_path, title_path, abstract_path, output_path, bert_name, do_lower_case, max_length, total):
    tokenizer = get_bert_tokenizer(bert_name, do_lower_case, max_length)
    logger.info(f'Tokenizing Text: max_length={max_length}')
    input_ids = []
    with open(journal_path, 'rb') as fp1, open(title_path, 'rb') as fp2, open(abstract_path, 'rb') as fp3:
        for journal, title, abstract in tqdm(zip(fp1, fp2, fp3), total=total, dynamic_ncols=True):
            input_ids_ = tokenizer(journal.decode(), title.decode() + tokenizer.sep_token + abstract.decode(),
                                   truncation='longest_first', padding='max_length', max_length=max_length)['input_ids']
            assert len(input_ids_) == max_length
            input_ids.append(input_ids_)
    input_ids = np.asarray(input_ids)
    if total is not None:
        assert len(input_ids) == total
    logger.info(f'Size of Samples: {len(input_ids)}')
    np.save(output_path.with_stem(F'{output_path.stem}_{bert_name.replace("-", "_")}'), input_ids)


@main.command()
@click.argument('train-labels-path', type=Path)
@click.argument('labels-list-path', type=Path)
def label(train_labels_path, labels_list_path, freq):
    with open(train_labels_path) as fp:
        labels = sorted(set([x for line in fp for x in line.split()]))
    logger.info(f'Having {len(labels)} Labels')
    with open(labels_list_path, 'w') as fp:
        for x in labels:
            print(x, file=fp)


if __name__ == '__main__':
    main()
