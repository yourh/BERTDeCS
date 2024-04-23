#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/21
@author yrh

"""

import os
import click
import torch
import torch.distributed as dist
from pathlib import Path
from ruamel.yaml import YAML
from jinja2 import Environment, FileSystemLoader
from torch.utils.data import DataLoader
from logzero import logger

from bertdecs import get_now, set_logfile, is_master, barrier, get_option_list, highlight
from bertdecs.data_utils import get_dataset_from_cnf, get_mlb, output_res
from bertdecs.models import BaseModel
from bertdecs.samplers import OrderedDistributedSampler as DistributedSampler


def init_dist(model_cnf):
    dist.init_process_group(backend='nccl')
    model_cnf['model']['local_rank'] = local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    logger.info(f'Using DDP in {os.uname()[1]} with rank {dist.get_rank()}, '
                f'PID: {os.getpid()}, PPID: {os.getppid()}')
    if dist.get_rank() > 0:
        logger.setLevel(100)
    barrier()
    return dist.get_world_size()


def get_dataloader(data_cnf, features, mlb, batch_size, num_workers=4, is_training=False, **kwargs):
    return DataLoader(d_ := get_dataset_from_cnf(data_cnf, features, mlb, is_training=is_training, **kwargs),
                      batch_size, shuffle=None if dist.is_initialized() else is_training, num_workers=num_workers,
                      sampler=DistributedSampler(d_, shuffle=is_training) if dist.is_initialized() else None)


@click.command()
@click.argument('data-cnf', type=Path)
@click.argument('model-cnf', type=Path)
@click.option('-d', '--data', 'data_name', type=click.STRING, default='DeCS_ES', show_default=True)
@click.option('--data-path', type=click.STRING, default='data', show_default=True)
@click.option('-r', '--restore', type=Path, default=None)
@click.option('-p', '--pretrained', type=Path, default=None)
@click.option('--strict-load', is_flag=True)
@click.option('--model-id', type=click.STRING, default=None)
@click.option('--labels', type=click.STRING, default='decs', show_default=True)
@click.option('--train', 'train_flag', is_flag=True)
@click.option('--train-name', type=click.STRING, default='train', show_default=True)
@click.option('--train-labels', type=click.STRING, default=None, show_default=True)
@click.option('--valid-name', type=click.STRING, default='dev_st1', show_default=True)
@click.option('--valid-labels', type=click.STRING, default=None, show_default=True)
@click.option('--eval', 'eval_flag', type=click.STRING, default=None, show_default=True)
@click.option('-e', '--num-epochs', type=click.INT, default=None)
@click.option('--model-batch-size', type=click.INT, default=None)
@click.option('-b', '--device-batch-size', type=click.INT, default=None)
@click.option('--model-path', type=Path, default='models')
@click.option('--result-path', type=Path, default='results')
@click.option('--log-path', type=Path, default='logs')
@click.option('-a', '--amp', 'enable_amp', is_flag=True)
@click.option('--dist', 'enable_dist', is_flag=True)
@click.option('--device', type=click.STRING, default='cuda')
def main(data_cnf, model_cnf, data_name, data_path, model_id, train_flag, eval_flag,
         num_epochs, model_batch_size, device_batch_size, model_path, result_path, log_path,
         labels, train_name, train_labels, valid_name, valid_labels, device, enable_amp, 
         enable_dist, restore, pretrained, strict_load):
    m_id = f'-Model_{model_id}' if model_id is not None else ''
    yaml = YAML(typ='safe')
    env = Environment(loader=FileSystemLoader(data_cnf.parent))
    data_template = env.get_template(data_cnf.name)
    data_cnf = yaml.load(data_template.render(data_path=data_path, labels=labels))
    model_cnf = yaml.load(model_cnf)
    model, model_name = None, model_cnf['name']
    run_name = f'{model_name}-{data_name}'
    model_path = model_cnf['model']['model_path'] = model_path / f'{run_name}{m_id}.pt'
    features = model_cnf['features']
    mlb = get_mlb(Path(data_cnf['labels_list']))
    labels_list = mlb.classes_ if mlb is not None else None
    world_size = init_dist(model_cnf) if enable_dist else 1
    set_logfile(log_path / run_name / get_now())
    logger.info(f'Model Name: {model_name}')
    logger.info(f'Model Path: {highlight(model_path)}')
    logger.info(f'Dataset Name: {data_name}')
    logger.info(f'Features: {features}')

    model_batch_size = model_batch_size or model_cnf['batch_size']
    device_batch_size = device_batch_size or model_batch_size
    if model_batch_size // world_size < device_batch_size:
        device_batch_size = model_batch_size // world_size
    gradient_accumulation_steps = max(1, model_batch_size // (world_size * device_batch_size))
    if model_batch_size != world_size * device_batch_size * gradient_accumulation_steps:
        logger.warning(f'model_batch_size({model_batch_size}) is not consistent with '
                       f'device_batch_size({device_batch_size}) and world_size{world_size}')
    logger.info(f'Batch Size of Model: {model_batch_size}, '
                f'Batch Size of DataLoader: {device_batch_size}, '
                f'Gradient Accumulation Steps: {gradient_accumulation_steps}, '
                f'World Size: {world_size}')
    logger.info(f'Labels: {len(labels_list) if labels_list is not None else None}')
    model_cnf['model']['enable_amp'] = enable_amp
    model = BaseModel.get_model(labels_list=labels_list, device=device, **model_cnf['model'])

    if train_flag:
        logger.info(f'Loading Training and Validation Set: {train_name}, {valid_name}')
        model_cnf['train']['num_epochs'] = num_epochs or model_cnf['train']['num_epochs']
        train_loader = get_dataloader(yaml.load(data_template.render(data=train_name, data_path=data_path,
                                                                     labels=labels, data_labels=train_labels)),
                                      features, mlb, device_batch_size, is_training=True)
        if valid_name:
            valid_loader = get_dataloader(yaml.load(data_template.render(data=valid_name, data_path=data_path,
                                                                         labels=labels, data_labels=valid_labels)),
                                          features, mlb, device_batch_size)
        else:
            valid_loader = None
        logger.info(f'Size of Training Set: {len(train_loader.dataset)}')
        logger.info(f'Size of Validation Set: {len(valid_loader.dataset) if valid_loader else None}')
        training_options = {**model_cnf.get('train', {}), 'gradient_accumulation_steps': gradient_accumulation_steps,
                            'restore': restore, 'pretrained': pretrained, 'strict_load': strict_load}
        model.train(train_loader, valid_loader, **training_options)

    if eval_flag:
        for test_name in get_option_list(eval_flag):
            logger.info(f'Loading Test Set {test_name}')
            top_k = model_cnf.get('eval', {}).get('top_k', 100)
            test_loader = get_dataloader(yaml.load(data_template.render(data=test_name, data_path=data_path,
                                                                        labels=labels)),
                                         features, mlb, device_batch_size)
            logger.info(f'Predicting {test_name}, Top: {top_k}')
            logger.info(f'Size of Test Set {test_name}: {len(test_loader.dataset)}')
            res = model.predict(test_loader, top_k, **model_cnf.get('eval', {}))
            if is_master():
                output_res(result_path / f'{run_name}-{test_name}{m_id}', res, mlb)


if __name__ == '__main__':
    try:
        main()
    finally:
        barrier()
        if dist.is_initialized():
            dist.destroy_process_group()
