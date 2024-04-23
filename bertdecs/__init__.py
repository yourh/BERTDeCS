#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/21
@author yrh

"""

import torch
import torch.distributed as dist
import datetime
import tqdm
import logging
import logzero
from pathlib import Path
from numpy.random import MT19937, RandomState, SeedSequence

__all__ = ['RS', 'get_now', 'set_logfile', 'is_master', 'barrier', 'get_option_list', 'highlight']

torch.backends.cuda.matmul.allow_tf32 = True
RS = RandomState(MT19937(SeedSequence(621668)))


class TqdmLoggingHandler(logging.StreamHandler):

    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.tqdm.write(msg)
        self.flush()


class AlignFormatter(logzero.LogFormatter):
    """

    """

    def format(self, record):
        if '\n' in (msg := record.getMessage()):
            return super().format(record)
        return f'{super().format(record).replace(msg, ""):<45}{msg}'


tqdm_handler = TqdmLoggingHandler()
tqdm_handler.setFormatter(logzero.LogFormatter())
logzero.logger.handlers.clear()
logzero.logger.addHandler(tqdm_handler)
logzero.formatter(AlignFormatter(), update_custom_handlers=True)


def get_now():
    return datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S')


def set_logfile(logfile: Path):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logzero.logfile(logfile.with_suffix('.log'))


def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def get_option_list(option):
    return option.split(',')


def highlight(msg):
    return f'\033[1;31m{msg}\033[0m'
