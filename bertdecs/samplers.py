#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2021/12/31
@author yrh

"""

from typing import TypeVar, Iterator

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, Dataset
from typing import Optional


T_co = TypeVar('T_co', covariant=True)

__all__ = ['OrderedDistributedSampler']


class OrderedDistributedSampler(DistributedSampler):
    """

    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = None, drop_last: bool = False) -> None:
        # if shuffle and seed is None:
        #     dist.broadcast(s_:=torch.randint(2**31, [1]).cuda(), src=0)
        #     seed = s_.item()
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            if self.seed is None:
                dist.broadcast(s_:=torch.randint(2**31, [1]).cuda(), src=0)
                seed = s_.item()
            else:
                seed = self.seed
            # logger.info(F'Seed of DistributedSampler is {seed}')
            # print(seed)
            g = torch.Generator()
            g.manual_seed(seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        # Return samples in order for convenience
        indices = indices[self.rank * (s_ := (self.total_size // self.num_replicas)): (self.rank + 1) * s_]
        assert len(indices) == self.num_samples

        return iter(indices)
