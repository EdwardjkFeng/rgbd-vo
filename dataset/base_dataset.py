""" Base dataset class """

from abc import ABCMeta, abstractmethod
import collections
import logging
from typing import Iterator
import omegaconf
from omegaconf import OmegaConf

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, get_worker_info


logger = logging.getLogger(__name__)


class LoopSampler(Sampler):
    def __init__(self, loop_size, total_size=None):
        self.loop_size = loop_size
        self.total_size = total_size - (total_size % loop_size)

    def __iter__(self):
        return (i % self.loop_size for i in range(self.total_size))
    
    def __len__(self):
        return self.total_size

    
class BaseDataset(metaclass=ABCMeta):
    """
    What the dataset model is expect to declare:
        default_conf: dictionary of the default configuration of the dataset.
        It overwrites base_default_conf in BaseModel, and it is overwritten by the user-provided config passed to __init__.
        Config can be nested.

        _init(self, conf): initialization method, where conf is the final config object (also accessible with `self.conf`). Accessing unknown configuration entries will raise an error.

        get_dataset(self, split): method that returns an instance of torch.utils.data.Dataset corresponding to the requested spilt string,
        which can be `'train'`, `'val'`, or `'test`'.
    """

    base_default_conf = {
        'name': '???',
        'num_workers': '???',
        'train_batch_size': '???',
        'val_batch_size': '???',
        'test_batch_size': '???',
        'shuffle_training': True,
        'batch_size': 1, 
        'num_threads': 1,
        'seed': 0,
    }
    default_conf = {}

    def __init__(self, conf):
        """ Perform some logic and call the _init method of the child model. """
        default_conf = OmegaConf.merge(
            OmegaConf.create(self.base_default_conf),
            OmegaConf.create(self.default_conf)
        )
        OmegaConf.set_struct(default_conf, True) # set conf immutable
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(self.conf, True)
        logger.info(f'Creating dataset {self.__class__.__name__}')
        self._init(self.conf)
    
    @abstractmethod
    def _init(self, conf):
        """ To be implemented in the child class. """
        raise NotImplementedError()
    
    @abstractmethod
    def get_dataset(self, split):
        """ To be implemented in the child class. """
        raise NotImplementedError()
    
    def get_data_loader(self, split, shuffle=None, pinned=True, distributed=False):
        """ Return a data loader for a given split. """
        assert split in ['train', 'val', 'test']
        dataset = self.get_dataset(split)
        try:
            batch_size = self.conf[split + '_batch_size']
        except omegaconf.MissingMandatoryValue:
            batch_size = self.conf.batch_size

        num_workers = self.conf.get('num_workers', batch_size)
        if shuffle is None:
            shuffle = (split == 'train' and self.conf.shuffle_training)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            pin_memory=pinned, num_workers=num_workers
        )

    def get_overfit_loader(self, split):
        """ Return an overfit data loader. 
        The training set is composed of a single duplicated batch, while the validation and test sets contain a single copy of this same batch.
        This is useful to debug a model and make sure that losses and metrics
        correlate well.
        """

        assert split in ['train', 'val', 'test']
        dataset = self.get_dataset('train')
        sampler = None
        num_workers = self.conf.get('num_workers', self.conf.batch_size)
        return DataLoader(
            dataset, batch_size=self.conf.batch_size,
            pin_memory=True, num_workers=num_workers,
            sampler=sampler, 
        )