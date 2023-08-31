import sys
sys.path.append("..")

import argparse
from pathlib import Path
import os
from omegaconf import OmegaConf

from utils.tools import Timers
from dataset.dummy_dataset import DummyLoader
from dataset import get_dataset

from tests import logger


def test_dataloader(conf):
    dataset = get_dataset(conf.data.name)(conf.data)
    print(dataset.images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--conf', type=str)
    parser.add_argument('dotlist', nargs='*')
    args = parser.parse_args()

    logger.info(f'Starting experiment {args.experiment}')

    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)
    

    
    test_dataloader(conf)