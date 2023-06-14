import argparse
import os
import yaml
import torch
import numpy as np
from nequip.utils import Config
from ase.io import Trajectory

from RElectDGen.utils.logging import write_to_tmp_dict
from RElectDGen.uncertainty.io import load_UQ

def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    parser.add_argument('--MLP_config_file', dest='MLP_config',
                        help='Nequip configuration file', type=str)
    args = parser.parse_args(argsin)

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    # MLP_config = Config.from_file(args.MLP_config)
    with open(args.MLP_config,'r') as fl:
            MLP_config = yaml.load(fl,yaml.FullLoader)

    return config, MLP_config

def main(args=None):
    logging_dict = {}
    config, MLP_config = parse_command_line(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        num_cores = config.get('cores',1)
        torch.set_num_threads(num_cores)
    print(config)
    print(MLP_config)
    UQ = load_UQ(config,MLP_config)
    
    if config.get('calibrate_UQ',True):
        print('Ready to Calibrate', flush=True)
        UQ.calibrate()
        print('Calibrated', flush=True)
    else:
        print('Warning, UQ is uncalibrated')
    train_idcs = np.array(MLP_config.get('train_idcs'))
    val_idcs = np.array(MLP_config.get('val_idcs'))
    MLP_idcs = np.concatenate([train_idcs,val_idcs])

    logging_dict['n_train_indices'] = len(train_idcs)
    logging_dict['n_validation_indices'] = len(val_idcs)
    logging_dict['n_MLP_indices'] = len(MLP_idcs)

    tmp_filename = config.get('tmp_analyze_file',f'tmp_prepare.json')
    tmp_filename = os.path.join(config.get('directory'),config.get('run_dir'),'tmp',tmp_filename)
    write_to_tmp_dict(tmp_filename,logging_dict)