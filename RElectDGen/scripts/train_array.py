import argparse, subprocess
from ase.io.trajectory import Trajectory
import numpy as np
import os
import yaml
import time
import json

import torch
from nequip.model import model_from_config
from nequip.data.transforms import TypeMapper
from nequip.utils import Config
from nequip.data import dataset_from_config

from ..uncertainty import models as uncertainty_models

from RElectDGen.calculate.calculator import nn_from_results
from RElectDGen.utils.save import check_NN_parameters
from RElectDGen.utils.logging import get_mae_from_results, write_to_tmp_dict, get_dataset_sizes

from memory_profiler import profile

def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    parser.add_argument('--MLP_config_file', dest='MLP_config',
                        help='Nequip configuration file', type=str)
    parser.add_argument('--array_index', dest='array_index',
                        help='active_learning_loop', type=int)
    args = parser.parse_args(argsin)

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    # with open(args.MLP_config,'r') as fl:
    #     MLP_config_new = yaml.load(fl,yaml.FullLoader)

    MLP_config_new = Config.from_file(args.MLP_config)

    return config, args.array_index


@profile
def main(args=None):
    
    start_time = time.time()
    config, array_index = parse_command_line(args)
    
    tmp_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('tmp_file','tmp.json'))

    if os.path.isfile(tmp_filename):
        with open(tmp_filename,'r') as fl:
            logging_dict = json.load(fl)
    else:
        logging_dict = {}
    
    MLP_config_filename = f'tmp_MLP_{array_index}.yaml'
    if logging_dict.get('load',False):
        commands = ['nequip-train-load', MLP_config_filename]
    else:
        commands = ['nequip-train', MLP_config_filename]

    if logging_dict.get('train',True):
        print('Training NN ... ', flush=True)
            
        process = subprocess.run(commands,capture_output=True)
        # print(process)
        [print(line) for line in process.stderr.split(b'\n')]

        # print(process.stderr.decode('ascii'))

    root = os.path.dirname(config['train_directory']) + f'_{array_index}'
    mae_dict = get_mae_from_results(root,index=array_index)

    logging_dict = {
        **mae_dict,
        f'train_time_{array_index}': time.time()-start_time
    }

    write_to_tmp_dict(tmp_filename,logging_dict)

if __name__ == '__main__':
    main()