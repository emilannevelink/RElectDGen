import argparse
import os
import yaml
import torch
import copy
import numpy as np
from nequip.utils import Config
from ase import Atoms
from ase.io import read, write
from ase.db import connect
import pandas as pd
import h5py

from RElectDGen.sampling.utils import sample_from_ase_db
from RElectDGen.utils.logging import write_to_tmp_dict
from RElectDGen.calculate.unc_calculator import load_unc_calc
from RElectDGen.sampling.md import md_from_atoms, sample_md_parallel
from RElectDGen.sampling.utils import get_uncertain, sort_traj_using_cutoffs, interpolate_T_steps, assemble_md_kwargs, truncate_using_cutoffs
from RElectDGen.statistics.cutoffs import get_all_dists_cutoffs, get_statistics_cutoff, get_best_dict
from RElectDGen.statistics.utils import load_cutoffs_distribution_info
from RElectDGen.uncertainty.io import get_dataset_uncertainties
from RElectDGen.statistics.subsample import subsample_uncertain
from RElectDGen.utils.data import reduce_trajectory
from RElectDGen.utils.md_utils import save_log_to_hdf5
from RElectDGen.utils.multiprocessing import batch_list
from RElectDGen.shell.slurm_tools import check_if_job_running, get_runtime, get_timelimit
from .sample_phase_space import sample_from_rows

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

    # MLP_config = Config.from_file(args.MLP_config)
    with open(args.MLP_config,'r') as fl:
            MLP_config = yaml.load(fl,yaml.FullLoader)

    return config, MLP_config, args.array_index

def main(args=None):
    logging_dict = {}
    config, MLP_config, array_index = parse_command_line(args)
    active_learning_index = config.get('active_learning_index')
    symbols = MLP_config.get('chemical_symbol_to_type')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        num_cores = config.get('cores',1)
        torch.set_num_threads(num_cores)
    
    UQ, unc_calc = load_unc_calc(config,MLP_config)

    ### get atoms objects
    data_directory = config.get('data_directory')
    db_filename = os.path.join(
        data_directory,
        config.get('ase_db_filename')
    )
    assert os.path.isfile(db_filename)
    max_md_samples = config.get('max_md_samples',1)
    nsamples = config.get('md_sampling_initial_conditions',1)
    nsample_parallel = config.get('md_sampling_parallel',1)
    with connect(db_filename) as db:
        rows_initial = sample_from_ase_db(db, nsamples, max_md_samples)[array_index::nsample_parallel]
    print(f'Sampling from {len(rows_initial)} starting configurations')
    
    # Load Uncertainty Thresholds
    distribution_filename = os.path.join(
        data_directory,
        config.get('distribution_filename')
    )
    unc_out_all = load_cutoffs_distribution_info(distribution_filename,str(active_learning_index))
    
    ### run MD on samples
    tmp_ase_traj_filename = os.path.join(
            data_directory,
            config.get('tmp_ase_traj_filename','')
        )
    if os.path.isfile(tmp_ase_traj_filename) and array_index==0:
        traj_uncertain = read(tmp_ase_traj_filename,index=':')
        print('Loaded trajectory of size: ', len(traj_uncertain))
        kill_id = config.get('last_sample_continuously_id')
        if kill_id is not None:
            if check_if_job_running(kill_id):
                command = f'scancel {kill_id}'
                os.system(command)
    else:
        traj_uncertain = []
    
    MLP_md_kwargs = config.get('MLP_md_kwargs')
    MLP_md_kwargs['data_directory'] = data_directory
    traj_add, nsamples = sample_from_rows(
        rows_initial,traj_uncertain,config,MLP_config,UQ,unc_calc,unc_out_all,True
    )
    
    if len(traj_add)>0:
        traj_add[0].info['nsamples']=nsamples
        print('Writing traj_add to tmp db')
        tmp_db_filename = os.path.join(
            data_directory,
            config.get('ase_db_filename').split('.db')[0]+f'_{array_index}.traj'
        )
        write(tmp_db_filename,traj_add)
    else:
        print('No samples greater than target trajectory')
    
    print('Sampling Complete')
    ### some sort of logging