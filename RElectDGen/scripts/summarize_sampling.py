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
    active_learning_index = config.get('active_learning_index')
    symbols = MLP_config.get('chemical_symbol_to_type')
    data_directory = config.get('data_directory')
    nsample_parallel = config.get('md_sampling_parallel',1)
    
    UQ, unc_calc = load_unc_calc(config,MLP_config)

    nsamples = 0
    traj_add = []
    for array_index in range(nsample_parallel):
        print('Reading index: ', array_index)
        tmp_db_filename = os.path.join(
            data_directory,
            config.get('ase_db_filename').split('.db')[0]+f'_{array_index}.traj'
        )
        if os.path.isfile(tmp_db_filename):
            traj_addi = read(tmp_db_filename,index=':')
            nsamples+= traj_addi[0].info['nsamples']
            traj_add += traj_addi
    
    distribution_filename = os.path.join(
        data_directory,
        config.get('distribution_filename')
    )
    unc_out_all = load_cutoffs_distribution_info(distribution_filename,str(active_learning_index))

    minimum_uncertainty_cutoffs = {}
    maximum_uncertainty_cutoffs = {}
    for symbol in MLP_config.get('chemical_symbol_to_type'):
        best_dict = get_best_dict(unc_out_all[symbol]['train_uncertainty_dict'],unc_out_all[symbol]['validation_uncertainty_dict'],use_validation_uncertainty)
        minimum_uncertainty_cutoffs[symbol] = get_statistics_cutoff(nsamples,best_dict)
        maximum_uncertainty_cutoffs[symbol] = unc_out_all[symbol]['max_cutoff']
        if maximum_uncertainty_cutoffs[symbol] < minimum_uncertainty_cutoffs[symbol]:
            print(f'Resetting Minimum Cutoff for {symbol}')
            minimum_uncertainty_cutoffs[symbol] = 0.5*maximum_uncertainty_cutoffs[symbol]
    
    print('minimum_uncertainty_cutoffs', minimum_uncertainty_cutoffs)
    print('maximum_uncertainty_cutoffs', maximum_uncertainty_cutoffs)

    target_uncertainty_cutoffs = {}
    for symbol in MLP_config.get('chemical_symbol_to_type'):
        target_uncertainty_cutoffs[symbol] = unc_out_all[symbol]['target_cutoff']
        if maximum_uncertainty_cutoffs[symbol]*0.9 < target_uncertainty_cutoffs[symbol]:
            print(f'Resetting Target Cutoff for {symbol}')
            target_uncertainty_cutoffs[symbol] = 0.5*maximum_uncertainty_cutoffs[symbol]
    print('Target Uncertainty: ',target_uncertainty_cutoffs)

    for symbol in MLP_config.get('chemical_symbol_to_type'):
        minimum_uncertainty_cutoffs[symbol] = min([
            minimum_uncertainty_cutoffs[symbol],
            target_uncertainty_cutoffs[symbol]
        ])
        
    traj_uncertain_sorted = sort_traj_using_cutoffs(
        traj_add,
        minimum_uncertainty_cutoffs,
        maximum_uncertainty_cutoffs
    )

    max_samples = config.get('max_samples',10)
    if 'traj_add' not in locals() or len(traj_add) < len(traj_uncertain_sorted):
        traj_add = subsample_uncertain(
            UQ,
            traj_uncertain_sorted,
            minimum_uncertainty_cutoffs,
            maximum_uncertainty_cutoffs,
            max_add=max_samples,
            method=None
        )