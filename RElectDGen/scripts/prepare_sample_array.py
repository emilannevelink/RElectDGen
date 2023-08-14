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
from RElectDGen.statistics.utils import save_cutoffs_distribution_info
from RElectDGen.uncertainty.io import get_dataset_uncertainties
from RElectDGen.statistics.subsample import subsample_uncertain
from RElectDGen.utils.data import reduce_trajectory
from RElectDGen.utils.md_utils import save_log_to_hdf5
from RElectDGen.utils.multiprocessing import batch_list
from RElectDGen.shell.slurm_tools import check_if_job_running, get_runtime, get_timelimit

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

    return config, MLP_config, args.config


def main(args=None):
    logging_dict = {}
    config, MLP_config, active_learning_config_filename = parse_command_line(args)
    active_learning_index = config.get('active_learning_index')
    symbols = MLP_config.get('chemical_symbol_to_type')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_directory = config.get('data_directory')
    if device == 'cpu':
        num_cores = config.get('cores',1)
        torch.set_num_threads(num_cores)
    
    UQ, unc_calc = load_unc_calc(config,MLP_config)

    ### get dataset uncertainties
    dataset_train_uncertainties, dataset_val_uncertainties = get_dataset_uncertainties(UQ)

    # Get Uncertainty Thresholds
    uncertainty_function = config.get('uncertainty_function', 'Nequip_latent_distance')
    use_validation_uncertainty = True if uncertainty_function in ['Nequip_latent_distance'] else False
    force_maxwell = config.get('force_maxwell',False)
    target_error = config.get('target_error')
    MLP_md_kwargs = config.get('MLP_md_kwargs')
    unc_out_all = {}
    for symbol in MLP_config.get('chemical_symbol_to_type'):
        unc_max_error_threshold_symbol = config.get('unc_max_error_threshold_symbol')
        if unc_max_error_threshold_symbol is None:
            max_error_dx_threshold = config.get('max_error_dx_threshold')
            if isinstance(max_error_dx_threshold, dict):
                unc_max_error_thresholdi = max_error_dx_threshold[symbol]
            else:
                unc_max_error_thresholdi = max_error_dx_threshold
            unc_max_error_threshold_symbol = unc_max_error_thresholdi * Atoms(symbol).get_masses()/MLP_md_kwargs.get('timestep',1)
        
        # print(dataset_train_uncertainties[symbol].shape,dataset_train_uncertainties[symbol])
        # print(dataset_val_uncertainties[symbol].shape,dataset_val_uncertainties[symbol])

        unc_out = get_all_dists_cutoffs(
            1000, # dummy number
            UQ.train_errors[symbol].detach().cpu().numpy(),
            UQ.test_errors[symbol].detach().cpu().numpy(),
            dataset_train_uncertainties[symbol],
            dataset_val_uncertainties[symbol],
            target_error = target_error,
            max_error=unc_max_error_threshold_symbol,
            use_validation_uncertainty = use_validation_uncertainty,
            force_maxwell = force_maxwell,
        )
        unc_out_all[symbol] = unc_out

        # print(unc_out['train_uncertainty_dict']['data'].shape,unc_out['train_uncertainty_dict']['data'])
        # print(unc_out['validation_uncertainty_dict']['data'].shape,unc_out['validation_uncertainty_dict']['data'])

    distribution_filename = os.path.join(
        data_directory,
        config.get('distribution_filename')
    )
    save_cutoffs_distribution_info(distribution_filename, unc_out_all, str(active_learning_index))

    data_directory = config.get('data_directory')
    db_filename = os.path.join(
        data_directory,
        config.get('ase_db_filename')
    )
    max_md_samples = config.get('max_md_samples',1)
    nsamples = config.get('md_sampling_initial_conditions',1)
    with connect(db_filename) as db:
        rows_initial = sample_from_ase_db(db, nsamples, max_md_samples)

    row_ids = [row['id'] for row in rows_initial]
    config['sample_row_ids'] = row_ids

    with open(active_learning_config_filename,'w') as fl:
        yaml.dump(config, fl)