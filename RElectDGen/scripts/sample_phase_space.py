import argparse
import os
import yaml
import torch
import numpy as np
from nequip.utils import Config
from ase import Atoms
from ase.io import Trajectory
from ase.db import connect

from RElectDGen.sampling.utils import sample_from_ase_db
from RElectDGen.utils.logging import write_to_tmp_dict
from RElectDGen.calculate.unc_calculator import load_unc_calc
from RElectDGen.sampling.md import md_from_atoms
from RElectDGen.sampling.utils import get_uncertain
from RElectDGen.statistics.cutoffs import get_all_dists_cutoffs, get_statistics_cutoff, get_best_dict
from RElectDGen.uncertainty.io import get_dataset_uncertainties
from RElectDGen.statistics.subsample import subsample_uncertain

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
    
    UQ, unc_calc = load_unc_calc(config,MLP_config)

    ### get atoms objects
    data_directory = config.get('data_directory')
    db_filename = os.path.join(
        data_directory,
        config.get('ase_db_filename')
    )
    assert os.path.isfile(db_filename)
    db = connect(db_filename)
    nsamples = config.get('md_sampling_initial_conditions',1)
    rows_initial = sample_from_ase_db(db, nsamples)

    
    ### get dataset uncertainties
    dataset_train_uncertainties, dataset_val_uncertainties = get_dataset_uncertainties(UQ)

    # Get Uncertainty Thresholds
    unc_out_all = {}
    for symbol in MLP_config.get('chemical_symbol_to_type'):
        unc_max_error_threshold = config.get('unc_max_error_threshold') * Atoms(symbol).get_masses()
        
        unc_out = get_all_dists_cutoffs(
            1000, # dummy number
            UQ.train_errors[symbol],
            UQ.test_errors[symbol],
            dataset_train_uncertainties[symbol],
            dataset_val_uncertainties[symbol],
            max_error=unc_max_error_threshold,
        )
        unc_out_all[symbol] = unc_out

    ### run MD on samples
    traj_uncertain = []
    nsamples = 0
    minimum_uncertainty_cutoffs = {}
    for row in rows_initial:
        atoms = row.toatoms()
        atoms.calc = unc_calc
        traj, stable = md_from_atoms(
            atoms,
            **config.get('MLP_md_kwargs'),
            data_directory=data_directory
        )

        nsamplesi = len(traj)*len(traj[0])
        nsamples += nsamplesi

        if stable:
            md_stable = row.get('md_stable') + 1
            db.update(row['id'],md_stable=md_stable)
        
        ### get uncertain samples
        for symbol in MLP_config.get('chemical_symbol_to_type'):
            best_dict = get_best_dict(unc_out_all[symbol]['train_uncertainty_dict'],unc_out_all[symbol]['validation_uncertainty_dict'])
            minimum_uncertainty_cutoffs[symbol] = get_statistics_cutoff(nsamplesi,best_dict)
        
        traj_uncertain += get_uncertain(traj,minimum_uncertainty_cutoffs)

    maximum_uncertainty_cutoffs = {}
    for symbol in MLP_config.get('chemical_symbol_to_type'):
        best_dict = get_best_dict(unc_out_all[symbol]['train_uncertainty_dict'],unc_out_all[symbol]['validation_uncertainty_dict'])
        minimum_uncertainty_cutoffs[symbol] = get_statistics_cutoff(nsamples,best_dict)
        maximum_uncertainty_cutoffs[symbol] = unc_out_all[symbol]['max_cutoff']
    
    print('minimum_uncertainty_cutoffs', minimum_uncertainty_cutoffs)
    print('maximum_uncertainty_cutoffs', maximum_uncertainty_cutoffs)

    traj_add = subsample_uncertain(
        UQ,
        traj_uncertain,
        minimum_uncertainty_cutoffs,
        maximum_uncertainty_cutoffs,
        max_add=10,
        method=None
    )

    # add traj to db
    active_learning_index = config.get('active_learning_index')
    for atoms in traj_add:
        db.write(atoms,md_stable=0,calc=False,active_learning_index=active_learning_index)
    
    print('Sampling Complete')
    ### some sort of logging