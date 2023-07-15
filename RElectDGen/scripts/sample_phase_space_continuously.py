import argparse
import os
import yaml
import torch
import copy
import numpy as np
from nequip.utils import Config
from ase import Atoms
from ase.io import Trajectory, write
from ase.db import connect
import pandas as pd
import h5py

from RElectDGen.sampling.utils import sample_from_ase_db
from RElectDGen.utils.logging import write_to_tmp_dict
from RElectDGen.calculate.unc_calculator import load_unc_calc
from RElectDGen.sampling.md import md_from_atoms, sample_md_parallel
from RElectDGen.sampling.utils import get_uncertain, sort_traj_using_cutoffs, interpolate_T_steps, assemble_md_kwargs
from RElectDGen.statistics.cutoffs import get_all_dists_cutoffs, get_statistics_cutoff, get_best_dict
from RElectDGen.statistics.utils import load_cutoffs_distribution_info
from RElectDGen.uncertainty.io import get_dataset_uncertainties
from RElectDGen.statistics.subsample import subsample_uncertain
from RElectDGen.utils.data import reduce_trajectory
from RElectDGen.utils.md_utils import save_log_to_hdf5
from RElectDGen.utils.multiprocessing import batch_list
from RElectDGen.shell.slurm_tools import check_if_job_running

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

    return config, args.config, MLP_config

def main(args=None):
    logging_dict = {}
    config, config_filename, MLP_config = parse_command_line(args)
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
    with connect(db_filename) as db:
        rows_initial = sample_from_ase_db(db, nsamples, max_md_samples)
    print(f'Sampling from {len(rows_initial)} starting configurations')
    
    # Load Uncertainty Thresholds
    distribution_filename = os.path.join(
        data_directory,
        config.get('distribution_filename')
    )
    unc_out_all = load_cutoffs_distribution_info(distribution_filename,str(active_learning_index))
    ### run MD on samples
    traj_uncertain = []
    nsamples = 0
    minimum_uncertainty_cutoffs = {}
    nstable = 0

    nbatch_sample = config.get('sample_n_parallel',1)

    # unc_calc_mp = {
    #     'module': 'RElectDGen.calculate.unc_calculator',
    #     'calculator_type': 'UncCalculator',
    #     'calculator_kwargs': {'uq_module': {'config': config, 'MLP_config': MLP_config}}
    # } 
    # rows_batched = batch_list(rows_initial,nbatch_sample)
    uncertainty_function = config.get('uncertainty_function', 'Nequip_latent_distance')
    use_validation_uncertainty = True if uncertainty_function in ['Nequip_latent_distance'] else False
    MLP_md_kwargs = config.get('MLP_md_kwargs')
    MLP_md_kwargs['data_directory'] = data_directory
    for row in rows_initial:
        # for rows in rows_batched:
        # md_kwargs_list = assemble_md_kwargs(rows,unc_calc_mp,MLP_md_kwargs,max_md_samples)
        print('Sampling from row: ', row['id'])
        atoms = row.toatoms()
        atoms.calc = unc_calc
        MLP_md_kwargs = interpolate_T_steps(MLP_md_kwargs,row,max_md_samples)
        traj, log, stable = md_from_atoms(
            atoms,
            **MLP_md_kwargs
        )
        # trajs, logs, stables = sample_md_parallel(md_kwargs_list,nbatch_sample)

        # all_sampled = []
        # for (row, traj, log, stable) in zip(rows, trajs, logs, stables):
        dump_hdf5 = os.path.join(
            data_directory,
            config.get('MLP_dump_hdf5_file','dumps/energies_temperatures.hdf5')
        )
        save_log_to_hdf5(log,dump_hdf5,stable)

        nsamplesi = len(traj)*len(traj[0])
        nsamples += nsamplesi

        if stable:
            nstable += 1
            md_stable = row.get('md_stable') + 1
            with connect(db_filename) as db:
                print('updating row: ', row['id'], f' to md_stable = {md_stable}')
                db.update(row['id'],md_stable=md_stable)
                print(db.get(row['id'])['md_stable'])
    
        ### Reduce trajectory
        traj_reduced = reduce_trajectory(traj,config,MLP_config)
        # all_sampled += traj_reduced

        ### get uncertain samples
        for symbol in MLP_config.get('chemical_symbol_to_type'):
            best_dict = get_best_dict(unc_out_all[symbol]['train_uncertainty_dict'],unc_out_all[symbol]['validation_uncertainty_dict'],use_validation_uncertainty)
            minimum_uncertainty_cutoffs[symbol] = get_statistics_cutoff(nsamplesi,best_dict)

        traj_uncertain += get_uncertain(traj_reduced,minimum_uncertainty_cutoffs,symbols)

        print(f'{nstable} stable of {len(rows_initial)} md trajectories')

        maximum_uncertainty_cutoffs = {}
        for symbol in MLP_config.get('chemical_symbol_to_type'):
            best_dict = get_best_dict(unc_out_all[symbol]['train_uncertainty_dict'],unc_out_all[symbol]['validation_uncertainty_dict'],use_validation_uncertainty)
            minimum_uncertainty_cutoffs[symbol] = get_statistics_cutoff(nsamples,best_dict)
            maximum_uncertainty_cutoffs[symbol] = unc_out_all[symbol]['max_cutoff']
            if maximum_uncertainty_cutoffs[symbol] < minimum_uncertainty_cutoffs[symbol]:
                print(f'Resetting Maximimum Cutoff for {symbol}')
                maximum_uncertainty_cutoffs[symbol] = 2*minimum_uncertainty_cutoffs[symbol]
        
        print('minimum_uncertainty_cutoffs', minimum_uncertainty_cutoffs)
        print('maximum_uncertainty_cutoffs', maximum_uncertainty_cutoffs)

        traj_uncertain_sorted = sort_traj_using_cutoffs(
            traj_uncertain,
            minimum_uncertainty_cutoffs,
            maximum_uncertainty_cutoffs
        )

        print(f'{len(traj_uncertain_sorted)} uncertain samples of {nsamples} total sampled')
        max_samples = config.get('max_samples',10)*2
        if 'traj_add' not in locals() or len(traj_add) < len(traj_uncertain_sorted):
            traj_add = subsample_uncertain(
                UQ,
                traj_uncertain_sorted,
                minimum_uncertainty_cutoffs,
                maximum_uncertainty_cutoffs,
                max_add=max_samples,
                method=None
            )
            print(f'Recalculating traj_add: {len(traj_add)}')
        else:
            print('Not recalculating traj_add')
        print('Length of Add trajectory: ',len(traj_add))

        # add traj to db
        target_uncertainty_cutoffs = {}
        for symbol in MLP_config.get('chemical_symbol_to_type'):
            target_uncertainty_cutoffs[symbol] = unc_out_all[symbol]['target_cutoff']
        print('Target Uncertainty: ',target_uncertainty_cutoffs)
        traj_target = get_uncertain(traj_add,target_uncertainty_cutoffs,symbols) # overwriting uncertainties in traj_add during subsample
        print(f'Length of target trajectory is {len(traj_target)}')

        if len(traj_add) == max_samples and len(traj_target)>0:
            break
        
        next_config_filename = f'{active_learning_index+1}.'.join(config_filename.split(f'{active_learning_index}.'))
        if os.path.isfile(next_config_filename):
            with open(next_config_filename,'r') as fl:
                config_next = yaml.load(fl,yaml.FullLoader)
            if check_if_job_running(config_next['sample_id']):
                break

        traj_uncertain = copy.deepcopy(traj_add)
    
        tmp_traj_filename = os.path.join(
            data_directory,
            config.get('tmp_ase_traj_filename')
        )
        if os.path.isfile(tmp_traj_filename):
            os.remove(tmp_traj_filename)
        if len(traj_add)>0:
            print('Writing traj_add to db')
            write(tmp_traj_filename,traj_add)
        else:
            print('traj_add is of length ', len(traj_add))
    
    print('Sampling Complete')
    ### some sort of logging