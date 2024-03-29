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

    return config, MLP_config


def sample_from_rows(
        rows_initial, 
        traj_uncertain,
        config, 
        MLP_config, 
        UQ,
        unc_calc, 
        unc_out_all,
        save_tmp=False
    ):
    data_directory = config.get('data_directory')
    db_filename = os.path.join(
        data_directory,
        config.get('ase_db_filename')
    )
    symbols = MLP_config.get('chemical_symbol_to_type')
    max_md_samples = config.get('max_md_samples',1)
    nstable = 0
    nsamples = 0
    uncertainty_function = config.get('uncertainty_function', 'Nequip_latent_distance')
    use_validation_uncertainty = True if uncertainty_function in ['Nequip_latent_distance'] else False
    
    for row in rows_initial:
        # for rows in rows_batched:
        # md_kwargs_list = assemble_md_kwargs(rows,unc_calc_mp,MLP_md_kwargs,max_md_samples)
        print('Sampling from row: ', row['id'])
        atoms = row.toatoms()
        traj_check = reduce_trajectory([atoms],config,MLP_config)
        if len(traj_check) == 0:
            # set md_stable to max
            db.update(row['id'],md_stable=max_md_samples)
            continue
        
        atoms.calc = unc_calc

        MLP_md_kwargs = config.get('MLP_md_kwargs')
        MLP_md_kwargs['data_directory'] = data_directory
        # atoms.calc = unc_calc_mp gave memory error
        MLP_md_kwargs = interpolate_T_steps(MLP_md_kwargs,row,max_md_samples)
        traj, log, stable = md_from_atoms(
            atoms,
            **MLP_md_kwargs
        )
        for atoms_i in traj:
            atoms_i.info['start_row_id'] = row['id']
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

        ### Reduce trajectory
        traj_truncated = truncate_using_cutoffs(traj,maximum_uncertainty_cutoffs)
        print('Length of traj_truncated: ', len(traj_truncated))
        
        traj_reduced = reduce_trajectory(traj_truncated,config,MLP_config)
        # all_sampled += traj_reduced

        if len(traj_reduced) < len(traj):
            print('Resetting stable to false, traj reduced shorter than length of traj')
            stable = False

        traj_uncertaini = get_uncertain(traj_reduced,minimum_uncertainty_cutoffs,symbols)
        print(f'{len(traj_uncertaini)} uncertain samples')
        traj_uncertain += traj_uncertaini

        if stable and len(traj_uncertaini)==0:
            nstable += 1
            md_stable = row.get('md_stable') + 1
            with connect(db_filename) as db:
                print('updating row: ', row['id'], f' to md_stable = {md_stable}')
                db.update(row['id'],md_stable=md_stable)
                print(db.get(row['id'])['md_stable'])

        print(f'{nstable} stable of {len(rows_initial)} md trajectories')

        
        # this function is a bit too complicated as it really just needs to sort now.
        # tuncate removes maximum
        # get uncertain removes less than minimum
        traj_uncertain_sorted = sort_traj_using_cutoffs(
            traj_uncertain,
            minimum_uncertainty_cutoffs,
            maximum_uncertainty_cutoffs
        )

        print(f'{len(traj_uncertain_sorted)} uncertain samples of {nsamples} total sampled')
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
            print(f'Recalculating traj_add: {len(traj_add)}')
        else:
            print('Not recalculating traj_add')
        print('Length of Add trajectory: ',len(traj_add))

        # # add traj to db
        # traj_target = get_uncertain(traj_add,target_uncertainty_cutoffs,symbols) # overwriting uncertainties in traj_add during subsample
        # print(f'Length of target trajectory is {len(traj_target)}')

        if len(traj_add) == max_samples:
            break
        sample_id = config.get('sample_id')
        try:
            if get_timelimit(sample_id) - get_runtime(sample_id) < 60:
                break
        except:
            print('Time limit doesnt work')

        traj_uncertain = copy.deepcopy(traj_add)
        tmp_traj_filename = os.path.join(
            data_directory,
            config.get('tmp_ase_traj_filename','')
        )
        if os.path.isfile(tmp_traj_filename):
            os.remove(tmp_traj_filename)
        if len(traj_add)>0 and save_tmp:
            print('Writing traj_add to db')
            write(tmp_traj_filename,traj_add)
    
    return traj_add, nsamples

def main(args=None):
    logging_dict = {}
    config, MLP_config = parse_command_line(args)
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

    ### run MD on samples
    tmp_ase_traj_filename = os.path.join(
            data_directory,
            config.get('tmp_ase_traj_filename','')
        )
    if os.path.isfile(tmp_ase_traj_filename):
        traj_uncertain = read(tmp_ase_traj_filename,index=':')
        print('Loaded trajectory of size: ', len(traj_uncertain))
        kill_id = config.get('last_sample_continuously_id')
        if kill_id is not None:
            if check_if_job_running(kill_id):
                command = f'scancel {kill_id}'
                os.system(command)
    else:
        traj_uncertain = []
    
    
    nbatch_sample = config.get('sample_n_parallel',1)

    # unc_calc_mp = {
    #     'module': 'RElectDGen.calculate.unc_calculator',
    #     'calculator_type': 'UncCalculator',
    #     'calculator_kwargs': {'uq_module': {'config': config, 'MLP_config': MLP_config}}
    # } 
    # rows_batched = batch_list(rows_initial,nbatch_sample)
    MLP_md_kwargs['data_directory'] = data_directory
    traj_add, nsamples = sample_from_rows(
        rows_initial,traj_uncertain, config,MLP_config,UQ,unc_calc,unc_out_all,True
    )
    
    if len(traj_add)>0:
        print('Writing traj_add to db')
        with connect(db_filename) as db:
            for atoms in traj_add:
                db.write(atoms,md_stable=0,calc=False,active_learning_index=active_learning_index,start_row_id=atoms.info['start_row_id'])
    else:
        print('No samples greater than target trajectory')
    
    print('Sampling Complete')
    ### some sort of logging