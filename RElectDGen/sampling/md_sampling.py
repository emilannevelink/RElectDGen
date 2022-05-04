import ase
import os
import numpy as np
import pandas as pd

from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.md import MDLogger

import yaml
import torch

from ..uncertainty import models as uncertainty_models

from ..calculate.calculator import nn_from_results
from ..utils.logging import write_to_tmp_dict, add_checks_to_config
from ..structure.build import get_initial_structure
import time
from .utils import sort_by_uncertainty
from ..sampling.utils import sample_from_dataset

def MD_sampling(config, loop_learning_count=1):
    print('Starting timer', flush=True)
    start = time.time()
    MLP_dict = {}
    
    if config.get('cluster', False) or config.get('MD_from_initial', False):
        supercell = get_initial_structure(config)
    else:
        traj_initial = sample_from_dataset(config)
        supercell = traj_initial[0]
    
    #Delete Bondlength constraints
    supercell.constraints = [constraint for constraint in supercell.constraints if type(constraint)!=ase.constraints.FixBondLengths]
    

    ### Setup NN ASE calculator
    calc_nn, model, MLP_config = nn_from_results()
    supercell.calc = calc_nn

    tmp0 = time.time()
    print('Time to initialize', tmp0-start)

    ### Calibrate Uncertainty Quantification
    UQ_func = getattr(uncertainty_models,config.get('uncertainty_function', 'Nequip_latent_distance'))

    UQ = UQ_func(model, config, MLP_config)
    UQ.calibrate()
    # print(UQ.params,flush=True)
    # UQ_dict = UQ_params_to_dict(UQ.params,'MLP')
    # for key in UQ.params:
    #     print(key, UQ.params[key],flush=True) 
    #     if UQ.params[key][1] < config.get('mininmum_uncertainty_scaling',0):
    #         UQ.params[key][1] = config.get('mininmum_uncertainty_scaling')
    #         print('change sigma to minimum',flush=True)
    #         print(key, UQ.params[key],flush=True) 
            
    tmp1 = time.time()
    print('Time to calibrate UQ ', tmp1-tmp0, 'Elapsed time ', tmp1-start, flush=True)
    tmp0 = tmp1


    ### Run MLP MD
    MLP_dict['MLP_MD_temperature'] = config.get('MLP_MD_temperature') + (loop_learning_count-1)*config.get('MLP_MD_dT')
    MaxwellBoltzmannDistribution(supercell, temperature_K=MLP_dict['MLP_MD_temperature'])
    ZeroRotation(supercell)
    
    print(MLP_dict['MLP_MD_temperature'],flush=True)

    dyn = VelocityVerlet(supercell, timestep=config.get('MLP_MD_timestep') * units.fs)
    MLP_MD_dump_file = os.path.join(config.get('data_directory'),config.get('MLP_MD_dump_file'))
    #MDLogger only has append, delete log file
    if os.path.isfile(MLP_MD_dump_file):
        os.remove(MLP_MD_dump_file)
    dyn.attach(MDLogger(dyn,supercell,MLP_MD_dump_file,mode='w'),interval=1)
    
    trajectory_file = os.path.join(config.get('data_directory'),config.get('MLP_trajectory_file'))
    traj = Trajectory(trajectory_file, 'w', supercell)
    dyn.attach(traj.write, interval=1)
    try:
        dyn.run(config.get('MLP_MD_steps'))
    except ValueError:
        print('Value Error: MLP isnt good enough for current number of steps')
    traj.close()

    tmp1 = time.time()
    print('Time to run MD', tmp1-tmp0, 'Elapsed time ', tmp1-start, flush=True)
    tmp0 = tmp1

    # print('Done with MD', flush = True)
    # Check temperature stability
    MLP_log = pd.read_csv(MLP_MD_dump_file,delim_whitespace=True)
    try:
        max_T_index = int(np.argwhere(MLP_log['T[K]'].values>2000)[0])
    except IndexError:
        max_T_index = int(config.get('MLP_MD_steps')+1)

    if max_T_index < config.get('MLP_MD_steps'):
        print(f'max T index {max_T_index} of {len(MLP_log)} MLP_MD_steps', flush=True)
    else:
        print(f'Temperature stable: max T index {max_T_index}', flush=True)

    traj = Trajectory(trajectory_file)
    traj = traj[:max_T_index] # Only use T stable indices
    MLP_dict['MLP_MD_steps'] = len(traj)

    max_samples = int(min([0.1*len(traj), config.get('max_samples')]))
    max_traj_len = config.get('max_atoms_to_segment',2*max_samples)

    expected_max_index = config.get('MLP_MD_steps')+1
    reduction_factor = 1
    if len(traj) > max_traj_len:
        reduction_factor = np.ceil(len(traj)/max_traj_len).astype(int)
        expected_max_index = int(np.ceil(expected_max_index/reduction_factor))
        traj = traj[::reduction_factor]
        print(f'reduced length of trajectory by {reduction_factor}, new length {len(traj)}, new max_index {expected_max_index}', flush=True)

    uncertainty, embeddings = UQ.predict_from_traj(traj,max=False)

    MLP_dict['MLP_error_value'] = float(uncertainty.mean())
    MLP_dict['MLP_error_std'] = float(uncertainty.std())
    print('MLP error value', MLP_dict['MLP_error_value'], flush=True)
    print('MLP error std', MLP_dict['MLP_error_std'],flush=True)

    min_sigma = config.get('UQ_min_uncertainty')
    max_sigma = config.get('UQ_max_uncertainty')

    uncertainty_thresholds = [max_sigma, min_sigma]
    config['uncertainty_thresholds'] = uncertainty_thresholds

    try:
        max_index = int((uncertainty.max(axis=1).values>5*max_sigma).nonzero()[0])
    except IndexError:
        print('Index Error', uncertainty, flush=True)
        max_index = len(uncertainty)
    print('max index: ', max_index,flush=True)

    checks = {
        'MD_mean_uncertainty': float(uncertainty.mean())<config.get('UQ_min_uncertainty'),
        'MD_std_uncertainty': float(uncertainty.std())<config.get('UQ_min_uncertainty')/2,
        'MD_max_index': max_index==expected_max_index,
    }

    if max_index < 10./reduction_factor and config.get('MLP_MD_steps')>10:
        MLP_log = pd.read_csv(MLP_MD_dump_file,delim_whitespace=True)
        try:
            max_index = int(np.argwhere(MLP_log['T[K]'].values>2000)[0])
        except IndexError:
            max_index = int(config.get('MLP_MD_steps')+1)
        print(f'max index not high enough resetting to {max_index}', flush=True)
        sorted = False
        uncertainty_thresholds[0] *= 2 
        print(f'reset uncertainty thresholds now max: {uncertainty_thresholds[0]}, min: {uncertainty_thresholds[1]}')
        # print('max index not high enough, adding 5 and 10')
        # traj_indices = [5,10]
        # traj = [traj[i] for i in traj_indices]
        # uncertainty = uncertainty[traj_indices]
    else:
        sorted = True
        
    traj = traj[:max_index]
    uncertainty = uncertainty[:max_index]
    config['sorted'] = sorted

    # Create a function for clustering later
    # print('isolating uncertain clusters', flush=True)
    # clusters, cluster_uncertainties = clusters_from_traj(traj, uncertainty, **config)

    tmp1 = time.time()
    print('Time to segment clusters', tmp1-tmp0, 'Elapsed time ', tmp1-start, flush=True)
    tmp0 = tmp1

    MLP_dict['number_uncertain_points'] = len(traj)
    if len(traj) == 0:
        print('No clusters to calculate', flush=True)
        MLP_dict['number_MD_samples'] = 0
        checks['MD_count'] = True
        traj_uncertain = traj
    else:
        
        # choose most uncertain
        min_uncertainty = config.get('UQ_min_uncertainty')
        max_uncertainty = config.get('UQ_max_uncertainty')
        traj_uncertain, embeddings_uncertain = sort_by_uncertainty(traj, embeddings, UQ, max_samples, min_uncertainty, max_uncertainty)

        MLP_dict['number_MD_samples'] = len(traj_uncertain)

        checks['MD_count'] = len(traj_uncertain)<config.get('max_samples')/2
    

    print('checks: ', checks)

    config = add_checks_to_config(config, checks)

    tmp1 = time.time()
    print('Time to finish', tmp1-tmp0, 'Elapsed time ', tmp1-start, flush=True)
    tmp0 = tmp1

    logging_dict = {
        **MLP_dict,
        **checks,
        'MLP_MD_time': tmp1-start,
    }

    tmp_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('tmp_file','tmp.json'))
    write_to_tmp_dict(tmp_filename,logging_dict)

    return traj_uncertain, embeddings_uncertain, config