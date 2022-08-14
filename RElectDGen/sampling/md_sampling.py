import ase
import os
import copy
import numpy as np
import pandas as pd

from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, ZeroRotation, Stationary
from ase import units
from ase.md import MDLogger
from RElectDGen.utils.io import add_to_trajectory

from RElectDGen.utils.md_utils import md_func_from_config

from ..uncertainty import models as uncertainty_models

from ..calculate.calculator import nn_from_results
from ..utils.logging import write_to_tmp_dict, add_checks_to_config
from ..structure.build import get_initial_structure
import time
from .utils import sort_by_uncertainty
from ..sampling.utils import sample_from_dataset, sample_from_initial_structures
from ..structure.segment import clusters_from_traj

def MD_sampling(config, loop_learning_count=1):
    print('Starting timer', flush=True)
    start = time.time()
    MLP_dict = {}
    
    supercell = get_initial_structure(config)
    traj_initial = sample_from_dataset(config)
    if (
        len(traj_initial)>0 and 
        not (config.get('cluster', False) or len(traj_initial[0])<len(supercell)) and 
        not config.get('MD_from_initial', False)): 
        supercell = traj_initial[0] #ensure you only sample from md
        initialize_velocity=False #velocity already in traj
    else:
        supercell = sample_from_initial_structures(config)
        initialize_velocity=True
    
    
    #Delete Bondlength constraints
    supercell.constraints = [constraint for constraint in supercell.constraints if type(constraint)!=ase.constraints.FixBondLengths]
    

    ### Setup NN ASE calculator
    calc_nn, model, MLP_config = nn_from_results()
    supercell.calc = calc_nn

    tmp0 = time.time()
    print('Time to initialize', tmp0-start,flush=True)

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
    if initialize_velocity:
        MaxwellBoltzmannDistribution(supercell, temperature_K=MLP_dict['MLP_MD_temperature'])
        ZeroRotation(supercell)
        Stationary(supercell)

    print(MLP_dict['MLP_MD_temperature'],flush=True)

    md_func, md_kwargs = md_func_from_config(config, MLP_dict['MLP_MD_temperature'])

    dyn = md_func(supercell, **md_kwargs)
    MLP_MD_dump_file = os.path.join(config.get('data_directory'),config.get('MLP_MD_dump_file'))
    #MDLogger only has append, delete log file
    if os.path.isfile(MLP_MD_dump_file):
        os.remove(MLP_MD_dump_file)
    dyn.attach(MDLogger(dyn,supercell,MLP_MD_dump_file,mode='w'),interval=1)
    
    trajectory_file = os.path.join(config.get('data_directory'),config.get('MLP_trajectory_file'))
    traj = Trajectory(trajectory_file, 'w', supercell)
    dyn.attach(traj.write, interval=1)
    nsteps = config.get('MLP_MD_steps')
    if 'npt' in str(type(dyn)).lower():
        nsteps += 1 # Fix different number of steps between NVE / NVT and NPT
    try:
        dyn.run(nsteps)
    except (ValueError, RuntimeError) as e:
        print(e)
        print('Value Error: MLP isnt good enough for current number of steps')
    traj.close()

    tmp1 = time.time()
    print('Time to run MD', tmp1-tmp0, 'Elapsed time ', tmp1-start, flush=True)
    tmp0 = tmp1

    # print('Done with MD', flush = True)
    # Check temperature stability
    MLP_log = pd.read_csv(MLP_MD_dump_file,delim_whitespace=True)
    try:
        MD_energies = MLP_log['Etot[eV]'].values
        MD_e0 = MD_energies[0]
        max_E_index = int(np.argwhere(np.abs((MD_energies-MD_e0)/MD_e0)>1)[0])
    except IndexError:
        max_E_index = int(config.get('MLP_MD_steps')+1)

    if max_E_index < config.get('MLP_MD_steps'):
        print(f'max E index {max_E_index} of {len(MLP_log)} MLP_MD_steps', flush=True)
    else:
        print(f'Total energy stable: max E index {max_E_index}', flush=True)

    traj = Trajectory(trajectory_file)
    final_supercell = copy.deepcopy(traj[-1])
    traj = traj[:max_E_index] # Only use E stable indices
    MLP_dict['MLP_MD_steps'] = len(traj)

    max_samples = int(config.get('max_samples'))
    max_traj_len = config.get('max_atoms_to_segment',2*max_samples)

    expected_max_index = config.get('MLP_MD_steps')+1
    reduction_factor = 1
    if len(traj) > max_traj_len:
        reduction_factor = np.ceil(len(traj)/max_traj_len).astype(int)
        expected_max_index = int(np.ceil(expected_max_index/reduction_factor))
        traj = traj[::reduction_factor]
        print(f'reduced length of trajectory by {reduction_factor}, new length {len(traj)}, new max_index {expected_max_index}', flush=True)

    uncertainty, embeddings = UQ.predict_from_traj(traj,max=False)

    max_val_ind = uncertainty.sum(dim=-1).max(axis=1)
    print(max_val_ind.values)
    print(max_val_ind.indices)
    print(np.array(traj[0].get_chemical_symbols())[max_val_ind.indices])
    MLP_dict['MLP_error'] = float(uncertainty.sum(dim=-1).mean())
    MLP_dict['MLP_error_std'] = float(uncertainty.sum(dim=-1).std())
    MLP_dict['MLP_error_base'] = float(uncertainty[:,:,0].mean())
    MLP_dict['MLP_error_base_std'] = float(uncertainty[:,:,0].std())
    MLP_dict['MLP_error_basestd'] = float(uncertainty[:,:,1].mean())
    MLP_dict['MLP_error_basestd_std'] = float(uncertainty[:,:,1].std())

    print('MLP error value', MLP_dict['MLP_error'], flush=True)
    print('MLP error std', MLP_dict['MLP_error_std'],flush=True)

    min_sigma = config.get('UQ_min_uncertainty')
    max_sigma = config.get('UQ_max_uncertainty')

    uncertainty_thresholds = [max_sigma, min_sigma]
    config['uncertainty_thresholds'] = uncertainty_thresholds

    try:
        max_index = int((uncertainty.sum(dim=-1).max(axis=1).values>5*max_sigma).nonzero()[0])
    except IndexError:
        print('Index Error', uncertainty, flush=True)
        max_index = len(uncertainty)
    print('max index: ', max_index,flush=True)

    checks = {
        'MD_mean_uncertainty': MLP_dict['MLP_error']<config.get('UQ_min_uncertainty'),
        'MD_std_uncertainty': MLP_dict['MLP_error_std']<config.get('UQ_min_uncertainty')/2,
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

    # Cluster atoms objects that are too large
    if len(supercell) > config.get('max_atoms_to_segment',np.inf):
        print('isolating uncertain clusters', flush=True)
        uncertainty_sum = uncertainty.sum(axis=-1) #reduce err and std to single value
        traj, undertainty_df, embeddings = clusters_from_traj(traj, uncertainty_sum, embeddings, **config)

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
        traj_uncertain, embeddings_uncertain, calc_inds_uncertain = sort_by_uncertainty(traj, embeddings, UQ, max_samples, min_uncertainty, max_uncertainty)

        MLP_dict['number_MD_samples'] = len(traj_uncertain)

        checks['MD_count'] = len(traj_uncertain)<=config.get('number_of_samples_check_value',config.get('max_samples')/2)
    
    ## Add last supercell to initial structures if no atoms were uncertain and
    ## at maximum timestep
    if (
        len(traj_uncertain)==0 and checks['MD_max_index'] and
        config.get('MLP_MD_steps') >= config.get('max_MLP_MD_steps',4000)
    ):
        if config.get('initial_structures_file') is not None:
            print('Adding to initial structures')
            initial_structures_filename = os.path.join(
                config.get('data_directory'),
                config.get('initial_structures_file')
            )
            add_to_trajectory(final_supercell,initial_structures_filename)

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