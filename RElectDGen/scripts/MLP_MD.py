import shutil
import h5py, uuid, json, pdb, ase, os, argparse, time
# from datetime import datetime
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.md import MDLogger

import yaml
import torch
# from nequip.data import AtomicData, AtomicDataDict
# from nequip.ase.nequip_calculator import NequIPCalculator
# from nequip.utils import Config

# home_directory = '/Users/emil/Google Drive/'
from ..utils.uncertainty import latent_distance_uncertainty_Nequip
# from e3nn_networks.utils.data_helpers import *

from RElectDGen.scripts.gpaw_MD import get_initial_MD_steps

from ..calculate.calculator import nn_from_results
from ..structure.segment import clusters_from_traj
from ..utils.logging import write_to_tmp_dict, UQ_params_to_dict
from ..structure.build import get_initial_structure
import time


def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    parser.add_argument('--MLP_config_file', dest='MLP_config',
                        help='Nequip configuration file', type=str)
    parser.add_argument('--loop_learning_count', dest='loop_learning_count', default=1,
                        help='active_learning_loop', type=int)
    args = parser.parse_args(argsin)

    
    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    return config, args.config, args.loop_learning_count

def main(args=None):
    print('Starting timer', flush=True)
    start = time.time()
    MLP_dict = {}
    

    config, filename_config, loop_learning_count = parse_command_line(args)
    supercell = get_initial_structure(config)
    
    #Delete Bondlength constraints
    supercell.constraints = [constraint for constraint in supercell.constraints if type(constraint)!=ase.constraints.FixBondLengths]
    

    ### Setup NN ASE calculator
    calc_nn, model, MLP_config = nn_from_results()
    torch._C._jit_set_bailout_depth(MLP_config.get("_jit_bailout_depth",2))
    supercell.calc = calc_nn

    tmp0 = time.time()
    print('Time to initialize', tmp0-start)

    ### Calibrate Uncertainty Quantification
    UQ = latent_distance_uncertainty_Nequip(model, MLP_config)
    UQ.calibrate()
    print(UQ.params,flush=True)
    UQ_dict = UQ_params_to_dict(UQ.params,'MLP')
    for key in UQ.params:
        print(key, UQ.params[key],flush=True) 
        if UQ.params[key][1] < config.get('mininmum_uncertainty_scaling',0):
            UQ.params[key][1] = config.get('mininmum_uncertainty_scaling')
            print('change sigma to minimum',flush=True)
            print(key, UQ.params[key],flush=True) 
            
    tmp1 = time.time()
    print('Time to calibrate UQ ', tmp1-tmp0, 'Elapsed time ', tmp1-start, flush=True)
    tmp0 = tmp1


    ### Run MLP MD
    MLP_dict['MLP_MD_temperature'] = config.get('MLP_MD_temperature') + (loop_learning_count-1)*config.get('MLP_MD_dT')
    MaxwellBoltzmannDistribution(supercell, temperature_K=MLP_dict['MLP_MD_temperature'])
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

    max_traj_len = config.get('max_atoms_to_segment',1001)

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

    try:
        max_index = int((uncertainty.max(axis=1).values>5*max_sigma).nonzero()[0])
    except IndexError:
        print('Index Error', uncertainty, flush=True)
        max_index = len(uncertainty)
    print('max index: ', max_index,flush=True)

    checks = [
        float(uncertainty.mean())<config.get('UQ_min_uncertainty'),
        float(uncertainty.std())<config.get('UQ_min_uncertainty')/2,
        max_index==expected_max_index,
    ]

    if max_index < 10/reduction_factor and config.get('MLP_MD_steps')>10:
        MLP_log = pd.read_csv(MLP_MD_dump_file,delim_whitespace=True)
        try:
            max_index = int(np.argwhere(MLP_log['T[K]'].values>2000)[0])
        except IndexError:
            max_index = int(config.get('MLP_MD_steps')+1)
        print(f'max index not high enough resetting to {max_index}', flush=True)
        sorted = False
        uncertainty_thresholds[0] = max([UQ.params[key][0]+UQ.params[key][1] for key in UQ.params])
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

    print('isolating uncertain clusters', flush=True)
    clusters, cluster_uncertainties = clusters_from_traj(traj, uncertainty, **config)

    tmp1 = time.time()
    print('Time to segment clusters', tmp1-tmp0, 'Elapsed time ', tmp1-start, flush=True)
    tmp0 = tmp1

    MLP_dict['number_uncertain_points'] = len(clusters)
    if len(clusters) == 0:
        print('No clusters to calculate', flush=True)
        MLP_dict['number_clusters_calculate'] = 0
        checks.append(True)
    else:
        cluster_file = os.path.join(config.get('data_directory'),config.get('GPAW_MD_dump_file').split('.log')[0]+'_clusters.xyz')
        write(cluster_file,clusters)

        uncertainties, cluster_embeddings = UQ.predict_from_traj(clusters,max=True)

        tmp1 = time.time()
        print('Time to predict cluster uncertainties', tmp1-tmp0, 'Elapsed time ', tmp1-start, flush=True)
        tmp0 = tmp1
        if sorted:
            mask = torch.logical_and(uncertainties>min_sigma,uncertainties<max_sigma)
        else:
            mask = uncertainties>min_sigma

        print(len(clusters),sum(mask),flush=True)

        cluster_uncertainties = cluster_uncertainties[mask.numpy()]
        clusters_old = copy.copy(clusters)
        clusters = [atoms for bool, atoms in zip(mask,clusters) if bool]
        cluster_embeddings = [embed for bool, embed in zip(mask,cluster_embeddings) if bool]

        print('writing uncertain clusters', flush=True)
        calc_inds = []
        embedding_distances = []
        keep_embeddings = {}
        if sum(mask)>0:
            for key in MLP_config.get('chemical_symbol_to_type'): 
                keep_embeddings[key] = torch.empty((0,cluster_embeddings[0].shape[-1]))
        for i, (traj_ind, atom_ind, uncert) in enumerate(cluster_uncertainties.values):
                if atom_ind in clusters[i].arrays['cluster_indices']:
                    embedding_all = cluster_embeddings[i]
                    ind = np.argwhere(clusters[i].arrays['cluster_indices']==atom_ind).flatten()[0]
                    
                    embeddingi_cluster = embedding_all[ind].numpy()
                    embeddingi_total = embeddings[int(traj_ind), int(atom_ind)].numpy()
                
                    embedding_distance = np.round(np.linalg.norm(embeddingi_total-embeddingi_cluster),4)

                    key = clusters[i].get_chemical_symbols()[ind]
                    if len(keep_embeddings[key])==0:
                        for key in MLP_config.get('chemical_symbol_to_type'):
                            mask = np.array(clusters[i].get_chemical_symbols()) == key
                            keep_embeddings[key] = torch.cat([keep_embeddings[key],embedding_all[mask]])
                        calc_inds.append(i)
                        embedding_distances.append(embedding_distance)
                    else:
                        UQ_dist = np.linalg.norm(keep_embeddings[key]-embeddingi_cluster,axis=1).min()*UQ.params[key][1]
                        if UQ_dist>2*config.get('UQ_min_uncertainty'):
                            for key in MLP_config.get('chemical_symbol_to_type'):
                                mask = np.array(clusters[i].get_chemical_symbols()) == key
                                keep_embeddings[key] = torch.cat([keep_embeddings[key],embedding_all[mask]])
                            calc_inds.append(i)
                            embedding_distances.append(embedding_distance)
                    
                    if len(calc_inds) >= config.get('max_samples'):
                        break
                else:
                    indices = clusters[i].arrays['cluster_indices']
                    print(f'atom_ind {atom_ind} is not in cluster {i} with indices {indices}', flush=True)
        
        MLP_dict['number_clusters_calculate'] = len(calc_inds)

        # Address first active learning loop over confidence
        if len(calc_inds) == 0 and get_initial_MD_steps(config)==-1:
            calc_inds = uncertainties.argsort()[:config.get('max_samples')]
            clusters = clusters_old

        if len(calc_inds)>0:
            print('Embedding distances: ', embedding_distances, flush=True)
            # calc_inds = [ind_sorted[0]]
            # for ind in ind_sorted[1:]:
            #     if not torch.any(torch.isclose(torch.tensor(calc_inds),torch.tensor(ind),atol=config.get('UQ_sampling_distance'))):
            #         calc_inds.append(ind) 
            #     if len(calc_inds) >= config.get('max_samples'):
            #         break
            # print(len(calc_inds), calc_inds)

            traj_calc = [clusters[i] for i in torch.tensor(calc_inds).tolist()]
            
            active_learning_configs = os.path.join(config.get('data_directory'),config.get('active_learning_configs'))
            traj_write = Trajectory(active_learning_configs,mode='w')
            [traj_write.write(atoms) for atoms in traj_calc]

            print(len(calc_inds), calc_inds)
            checks.append(len(calc_inds)<config.get('max_samples')/2)
        else:
            print('No uncertain data points')
            checks.append(True)

    print('checks: ', checks)

    if config.get('checks') is not None:
        config['checks'].append(checks)
    else:
        config['checks'] = [checks]

    with open(filename_config,'w') as fl:
        yaml.dump(config, fl)

    tmp1 = time.time()
    print('Time to finish', tmp1-tmp0, 'Elapsed time ', tmp1-start, flush=True)
    tmp0 = tmp1

    MLP_dict['check_error'] = checks[0]
    MLP_dict['check_std'] = checks[1]
    MLP_dict['check_steps'] = checks[2]
    MLP_dict['check_npoints'] = checks[3]

    logging_dict = {
        **UQ_dict,
        **MLP_dict,
        'MLP_MD_time': tmp1-start,
    }

    tmp_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('tmp_file','tmp.json'))
    write_to_tmp_dict(tmp_filename,logging_dict)

if __name__ == "__main__":
    main()