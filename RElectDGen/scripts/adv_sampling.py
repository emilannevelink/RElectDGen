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
from sentry_sdk import flush

import yaml
import torch
from nequip.data import AtomicData, AtomicDataDict
# from nequip.ase.nequip_calculator import NequIPCalculator
# from nequip.utils import Config

# home_directory = '/Users/emil/Google Drive/'
from ..utils.uncertainty import latent_distance_uncertainty_Nequip_adversarial
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
    MLP_config['params_func'] = config.get('params_func','optimize2params')
    UQ = latent_distance_uncertainty_Nequip_adversarial(model, MLP_config)
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

    traj = read(MLP_config.get('dataset_file_name'), index=':')
    n_adversarial_samples = int(config.get('n_adversarial_samples',2*config.get('max_samples')))
    traj_indices = torch.randperm(len(traj))[:n_adversarial_samples]

    MLP_dict['MLP_MD_temperature'] = config.get('MLP_MD_temperature') + (loop_learning_count-1)*config.get('MLP_MD_dT')

    T = MLP_dict['MLP_MD_temperature']
    adversarial_learning_rate = config.get('adversarial_learning_rate')
    traj_updated = []
    embeddings = []
    for i in traj_indices:
        atoms = traj[i]
        
        # print(i, atoms.get_positions())
        adv_updates = config.get('adversarial_steps', 100)
        for _ in range(adv_updates):
            data = UQ.transform(AtomicData.from_ase(atoms=atoms,r_max=UQ.r_max, self_interaction=UQ.self_interaction))
            data['pos'].requires_grad = True
            
            adv_loss = UQ.adversarial_loss(data, T)
            
            grads = torch.autograd.grad(adv_loss,data['pos'])
            
            
            atoms.set_positions(
                atoms.get_positions() + adversarial_learning_rate*grads[0].cpu().numpy()
            )
        
        # print(grads[0])
        # print(atoms.get_positions())
        embeddings.append(UQ.atom_embedding)
        traj_updated.append(atoms)

 
    print('writing uncertain clusters', flush=True)
    calc_inds = []
    uncertainties = []
    # embedding_distances = {}
    keep_embeddings = {}
    embeddings_all = {}
    for key in MLP_config.get('chemical_symbol_to_type'): 
        keep_embeddings[key] = torch.empty((0,UQ.train_embeddings[key].shape[-1]))
    for i, (embedding_i, atoms) in enumerate(zip(embeddings,traj_updated)):
        
        # active_uncertainty = []
        # for key in MLP_config.get('chemical_symbol_to_type'): 
        #     embeddings_all[key] = torch.cat([UQ.train_embeddings[key].detach(), keep_embeddings[key]])
        #     mask = np.array(atoms.get_chemical_symbols()) == key
        #     embedding_distance = torch.cdist(embeddings_all[key],embedding_i[mask],p=2).numpy().min(axis=0)


        #     active_uncertainty.append(UQ.params[0] + embedding_distance*UQ.params[1])
        data = UQ.transform(AtomicData.from_ase(atoms=atoms,r_max=UQ.r_max, self_interaction=UQ.self_interaction))
        active_uncertainty = UQ.predict_uncertainty(data['atom_types'], embedding_i, extra_embeddings=keep_embeddings).detach().cpu().numpy()

        if np.any(active_uncertainty>config.get('UQ_min_uncertainty')):
            calc_inds.append(i)
            uncertainties.append(active_uncertainty.max())
            for key in MLP_config.get('chemical_symbol_to_type'): 
                mask = np.array(atoms.get_chemical_symbols()) == key
                keep_embeddings[key] = torch.cat([keep_embeddings[key],embedding_i[mask]])

    checks = [False, False, False] # Keep for restart

    print(uncertainties, flush = True)
    MLP_dict['number_clusters_calculate'] = len(calc_inds)

    # Address first active learning loop over confidence
    if len(calc_inds) == 0 and get_initial_MD_steps(config)==-1:
        calc_inds = np.arange(config.get('max_samples'),dtype=int)

    if len(calc_inds)>0:
        traj_calc = [traj_updated[i] for i in torch.tensor(calc_inds).tolist()]
        
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