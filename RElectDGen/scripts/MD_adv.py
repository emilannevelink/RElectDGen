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

from scipy.optimize import minimize

import yaml
import torch
from nequip.data import AtomicData, AtomicDataDict
# from nequip.ase.nequip_calculator import NequIPCalculator
# from nequip.utils import Config

# home_directory = '/Users/emil/Google Drive/'
# from ..utils.uncertainty import latent_distance_uncertainty_Nequip_adversarial, latent_distance_uncertainty_Nequip_adversarialNN
# from e3nn_networks.utils.data_helpers import *

from RElectDGen.scripts.gpaw_MD import get_initial_MD_steps

from ..calculate._MLIP import nn_from_results, nns_from_results
from ..structure.segment import clusters_from_traj
from ..utils.logging import write_to_tmp_dict, add_checks_to_config
from ..structure.build import get_initial_structure
import time

from ..sampling.utils import sort_by_uncertainty
from ..uncertainty import models as uncertainty_models
from ..sampling.md_sampling import MD_sampling
from ..sampling.adv_sampling import adv_sampling

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
    

    MD_uncertain, MD_embedding_uncertain, config = MD_sampling(config, loop_learning_count)
    sort_by_type = config.get('mdadv_sort_type', 'sort_by_uncertainty')
    max_samples = int(config.get('max_samples'))
    if len(MD_uncertain)>=max_samples:
        adv_uncertain = []
        adv_embedding_uncertain = []
        checks = {
            'adv_mean_uncertainty': False,
            'adv_std_uncertainty': False,
            'adv_position_difference': False,
            'adv_count': False
        }
        config = add_checks_to_config(config, checks)
    else:
        adv_uncertain, adv_embedding_uncertain, config = adv_sampling(config, MD_uncertain, loop_learning_count)

    uncertain = MD_uncertain + adv_uncertain
    embeddings = MD_embedding_uncertain + adv_embedding_uncertain

    
    if sort_by_type == 'sort_by_uncertainty':
        train_directory = config.get('train_directory','results')
        if train_directory[-1] == '/':
            train_directory = train_directory[:-1]
        
        uncertainty_function = config.get('uncertainty_function', 'Nequip_latent_distance')
        ### Setup NN ASE calculator
        if uncertainty_function in ['Nequip_ensemble']:
            n_ensemble = config.get('n_uncertainty_ensembles',4)
            calc_nn, model, MLP_config = nns_from_results(train_directory,n_ensemble)
            r_max = MLP_config[0].get('r_max')
        else:
            calc_nn, model, MLP_config = nn_from_results()
            r_max = MLP_config.get('r_max')
        

        UQ_func = getattr(uncertainty_models,uncertainty_function)

        UQ = UQ_func(model, config, MLP_config)
        UQ.calibrate()

        
        min_uncertainty = config.get('UQ_min_uncertainty')
        max_uncertainty = np.inf # don't remove any previous samples config.get('UQ_max_uncertainty')*config.get('adversarial_max_UQ_factor', 1)  # to not remove the adversarial samples
        traj_uncertain, traj_embedding, calc_inds_uncertain = sort_by_uncertainty(uncertain, embeddings, UQ, max_samples, min_uncertainty, max_uncertainty)

    else:
        
        traj_uncertain = uncertain[:max_samples]
        traj_embedding = embeddings[:max_samples]
        
        calc_inds_uncertain = np.arange(len(traj_uncertain),dtype=int).tolist()
        
        print(calc_inds_uncertain)
        print(len(traj_uncertain),traj_uncertain)
    
    config['calc_inds_uncertain'] = calc_inds_uncertain
    config['n_MD_uncertain'] = len(MD_uncertain)

    if len(traj_uncertain)>0:   
        active_learning_configs = os.path.join(config.get('data_directory'),config.get('active_learning_configs'))
        traj_write = Trajectory(active_learning_configs,mode='w')
        [traj_write.write(atoms) for atoms in traj_uncertain]

    else:
        print('No uncertain data points')
    
    checks = {
        'sampling_count': len(traj_uncertain)<=config.get('number_of_samples_check_value',config.get('max_samples')/2)
    }
    
    config = add_checks_to_config(config, checks)

    with open(filename_config,'w') as fl:
        yaml.dump(config, fl)

    tmp1 = time.time()
    logging_dict = {
        **checks,
        'sampling_time': tmp1-start,
    }

    tmp_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('tmp_file','tmp.json'))
    write_to_tmp_dict(tmp_filename,logging_dict)

if __name__ == "__main__":
    main()