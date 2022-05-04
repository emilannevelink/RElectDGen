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
from ..utils import uncertainty
from ..uncertainty import models as uncertainty_models
# from ..utils.uncertainty import latent_distance_uncertainty_Nequip_adversarial, latent_distance_uncertainty_Nequip_adversarialNN
# from e3nn_networks.utils.data_helpers import *

from RElectDGen.scripts.gpaw_MD import get_initial_MD_steps

from ..calculate.calculator import nn_from_results
from ..structure.segment import clusters_from_traj
from ..utils.logging import write_to_tmp_dict, add_checks_to_config
from ..structure.build import get_initial_structure
from .downsample import sort_by_uncertainty
import time


def adv_loss(atoms, UQ, T):
    data = UQ.transform(AtomicData.from_ase(atoms=atoms,r_max=UQ.r_max, self_interaction=UQ.self_interaction))
    data['pos'].requires_grad = True
    
    out = UQ.adversarial_loss(data, T)
    return out

def d_min_func(positions, UQ, atoms, T):
            
        atoms.set_positions(
            positions.reshape(atoms.positions.shape)
        )
        data = UQ.transform(AtomicData.from_ase(atoms=atoms,r_max=UQ.r_max, self_interaction=UQ.self_interaction))
        data['pos'].requires_grad = True
        
        loss = -UQ.adversarial_loss(data, T)

        grads = torch.autograd.grad(loss,data['pos'])

        return grads[0].flatten().numpy()

def min_func(positions, UQ, atoms, T):
    atoms.set_positions(
        positions.reshape(atoms.positions.shape)
    )

    neg_loss = -adv_loss(atoms, UQ, T)

    return neg_loss.detach().numpy()

def adv_sampling(config, traj_initial, loop_learning_count=1):

    start = time.time()
    adv_dict = {}
    
    ### Setup NN ASE calculator
    calc_nn, model, MLP_config = nn_from_results()
    
    tmp0 = time.time()
    print('Time to initialize', tmp0-start)

    UQ_func = getattr(uncertainty_models,config.get('uncertainty_function', 'Nequip_latent_distance'))

    UQ = UQ_func(model, config, MLP_config)
    UQ.calibrate()
            
    tmp1 = time.time()
    print('Time to calibrate UQ ', tmp1-tmp0, 'Elapsed time ', tmp1-start, flush=True)
    tmp0 = tmp1
    
    adv_dict['MLP_adv_temperature'] = config.get('MLP_adv_temperature') + (loop_learning_count-1)*config.get('MLP_adv_dT')
    T = adv_dict['MLP_adv_temperature']

    max_samples = int(config.get('max_samples'))
    n_adversarial_samples = int(config.get('n_adversarial_samples',2*max_samples))
    
    traj_indices = torch.randperm(len(traj_initial))[:2*n_adversarial_samples]
    adv_losses = []
    for i in traj_indices:
        atoms = copy.deepcopy(traj_initial[i])
        atoms.positions += 0.01*(np.random.rand(*atoms.positions.shape)-0.5)
        loss = adv_loss(atoms, UQ, T)
        adv_losses.append(float(loss))

    sort_indices = torch.argsort(torch.tensor(adv_losses), descending=True) #sort highest to lowest
    traj_indices = traj_indices[sort_indices[:n_adversarial_samples]]
    
    max_displacement = config.get('maximum_adversarial_displacement', 1)
    traj_adv = []
    embeddings = []
    uncertainties = []
    positions_differences = []
    for i in traj_indices:
        record = True
        atoms = copy.deepcopy(traj_initial[i])
        
        positions = atoms.get_positions().flatten()
        positions += 0.01*(np.random.rand(*positions.shape)-0.5)

        res = minimize(min_func,positions,args=(UQ, atoms, T), jac=d_min_func, method='CG')

        print(res)

        atoms.set_positions(
                res.x.reshape(atoms.positions.shape)
            )
        atoms_save = copy.deepcopy(atoms)
        
        positions_differences.append(np.max(np.abs(atoms.positions-traj_initial[i].positions)))
        # print(i, atoms.positions-traj_initial[i].positions, flush=True)
        # print(atoms.positions, flush = True)
        # print(traj_initial[i].positions, flush = True)
        
        if positions_differences[-1]<max_displacement:
        
            val_adv_loss = adv_loss(atoms_save, UQ, T)
            uncertainties.append(torch.tensor(UQ.uncertainties).mean())
            embeddings.append(torch.tensor(UQ.atom_embedding))
            traj_adv.append(atoms_save)

    print(len(uncertainties), flush=True)
    if len(uncertainties)>0:
        uncertainties = torch.tensor(uncertainties).numpy()
        checks = {
            'adv_mean_uncertainty': float(uncertainties.mean())<config.get('UQ_min_uncertainty'),
            'adv_std_uncertainty': float(uncertainties.std())<config.get('UQ_min_uncertainty')/2,
        }
    else:
        checks = {
            'adv_mean_uncertainty': False,
            'adv_std_uncertainty': False,
        }

    traj_dump_file = os.path.join(config.get('data_directory'),config.get('adv_trajectory_file'))
    writer = Trajectory(traj_dump_file, 'w')
    for atoms in traj_adv:
        writer.write(atoms)

    adv_dict['number_uncertain_points'] = len(traj_adv)
    traj_uncertain, embeddings_uncertain = sort_by_uncertainty(traj_adv, embeddings, UQ, max_samples)

    adv_dict['number_adversarial_samples'] = len(traj_uncertain)

    checks['adv_count'] = len(traj_uncertain)<config.get('max_samples')/2
    checks['adv_position_difference'] = float(np.max(positions_differences)) < config.get('minimum_position_difference', 0.05)

    print('checks: ', checks)

    config = add_checks_to_config(config, checks)

    tmp1 = time.time()
    print('Time to finish', tmp1-tmp0, 'Elapsed time ', tmp1-start, flush=True)
    tmp0 = tmp1

    logging_dict = {
        **adv_dict,
        **checks,
        'MLP_MD_time': tmp1-start,
    }

    tmp_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('tmp_file','tmp.json'))
    write_to_tmp_dict(tmp_filename,logging_dict)

    return traj_uncertain, embeddings_uncertain, config