import ase

from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, ZeroRotation, Stationary
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.md import MDLogger


from RElectDGen.utils.md_utils import md_func_from_config
from ..calculate.calculator import nn_from_results
from ..structure.build import get_initial_structure
import time

import os
import argparse

from ase.io.trajectory import Trajectory
import yaml


def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    args = parser.parse_args(argsin)

    
    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    return config

def main(args=None):

    config = parse_command_line(args)
    
    print('Starting timer', flush=True)
    start = time.time()
    
    supercell = get_initial_structure(config)
    
    #Delete Bondlength constraints
    supercell.constraints = [constraint for constraint in supercell.constraints if type(constraint)!=ase.constraints.FixBondLengths]
    
    train_directory = config.get('train_directory')
    
    ### Setup NN ASE calculator
    calc_nn, model, MLP_config = nn_from_results(train_directory)
    supercell.calc = calc_nn

    tmp0 = time.time()
    print('Time to initialize', tmp0-start)


    ### Run MLP MD
    MaxwellBoltzmannDistribution(supercell, temperature_K=config.get('MLP_MD_temperature'))
    ZeroRotation(supercell)
    Stationary(supercell)

    md_func, md_kwargs = md_func_from_config(config)

    dyn = md_func(supercell, **md_kwargs)
    MLP_MD_dump_file = os.path.join(config.get('data_directory'),config.get('MLP_MD_dump_file'))
    
    #MDLogger only has append, delete log file
    if os.path.isfile(MLP_MD_dump_file):
        os.remove(MLP_MD_dump_file)
    dyn.attach(MDLogger(dyn,supercell,MLP_MD_dump_file,mode='w'),interval=1)
    
    trajectory_file = os.path.join(config.get('data_directory'),config.get('MLP_trajectory_file'))
    traj = Trajectory(trajectory_file, 'w', supercell)
    dyn.attach(traj.write, interval=int(config.get('MLP_MD_dump_interval',1)))
    try:
        dyn.run(config.get('MLP_MD_steps'))
    except ValueError:
        print('Value Error: MLP isnt good enough for current number of steps')
    traj.close()

if __name__ == "__main__":
    main()
