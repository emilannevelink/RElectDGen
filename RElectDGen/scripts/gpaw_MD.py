import h5py, uuid, json, pdb, ase, os, argparse, time

from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units


from e3nn_networks.utils.input_output import load_learning_data, write_learning_data, extract_traj_data

from ..calculate.calculator import oracle_from_config
from ..structure.structure import structure_from_config

# from scripts.unified.structure import structure_from_config
# from scripts.unified.calculator import oracle_from_config
from ase.parallel import world

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', dest='config',
                    help='active_learning configuration file', type=str)
parser.add_argument('--MLP_config_file', dest='MLP_config',
                    help='Nequip configuration file', type=str)
args = parser.parse_args()

import yaml
# if world.rank==0:
with open(args.config,'r') as fl:
    config = yaml.load(fl,yaml.FullLoader)

structure_file = os.path.join(config.get('data_directory'),config.get('structure_file'))
if os.path.isfile(structure_file):
    supercell = read(structure_file)
else:
    supercell = structure_from_config(config)
    write(structure_file,supercell)

config['cell'] = supercell.get_cell()

trajectory_file = os.path.join(config.get('data_directory'),config.get('trajectory_file'))
data_file = os.path.join(config.get('data_directory'),config.get('hdf5_file'))
initial_MD_steps = config.get('GPAW_MD_steps')

print(data_file)
print(trajectory_file)


if os.path.isfile(trajectory_file):
    print('traj file')
    try:
        traj = Trajectory(trajectory_file)
        # print('traj file load')
        # data_write = extract_traj_data(traj)
        # write_learning_data(data_file, data_write, append=False)
    
        supercell = traj[-1]
        initial_MD_steps-=len(traj)
    except:
        print('Loading Trajectory fialed, re-initializing',flush=True)


if initial_MD_steps > 0:

    calc_oracle = oracle_from_config(config)
    supercell.calc = calc_oracle    
    MaxwellBoltzmannDistribution(supercell, temperature_K=config.get('GPAW_MD_temperature'))
    GPAW_MD_dump_file = os.path.join(config.get('data_directory'),config.get('GPAW_MD_dump_file'))
    dyn = VelocityVerlet(supercell, timestep=config.get('GPAW_MD_timestep') * units.fs, logfile=GPAW_MD_dump_file)
    traj = Trajectory(trajectory_file, 'a', supercell)
    dyn.attach(traj.write, interval=1)
    dyn.run(initial_MD_steps)
    traj.close()
    traj = Trajectory(trajectory_file)
    data_write = extract_traj_data(traj)
    write_learning_data(data_file, data_write, append=False)