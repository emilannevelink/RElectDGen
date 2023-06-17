import os
import time
import numpy as np
import pandas as pd
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, ZeroRotation, Stationary
from ase import units
from ase.md import MDLogger
from ase.io import Trajectory
from ase.parallel import world

from ..utils.md_utils import md_func_fn

def md_from_atoms(
    atoms: Atoms,
    md_func_name: str = 'nvt',
    temperature: float = 300.,
    steps: int = 1000,
    timestep: float = 0.001,
    initialize_velocity: bool = False,
    md_func_dict: dict = {},
    dump_file: str = None,
    trajectory_file: str = None,
    delete_tmp: bool = True,
    data_directory: str = '',
    **kwargs
):  
    print('Starting timer', flush=True)
    start = time.time()
    stable = True
    ## set random seed for multiprocessing
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    rind = int(np.random.rand(1)*1e6)

    if initialize_velocity:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        ZeroRotation(atoms)
        Stationary(atoms)
        print(atoms.get_velocities())
        print(atoms.get_masses())

    print(temperature,flush=True)

    md_func, md_kwargs = md_func_fn(md_func_name, temperature,timestep,**md_func_dict)

    dyn = md_func(atoms, **md_kwargs)
    
    tmp0 = time.time()
    
    if dump_file is None:
        dump_file = f'MD_dump_file_{rind}.csv'
    print(data_directory)
    dump_file = os.path.join(data_directory,dump_file)

    #MDLogger only has append, delete log file
    if os.path.isfile(dump_file) and world.rank == 0:
        os.remove(dump_file)
    dyn.attach(MDLogger(dyn,atoms,dump_file,mode='w'),interval=1)
    
    if trajectory_file is None:
        trajectory_file = f'MD_traj_file_{rind}.traj'
    trajectory_file = os.path.join(data_directory,trajectory_file)

    traj = Trajectory(trajectory_file, 'w', atoms)
    dyn.attach(traj.write, interval=1)
    
    if 'npt' in str(type(dyn)).lower():
        steps += 1 # Fix different number of steps between NVE / NVT and NPT
    try:
        dyn.run(steps)
    except (ValueError, RuntimeError) as exception:
        print(exception)
        print('Value Error: MLP isnt good enough for current number of steps')
        stable = False
    
    traj.close()

    tmp1 = time.time()
    print('Time to run MD', tmp1-tmp0, 'Elapsed time ', tmp1-start, flush=True)
    tmp0 = tmp1

    # print('Done with MD', flush = True)
    # Check temperature stability
    MLP_log = pd.read_csv(dump_file,delim_whitespace=True)
    try:
        MD_energies = MLP_log['Etot[eV]'].values
        MD_e0 = MD_energies[0]
        max_E_index = int(np.argwhere(np.abs((MD_energies-MD_e0)/MD_e0)>1)[0])
    except IndexError:
        max_E_index = int(steps+1)

    if max_E_index < steps:
        print(f'max E index {max_E_index} of {len(MLP_log)} MLP_MD_steps', flush=True)
        stable = False
    else:
        print(f'Total energy stable: max E index {max_E_index}', flush=True)

    traj = Trajectory(trajectory_file)
    traj = traj[:max_E_index] # Only use E stable indices

    if delete_tmp:
        os.remove(dump_file)
        os.remove(trajectory_file)

    return traj, stable