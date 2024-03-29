import os
import time
import numpy as np
import pandas as pd
import importlib
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, ZeroRotation, Stationary
from ase import units
from ase.md import MDLogger
from ase.io import Trajectory, read
from ase.parallel import world
from ase.calculators.calculator import Calculator
import torch.multiprocessing as mp

from ..utils.md_utils import md_func_fn
from ..sampling.utils import get_discontinuity
from ..utils.multiprocessing import starmap_with_kwargs

def md_from_atoms(
    atoms: Atoms,
    md_func_name: str = 'nvt',
    temperature: float = 300.,
    steps: int = 1000,
    timestep: float = 1,
    initialize_velocity: bool = False,
    md_func_dict: dict = {},
    dump_file: str = None,
    trajectory_file: str = None,
    delete_tmp: bool = True,
    data_directory: str = '',
    **kwargs
):
    if world.rank == 0:
        print(atoms,md_func_name,temperature)
    if isinstance(atoms.calc,dict):
        mod = importlib.import_module(atoms.calc.get('module'))
        calc_class = getattr(mod,atoms.calc.get('calculator_type'))
        calc = calc_class(**atoms.calc.get('calculator_kwargs',{}))
        atoms.calc = calc
    assert isinstance(atoms.calc,Calculator)
    print('Starting timer', flush=True)
    start = time.time()
    stable = True
    ## set random seed for multiprocessing
    ## this works well for multiprocessing but breaks GPAW
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    rind = int(np.random.rand(1)*1e6)

    if initialize_velocity:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        ZeroRotation(atoms)
        Stationary(atoms)
    if world.rank == 0:
        print(temperature,flush=True)

    md_func, md_kwargs = md_func_fn(md_func_name, temperature,timestep,cell=atoms.get_cell(),**md_func_dict)

    dyn = md_func(atoms, **md_kwargs)
    
    
    
    if dump_file is None:
        dump_file = f'MD_dump_file_{rind}.csv'
    
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
    tmp0 = time.time()
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
    if world.rank == 0:
        print('Time to run MD', tmp1-tmp0, 'Elapsed time ', tmp1-start, flush=True)
    tmp0 = tmp1

    # print('Done with MD', flush = True)
    # Check energy stability
    
    MLP_log = pd.read_csv(dump_file,delim_whitespace=True)

    MD_energies = MLP_log['Etot[eV]'].values
    max_E_index = get_discontinuity(MD_energies)

    if max_E_index < steps:
        if world.rank == 0:
            print(f'max E index {max_E_index} of {len(MLP_log)} MLP_MD_steps', flush=True)
        stable = False
        max_E_index -= 10
    else:
        if world.rank == 0:
            print(f'Total energy stable: max E index {max_E_index}', flush=True)

    MD_temperatures = MLP_log['T[K]'].values
    max_T_index = get_discontinuity(MD_temperatures)

    if max_T_index < steps:
        if world.rank == 0:
            print(f'max T index {max_T_index} of {len(MLP_log)} MLP_MD_steps', flush=True)
        stable = False
        max_T_index -= 10
    else:
        if world.rank == 0:
            print(f'Temperature stable: max T index {max_T_index}', flush=True)

    max_index = min([max_E_index,max_T_index])
    max_index = max([max_index,0])


    traj = read(trajectory_file,index=':')
    traj = traj[:max_index] # Only use E stable indices

    if delete_tmp and world.rank == 0:
        os.remove(dump_file)
        os.remove(trajectory_file)

    return traj, MLP_log, stable

def sample_md_parallel(md_kwargs: list[dict],npool:int):
    trajs = []
    logs = []
    stables = []
    with mp.Pool(npool) as p:
        for res in starmap_with_kwargs(p,md_from_atoms,md_kwargs):
            trajs.append(res[0])
            logs.append(res[1])
            stables.append(res[2])
    
    return trajs, logs, stables