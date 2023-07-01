import os
import pandas as pd
import h5py
import numpy as np
from ase import units

def md_func_from_config(config,temperature=None,prefix='MLP'):
    if temperature is None:
        temperature = config.get(f'{prefix}_MD_temperature')
        
    md_kwargs = {
        'timestep': config.get(f'{prefix}_MD_timestep') * units.fs
    }
    md_func_name = config.get('MD_sampling_func', 'nvt')
    if 'npt' in md_func_name and 'GPAW' in prefix:
        print('GPAW doesnt support stresses, reverting to NVT')
        md_func_name = 'nvt'
    if md_func_name == 'nve':
        from ase.md.verlet import VelocityVerlet as md_func
    elif md_func_name == 'nvt':
        from ase.md.nvtberendsen import NVTBerendsen as md_func
        taut = config.get('NVT_taut')
        if taut is None:
            md_kwargs['taut'] = md_kwargs['timestep']*500
        else:
            md_kwargs['taut'] = taut*units.fs
        md_kwargs['temperature'] = temperature
    elif md_func_name == 'npt':
        from ase.md.npt import NPT as md_func
        md_kwargs['temperature_K'] = temperature
        md_kwargs['externalstress'] = np.array(config.get('external_stress')) * 1.01325e-4/160.21766
        ttime = config.get('NPT_ttime')
        if ttime is None:
            md_kwargs['ttime'] = md_kwargs['timestep']*500
        else:
            md_kwargs['ttime'] = ttime*units.fs
        ptime = config.get('NPT_ptime')
        if ptime is None:
            md_kwargs['pfactor'] = (md_kwargs['timestep']*1000)**2*0.6 #convert ptime to pfactor
        else:
            md_kwargs['pfactor'] = (ptime*units.fs)**2*0.6 #convert ptime to pfactor
        md_kwargs['mask'] = np.eye(3) # only cell vectors to change magnitude; disallow shear

    return md_func, md_kwargs

def md_func_fn(
    md_func_name,
    temperature,
    timestep,
    **kwargs
    ):
    md_kwargs = {
        'timestep': timestep * units.fs
    }
    if md_func_name == 'nve':
        from ase.md.verlet import VelocityVerlet as md_func
    elif md_func_name == 'nvt':
        # from ase.md.nvtberendsen import NVTBerendsen as md_func
        # taut = kwargs.get('NVT_taut')
        # if taut is None:
        #     md_kwargs['taut'] = timestep*500
        # else:
        #     md_kwargs['taut'] = taut*units.fs
        # md_kwargs['temperature'] = temperature
        from ase.md.npt import NPT as md_func
        md_kwargs['temperature_K'] = temperature
        ttime = kwargs.get('NVT_ttime')
        if ttime is None:
            md_kwargs['ttime'] = md_kwargs['timestep']*500
        else:
            md_kwargs['ttime'] = ttime*units.fs
        md_kwargs['externalstress'] = 0
        # print(md_kwargs)
        
    elif md_func_name == 'npt':
        from ase.md.npt import NPT as md_func
        md_kwargs['temperature_K'] = temperature
        md_kwargs['externalstress'] = np.array(kwargs.get('external_stress')) * 1.01325e-4/160.21766
        ttime = kwargs.get('NPT_ttime')
        if ttime is None:
            md_kwargs['ttime'] = md_kwargs['timestep']*500
        else:
            md_kwargs['ttime'] = ttime*units.fs
        ptime = kwargs.get('NPT_ptime')
        if ptime is None:
            md_kwargs['pfactor'] = (md_kwargs['timestep']*1000)**2*0.6 #convert ptime to pfactor
        else:
            md_kwargs['pfactor'] = (ptime*units.fs)**2*0.6 #convert ptime to pfactor
        md_kwargs['mask'] = np.eye(3) # only cell vectors to change magnitude; disallow shear

    return md_func, md_kwargs

def save_log_to_hdf5(MLP_log,dump_hdf5_filename,stable):
    if not stable:
        
        with h5py.File(dump_hdf5_filename,'a') as hf:
            id = str(len(hf.keys())+1)
            gr = hf.create_group(id)
            gr.create_dataset('time',data=MLP_log['Time[ps]'].values)
            gr.create_dataset('energies',data=MLP_log['Etot[eV]'].values)
            gr.create_dataset('temperatures',data=MLP_log['T[K]'].values)