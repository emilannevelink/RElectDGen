import numpy as np
from ase import units

def md_func_from_config(config,temperature=None):
    if temperature is None:
        temperature = config.get('MLP_MD_temperature')
        
    md_kwargs = {
        'timestep': config.get('MLP_MD_timestep') * units.fs
    }
    md_func_name = config.get('MD_sampling_func', 'nvt')
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
            md_kwargs['ttime'] = md_kwargs['timestep']*100
        else:
            md_kwargs['ttime'] = ttime*units.fs
        ptime = config.get('NPT_ptime')
        if ptime is None:
            md_kwargs['pfactor'] = (md_kwargs['timestep']*500)**2*0.6 #convert ptime to pfactor
        else:
            md_kwargs['pfactor'] = (ptime*units.fs)**2*0.6 #convert ptime to pfactor
        md_kwargs['mask'] = np.eye(3) # only cell vectors to change magnitude; disallow shear

    return md_func, md_kwargs