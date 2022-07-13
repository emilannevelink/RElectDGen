import argparse, subprocess
from ase.parallel import world
import operator
import numpy as np
import yaml

def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    parser.add_argument('--MLP_config_file', dest='MLP_config',
                        help='Nequip configuration file', type=str)
    args = parser.parse_args(argsin)

    

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    return config, args.config

def read_file(file, keyword, value, operation):
    with open(file,'r') as fl:
        lines = fl.readlines()

    for line in lines:
        if keyword in line:
            state = float(line.split(' ')[-1])

            return operation(state,value)

    return False

def main(args = None):

    config, filename_config = parse_command_line(args)

    job_info = config.get('job_info')
    job_ids = job_info['job_ids']
    job_types = job_info['job_types']
    print(job_ids,flush=True)

    # MLP_MD_inds = [i for i, job_type in enumerate(job_types) if 'MLP_MD' in job_type]

    # final_MLP = f'logs/output.{job_ids[MLP_MD_inds[-1]]}'

    
    checks = config.get('checks', {})
    config['checks'] = {}

    #default termination limits
    max_MLP_MD_temperature = config.get('max_MLP_temperature',350)
    max_MLP_MD_dT = config.get('max_MLP_MD_dT',100)
    max_MLP_MD_steps = config.get('max_MLP_MD_steps',4000)
    max_adv_temperature = config.get('max_adv_temperature',100000)
    max_MLP_adv_dT = config.get('max_MLP_adv_dT',50000)
    min_UQ_min_uncertainty = config.get('UQ_terminal_accuracy',0.04)

    termination_conditions = [
        config['MLP_MD_temperature']>=max_MLP_MD_temperature,
        config['MLP_MD_dT']>=max_MLP_MD_dT,
        config['MLP_MD_steps']>=max_MLP_MD_steps,
        config['MLP_adv_temperature']>=max_adv_temperature,
        config['MLP_adv_dT']>=max_MLP_adv_dT,
        config.get('UQ_min_uncertainty')<=min_UQ_min_uncertainty,
    ]

    print('checks', checks)
    if len(checks) == 0:
        raise RuntimeError('Sampling Error, checks is empty')
    
    if np.all(checks.get('MD_mean_uncertainty',[False])+checks.get('MD_std_uncertainty',[False])):
        config['MLP_MD_temperature']*=2
        config['MLP_MD_dT']*=1.5
    if np.all(checks.get('MD_max_index',[False])):
        config['MLP_MD_steps']*=2
    

    if (np.all(checks.get('adv_mean_uncertainty',[False])+checks.get('adv_std_uncertainty',[False])) or
        np.all(checks.get('adv_position_difference',[False]))):
        config['MLP_adv_temperature']*=2
        config['MLP_adv_dT']*=2

    # Reset to limits
    
    n_extrema = 0
    if config.get('MLP_MD_temperature', max_MLP_MD_temperature) >= max_MLP_MD_temperature:
        config['MLP_MD_temperature'] = max_MLP_MD_temperature
        n_extrema+=1
    if config.get('MLP_MD_dT', max_MLP_MD_dT) >= max_MLP_MD_dT:
        config['MLP_MD_dT'] = max_MLP_MD_dT
        n_extrema+=1
    if config.get('MLP_MD_steps', max_MLP_MD_steps) >= max_MLP_MD_steps:
        config['MLP_MD_steps'] = max_MLP_MD_steps
        n_extrema+=1
    if config.get('MLP_adv_temperature',max_adv_temperature) >= max_adv_temperature:
        config['MLP_adv_temperature']=max_adv_temperature
        n_extrema+=1
    if config.get('MLP_adv_dT', max_MLP_adv_dT) >= max_MLP_adv_dT:
        config['MLP_adv_dT']=max_MLP_adv_dT
        n_extrema+=1
    
    n_extrema_lower_uncertainty = config.get('n_extrema_lower_uncertainty', 0)
    if (
        n_extrema >= n_extrema_lower_uncertainty and 
        np.all(checks.get('MD_count',[True])) and np.all(checks.get('adv_count',[True])) and np.all(checks.get('sampling_count',[True]))
        ):
        config['UQ_min_uncertainty']/=2

    
    if config.get('UQ_min_uncertainty', min_UQ_min_uncertainty) <= min_UQ_min_uncertainty:
        config['UQ_min_uncertainty'] = min_UQ_min_uncertainty

    checks_history = config.get('checks_history',[])
    checks_history.append(np.all(list(checks.values()),axis=1).tolist())
    config['checks_history'] = checks_history
    print('termination conditions', termination_conditions)
    if config.get('restart',False):
        if not np.all(termination_conditions) or not np.all(checks):
            config['i_temperature_sweep']+=1
            with open(filename_config,'w') as fl:
                yaml.dump(config, fl)
            
            commands = ['REDGEN-start', filename_config]
            process = subprocess.run(commands,capture_output=True)
            print('Job ids',process.stdout)
            print('Error',process.stderr)

        else:
            print(f'reached termination condition {termination_conditions}')
    else:
        print('restart is False')

if __name__ == "__main__":
    main()
                