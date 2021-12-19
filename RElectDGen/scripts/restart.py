import argparse, subprocess
from ase.parallel import world
import operator
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', dest='config',
                    help='active_learning configuration file', type=str)
parser.add_argument('--MLP_config_file', dest='MLP_config',
                    help='Nequip configuration file', type=str)
args = parser.parse_args()

import yaml

def read_file(file, keyword, value, operation):
    with open(file,'r') as fl:
        lines = fl.readlines()

    for line in lines:
        if keyword in line:
            state = float(line.split(' ')[-1])

            return operation(state,value)

    return False

with open(args.config,'r') as fl:
    config = yaml.load(fl,yaml.FullLoader)

job_info = config.get('job_info')
job_ids = job_info['job_ids']
job_types = job_info['job_types']
print(job_ids,flush=True)

MLP_MD_inds = [i for i, job_type in enumerate(job_types) if 'MLP_MD' in job_type]

final_MLP = f'logs/output.{job_ids[MLP_MD_inds[-1]]}'

checks = []
checks.append(read_file(final_MLP,'error value',config.get('UQ_min_uncertainty'), operator.lt))
checks.append(read_file(final_MLP,'error std',config.get('UQ_min_uncertainty')/2, operator.lt))
# checks.append(read_file(final_MLP,'max index',config.get('MLP_MD_steps')+1, operator.eq))

final_temperature = config['MLP_MD_temperature'] + config['n_temperature_sweep']*config['MLP_MD_dT']

termination_conditions = [
    config.get('i_temperature_sweep')>=config.get('max_temperature_sweep'),
    final_temperature>=config.get('max_MLP_temperature')
]

print('checks', checks)
if np.all(checks):
    config['MLP_MD_temperature']*=2
    config['MLP_MD_dT']*=2
    config['MLP_MD_steps']*=2
    config['n_temperature_sweep']+=1

print('termination conditions', termination_conditions)
if not np.any(termination_conditions):
    config['i_temperature_sweep']+=1
    with open(args.config,'w') as fl:
        yaml.dump(config, fl)
    
    python_file = 'start_active_learning.py'
    commands = ['python', python_file, '--config', args.config]
    process = subprocess.run(commands,capture_output=True)
    print('Job ids',process.stdout)

else:
    print(f'reached termination condition {termination_conditions}')


            