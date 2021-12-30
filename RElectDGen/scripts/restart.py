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

    
    checks = np.array(config.get('checks'))
    config['checks'] = []
    # except:
    #     checks = []
    #     checks.append(read_file(final_MLP,'error value',config.get('UQ_min_uncertainty'), operator.lt))
    #     checks.append(read_file(final_MLP,'error std',config.get('UQ_min_uncertainty')/2, operator.lt))
    # checks.append(read_file(final_MLP,'max index',config.get('MLP_MD_steps')+1, operator.eq))

    final_temperature = config['MLP_MD_temperature'] + config['n_temperature_sweep']*config['MLP_MD_dT']

    termination_conditions = [
        config.get('i_temperature_sweep')>=config.get('max_temperature_sweep'),
        final_temperature>=config.get('max_MLP_temperature'),
        config.get('UQ_min_uncertainty')<=config.get('UQ_terminal_accuracy',.01)
    ]

    print('checks', checks)
    if np.all(checks[:,:2]):
        config['MLP_MD_temperature']*=2
        config['MLP_MD_dT']*=2
    if np.all(checks[:,2]):
        config['MLP_MD_steps']*=2
    if np.all(checks[:,3]):
        config['UQ_min_uncertainty']/=2
        # config['n_temperature_sweep']+=1

    checks_history = config.get('checks_history',[])
    checks_history.append(np.all(checks,axis=0).tolist())
    config['checks_history'] = checks_history
    print('termination conditions', termination_conditions)
    if not np.any(termination_conditions):
        config['i_temperature_sweep']+=1
        with open(filename_config,'w') as fl:
            yaml.dump(config, fl)
        
        commands = ['REDGEN-start', filename_config]
        process = subprocess.run(commands,capture_output=True)
        print('Job ids',process.stdout)

    else:
        print(f'reached termination condition {termination_conditions}')

if __name__ == "__main__":
    main()
                