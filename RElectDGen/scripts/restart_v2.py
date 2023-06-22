import os
import argparse
import subprocess
import operator
import numpy as np
import yaml

from ase.db import connect

def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    parser.add_argument('--MLP_config_file', dest='MLP_config',
                        help='Nequip configuration file', type=str)
    args = parser.parse_args(argsin)

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    with open(args.MLP_config,'r') as fl:
        MLP_config = yaml.load(fl,yaml.FullLoader)

    return config, MLP_config

def read_file(file, keyword, value, operation):
    with open(file,'r') as fl:
        lines = fl.readlines()

    for line in lines:
        if keyword in line:
            state = float(line.split(' ')[-1])

            return operation(state,value)

    return False

def main(args = None):

    config, MLP_config = parse_command_line(args)

    job_info = config.get('job_info')
    job_ids = job_info['job_ids']
    job_types = job_info['job_types']
    print(job_ids,flush=True)
    
    continue_al = False
    active_learning_index = config.get('active_learning_index')
    termination_conditions = config.get('termination_conditions',{})

    ## check db
    db_filename = os.path.join(
        config.get('data_directory'),
        config.get('ase_db_filename')
    )
    assert os.path.isfile(db_filename)
    db = connect(db_filename)
    n_recalculate = db.count(active_learning_index=active_learning_index)
    max_md_samples = config.get('max_md_samples',1)
    n_unstable = db.count(f'success=True,md_stable<{max_md_samples}')
    
    if n_unstable > termination_conditions.get('max_n_unstable',0):
        print('The number of unstable samples in the dataset is: ', n_unstable)
        continue_al = True

    max_samples = config.get('max_samples',10)
    if n_recalculate < max_samples:
        print('Updated n sample from ', config['md_sampling_initial_conditions'])
        config['md_sampling_initial_conditions'] += max_samples - n_recalculate
        print('to ', config['md_sampling_initial_conditions'])

    if config.get('restart',False) and continue_al:
        active_learning_index = config.get('active_learning_index') + 1 #Increment active learning index by one
        config['active_learning_index'] = active_learning_index
        
        MLP_config_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('MLP_config','MLP.yaml'))
        MLP_config_base = MLP_config_filename.split('.yaml')[0]
        MLP_config_filename_next = MLP_config_base + f'_{active_learning_index}.yaml'
        if int(MLP_config['run_name'].split('_')[-1]) == active_learning_index-1:
            MLP_config['run_name'] = '_'.join(MLP_config['run_name'].split('_')[:-1])+f'_{active_learning_index}'
        else:
            MLP_config['run_name'] = MLP_config['run_name']+f'_{active_learning_index}'
        active_learning_config = os.path.join(
            config.get('directory'),
            config.get('run_dir'),
            eval(config.get('run_config_filename_func')) #this uses active_learning_index
        )

        with open(active_learning_config,'w') as fl:
            yaml.dump(config,fl)
        with open(MLP_config_filename_next,'w') as fl:
            yaml.dump(MLP_config,fl)
        
        commands = ['REDGEN-start', active_learning_config]
        process = subprocess.run(commands,capture_output=True)
        print('Job ids',process.stdout)
        print('Error',process.stderr)

    else:
        print('restart is False')

if __name__ == "__main__":
    main()
                