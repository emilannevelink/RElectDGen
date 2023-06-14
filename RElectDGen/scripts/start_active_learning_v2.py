import argparse
import subprocess
import os, yaml
from RElectDGen.scripts.gpaw_MD_db import check_oracle_steps

def parse_command_line(args):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file', dest='config', default='runs/test_python_run/active_learning.yaml',
    #                     help='active_learning configuration file', type=str)
    parser.add_argument('config', metavar='config_file', type=str,
                        help='active_learning configuration file')
    args = parser.parse_args()


    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    return config

def main(args=None):

    config= parse_command_line(args)
    
    config['scripts_path'] = os.path.dirname(os.path.abspath(__file__))

    data_root = os.environ.get('PROJECT',os.environ.get('HOME'))
    dir_root = os.environ.get('HOME')
    if data_root not in config.get('data_directory',''):
        config['data_directory'] = os.path.join(data_root,config.get('directory'))    
    if dir_root not in config.get('directory'):
        config['directory'] = os.path.join(dir_root,config.get('directory'))
    
    
    active_learning_index = config.get('active_learning_index')
    active_learning_config = os.path.join(
        config.get('directory'),
        config.get('run_dir'),
        eval(config.get('run_config_filename_func')) #this uses active_learning_index
    )

    with open(active_learning_config,'w+') as fl:
        yaml.dump(config, fl)
            
    MLP_config_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('MLP_config','MLP.yaml'))
    MLP_config_base = MLP_config_filename.split('.yaml')[0]
    MLP_config_current = MLP_config_base + f'_{active_learning_index}.yaml'
    if active_learning_index == 0:
        with open(MLP_config_filename,'r') as fl:
            MLP_config = yaml.load(fl,yaml.FullLoader)
        MLP_config['run_name'] = MLP_config['run_name']+f'_{active_learning_index}'
    else:
        with open(MLP_config_current,'r') as fl:
            MLP_config = yaml.load(fl,yaml.FullLoader)

    dataset_filename = os.path.join(config.get('data_directory'), config.get('combined_trajectory'))
    config['dataset_file_name'] = dataset_filename
    assert dataset_filename in MLP_config.get('dataset_file_name'), f'Dataset wrong in MLP.yaml, include {dataset_filename}'

    print(config.get('machine'),flush=True)

    generate_shell_command = ['REDGEN-generate-shell', '--config_file', active_learning_config]
    command_string = ' '.join(generate_shell_command)
    process = subprocess.run(command_string, check=True,capture_output=True, shell=True)

    filenames = config.get('shell_filenames')
    job_ids = []
    job_types = []

    ### Set global attributes
    config['max_electrons'] = int(
        config.get('electrons_per_core', 2.75)*
        config.get('gpaw_cores',config.get('cores',1)))
    # config.get('gpaw_nodes',config.get('nodes',1)))

    # for i in range(1,1+config.get('n_temperature_sweep')):
    i = 0
    location = config.get('dir_shell', 'submits')
    for shell_file in filenames:
        shell_file = os.path.join(location,shell_file)
        if 'gpaw_MD' in shell_file and check_oracle_steps(config)>0:
            commands = ['sbatch', shell_file, active_learning_config, MLP_config_current]
            command_string = ' '.join(commands)
            process = subprocess.run(command_string, capture_output=True, shell=True)
            job_ids.append(int(process.stdout.split(b' ')[-1]))
            job_types.append(shell_file)
        elif 'gpaw_array' in shell_file:
            narray = int(config.get('max_samples')-1)
            if len(job_ids)>0:
                commands = ['sbatch', f'--dependency=afterok:{job_ids[-1]}', f'--array=0-{narray}', shell_file, active_learning_config, MLP_config_current, str(i)]
            else:
                commands = ['sbatch', f'--array=0-{narray}', shell_file, active_learning_config, MLP_config_current, str(i)]
            
            command_string = ' '.join(commands)
            process = subprocess.run(command_string, capture_output=True, shell=True)
            job_ids.append(int(process.stdout.split(b' ')[-1]))
            job_types.append(shell_file)

            # shell_file = 'submits/gpaw_summary.sh'
            # commands = ['sbatch', f'--dependency=afterok:{job_ids[-1]}', shell_file, active_learning_config, MLP_config_current, str(i)]

            # command_string = ' '.join(commands)
            # process = subprocess.run(command_string, capture_output=True, shell=True)
            # job_ids.append(int(process.stdout.split(b' ')[-1]))
            # job_types.append(shell_file)

        elif 'train_prep' in shell_file:
            if len(job_ids)>0:
                commands = ['sbatch', f'--dependency=afterok:{job_ids[-1]}', shell_file, active_learning_config, MLP_config_current, str(i)]
            else:
                commands = ['sbatch', shell_file, active_learning_config, MLP_config_current, str(i)]

            command_string = ' '.join(commands)
            process = subprocess.run(command_string, capture_output=True, shell=True)
            job_ids.append(int(process.stdout.split(b' ')[-1]))
            job_types.append(shell_file)

        elif 'train_array' in shell_file:
            n_ensemble = config.get('n_uncertainty_ensembles',4)
            if len(job_ids)>0:
                commands = ['sbatch', f'--dependency=afterok:{job_ids[-1]}', f'--array=0-{n_ensemble-1}', shell_file, active_learning_config, MLP_config_current, str(i)]
            else:
                commands = ['sbatch', f'--array=0-{n_ensemble-1}', shell_file,active_learning_config, MLP_config_current, str(i)]
            
            command_string = ' '.join(commands)
            process = subprocess.run(command_string, capture_output=True, shell=True)
            job_ids.append(int(process.stdout.split(b' ')[-1]))
            job_types.append(shell_file)

        else:
            if len(job_ids)>0:
                commands = ['sbatch', f'--dependency=afterok:{job_ids[-1]}', shell_file, active_learning_config, MLP_config_current, str(i)]
            else:
                commands = ['sbatch', shell_file, active_learning_config, MLP_config_current, str(i)]

            command_string = ' '.join(commands)
            process = subprocess.run(command_string, capture_output=True, shell=True)
            job_ids.append(int(process.stdout.split(b' ')[-1]))
            job_types.append(shell_file)
        
        if 'restart' in shell_file:
            restart_ids = config.get('restart_ids',[])
            restart_ids.append(int(process.stdout.split(b' ')[-1]))
            config['restart_ids'] = restart_ids
            
        
    # if 'restart.sh' in filenames and len(job_ids)>0:

    #     shell_file = 'submits/restart.sh'
    #     commands = ['sbatch', f'--dependency=afterok:{job_ids[-1]}', shell_file, active_learning_config, MLP_config_current]
        
    #     command_string = ' '.join(commands)
    #     process = subprocess.run(command_string, capture_output=True, shell=True)
    #     job_ids.append(int(process.stdout.split(b' ')[-1]))
    #     job_types.append(shell_file)

        
    
    job_info = {
            'job_ids': job_ids,
            'job_types': job_types,
    }
    config['job_info'] = job_info
    
    with open(active_learning_config,'w') as fl:
        yaml.dump(config, fl)

    with open(MLP_config_current,'w') as fl:
        yaml.dump(MLP_config, fl)

    print(job_ids)

if __name__ == "__main__":
    main()