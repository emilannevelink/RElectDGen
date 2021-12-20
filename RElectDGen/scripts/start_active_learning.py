import argparse
import subprocess
import os, yaml

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
    print(os.path.dirname(os.path.abspath(__file__)), flush=True)

    config['data_directory'] = os.path.join(os.environ.get('PROJECT',os.environ.get('HOME')),config.get('directory'))
    config['directory'] = os.path.join(os.environ.get('HOME'),config.get('directory'))
    active_learning_config = os.path.join(config.get('directory'),config.get('run_dir'),config.get('run_config_file'))

    with open(active_learning_config,'w+') as fl:
        yaml.dump(config, fl)
            
    MLP_config_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('MLP_config','MLP.yaml'))

    with open(MLP_config_filename,'r') as fl:
        MLP_config = yaml.load(fl,yaml.FullLoader)

    dataset_filename = os.path.join(config.get('data_directory'), config.get('combined_trajectory'))
    assert dataset_filename in MLP_config.get('dataset_file_name'), f'Dataset wrong in MLP.yaml, include {dataset_filename}'

    print(config.get('machine'),flush=True)

    generate_shell_command = ['REDGEN-generate-shell', '--config_file', active_learning_config]
    process = subprocess.run(generate_shell_command, check=True,capture_output=True)

    filenames = config.get('shell_filenames')
    job_ids = []
    job_types = []

    shell_file = 'submits/gpaw_MD.sh'
    if shell_file.split('/')[-1] in filenames:
        commands = ['sbatch', shell_file, config.get("directory"), active_learning_config, MLP_config_filename]
        process = subprocess.run(commands, capture_output=True)
        job_ids.append(int(process.stdout.split(b' ')[-1]))
        job_types.append(shell_file)


    for i in range(1,1+config.get('n_temperature_sweep')):

        shell_file = 'submits/train_NN.sh'
        if shell_file.split('/')[-1] in filenames:
            if len(job_ids)>0:
                commands = ['sbatch', f'--dependency=afterok:{job_ids[-1]}', shell_file, config.get('directory'),active_learning_config, MLP_config_filename]
            else:
                commands = ['sbatch', shell_file, config.get('directory'),active_learning_config, MLP_config_filename]
            process = subprocess.run(commands,capture_output=True)
            job_ids.append(int(process.stdout.split(b' ')[-1]))
            job_types.append(shell_file)

        shell_file = 'submits/MLP_MD.sh'
        if shell_file.split('/')[-1] in filenames:
            if len(job_ids)>0:
                commands = ['sbatch', f'--dependency=afterok:{job_ids[-1]}', shell_file, config.get('directory'),active_learning_config, MLP_config_filename, str(i)]
            else:
                commands = ['sbatch', shell_file, config.get('directory'),active_learning_config, MLP_config_filename, str(i)]
            process = subprocess.run(commands,capture_output=True)
            job_ids.append(int(process.stdout.split(b' ')[-1]))
            job_types.append(shell_file)

        if 'gpaw_active.sh' in filenames:
            shell_file = 'submits/gpaw_active.sh'
            if len(job_ids)>0:
                commands = ['sbatch', f'--dependency=afterok:{job_ids[-1]}', shell_file, config.get('directory'),active_learning_config, MLP_config_filename, str(i)]
            else:
                commands = ['sbatch', shell_file, config.get('directory'),active_learning_config, MLP_config_filename, str(i)]
            process = subprocess.run(commands,capture_output=True)
            job_ids.append(int(process.stdout.split(b' ')[-1]))
            job_types.append(shell_file)
        elif 'gpaw_array.sh' in filenames:
            shell_file = 'submits/gpaw_array.sh'
            narray = int(config.get('max_samples')-1)
            if len(job_ids)>0:
                commands = ['sbatch', f'--dependency=afterok:{job_ids[-1]}', f'--array=0-{narray}', shell_file, config.get('directory'),active_learning_config, MLP_config_filename, str(i)]
            else:
                commands = ['sbatch', f'--array=0-{narray}', shell_file, config.get('directory'),active_learning_config, MLP_config_filename, str(i)]
            process = subprocess.run(commands,capture_output=True)
            job_ids.append(int(process.stdout.split(b' ')[-1]))
            job_types.append(shell_file)

            shell_file = 'submits/gpaw_summary.sh'
            commands = ['sbatch', f'--dependency=afterok:{job_ids[-1]}', shell_file, config.get('directory'),active_learning_config, MLP_config_filename, str(i)]
            process = subprocess.run(commands,capture_output=True)
            job_ids.append(int(process.stdout.split(b' ')[-1]))
            job_types.append(shell_file)
        
    if config.get('restart',False) and 'restart.sh' in filenames and len(job_ids)>0:

        job_info = {
            'job_ids': job_ids,
            'job_types': job_types,
        }
        config['job_info'] = job_info

        with open(active_learning_config,'w') as fl:
            yaml.dump(config, fl)
        
        shell_file = 'submits/restart.sh'
        commands = ['sbatch', f'--dependency=afterok:{job_ids[-1]}', shell_file, config.get('directory'), active_learning_config, MLP_config_filename]
        process = subprocess.run(commands,capture_output=True)
        job_ids.append(int(process.stdout.split(b' ')[-1]))

        restart_ids = config.get('restart_ids',[])
        restart_ids.append(int(process.stdout.split(b' ')[-1]))
        config['restart_ids'] = restart_ids
        with open(active_learning_config,'w') as fl:
            yaml.dump(config, fl)

    print(job_ids)

if __name__ == "__main__":
    main()