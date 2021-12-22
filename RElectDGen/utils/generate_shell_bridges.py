from slurm_tools import gen_job_array, gen_job_script
import os
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', dest='config',
                    help='active_learning configuration file', type=str)
args = parser.parse_args()

with open(args.config,'r') as fl:
    config = yaml.load(fl,yaml.FullLoader)

branch='test'

location = config.get('dir_shell','submits')

if not os.path.isdir('logs'):
    os.makedirs('logs')
if not os.path.isdir(location):
    os.makedirs(location)

# filenames = [
#     'gpaw_MD.sh',
#     'train_NN.sh',
#     'MLP_MD.sh',
#     'gpaw_active.sh',
#     'gpaw_array.sh',
#     'gpaw_summary.sh',
#     'restart.sh'
# ]

filenames = config.get('shell_filenames')

if config.get('gpaw_cores') is not None:
    gpaw_cores = config.get('gpaw_cores')
    gpaw_nodes = config.get('gpaw_nodes',1)
else:
    gpaw_cores = config.get('cores')
    gpaw_nodes = config.get('nodes',1)

python_cores = config.get('cores')
python_nodes = config.get('nodes',1)

for file in filenames:
    fname = os.path.join(location,file)
    
    slurm_config = {
        'n': 18,
        't': '01-00:00',
        '--mem-per-cpu': 2000,
        'p': 'RM-shared',
        'A': False,
        'J': file.split('.')[0],
        'o': 'logs/output.%j',
        'e': 'logs/error.%j'
    }

    if config.get('queue') == 'RM-shared':
        slurm_config['p'] = 'RM-shared'


    

    if 'train' in file:
        commands = [
            "module load anaconda3",
            'conda activate nequip',
            'rm results/processed*/ -r',
            # 'python ${1}scripts/'+f'{branch}/combine_datasets.py --config_file $2 --MLP_config_file $3',
            # 'nequip-train $3',
            # 'python ${1}scripts/'+f'{branch}/train_NN.py --config_file $2 --MLP_config_file $3',
            'REDGEN-combine-datasets --config_file $2 --MLP_config_file $3',
            'REDGEN-train-NN --config_file $2 --MLP_config_file $3',
        ]
        slurm_config['n'] = python_cores
        slurm_config['N'] = python_nodes
        slurm_config['--ntasks'] = 1
        slurm_config['--cpus-per-task'] = python_cores
    elif 'restart' in file:
        commands = [
            "module load anaconda3",
            'conda activate nequip',
            # 'python ${1}scripts/'+f'{branch}/restart.py --config_file $2 --MLP_config_file $3',
            'REDGEN-restart --config_file $2 --MLP_config_file $3'
        ]
        slurm_config['n'] = python_cores
        slurm_config['N'] = python_nodes
        slurm_config['--ntasks'] = 1
        slurm_config['--cpus-per-task'] = python_cores
    else:
        commands = ["module load anaconda3", "conda activate nequip"]

        if 'MLP' in file:
            commands += ['REDGEN-MLP-MD --config_file $2  --MLP_config_file $3 --loop_learning_count $4']
            slurm_config['n'] = python_cores
            slurm_config['N'] = python_nodes
            slurm_config['--ntasks'] = 1
            slurm_config['--cpus-per-task'] = python_cores
        elif 'MD' in file:
            file = os.path.join(config.get('scripts_path'),'gpaw_MD.py')
            commands += [f'mpiexec -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3']
            slurm_config['n'] = gpaw_cores
            slurm_config['N'] = gpaw_nodes
        elif 'active' in file:
            file = os.path.join(config.get('scripts_path'),'gpaw_active.py')
            commands += [f'mpiexec -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3 --loop_learning_count $4']
            slurm_config['n'] = gpaw_cores
            slurm_config['N'] = gpaw_nodes
        elif 'array' in file:
            file = os.path.join(config.get('scripts_path'),'gpaw_active_array.py')
            commands += [f'mpiexec -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3 --loop_learning_count $4' + " --array_index ${SLURM_ARRAY_TASK_ID}"]
            slurm_config['n'] = gpaw_cores
            slurm_config['N'] = gpaw_nodes
        elif 'summary' in file:
            file = os.path.join(config.get('scripts_path'),'gpaw_summary_array.py')
            commands += [f'mpiexec -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3 --loop_learning_count $4']
            slurm_config['n'] = gpaw_cores
            slurm_config['N'] = gpaw_nodes
        
    if 'array' in file:
        gen_job_array(commands,'',slurm_config,fname=fname)
    else:
        gen_job_script(commands,slurm_config,fname=fname)
        

