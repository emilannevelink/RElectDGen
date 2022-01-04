from yaml import nodes
from slurm_tools import gen_job_array, gen_job_script
import os
import argparse
import yaml

def shell_from_config(config):

    location = os.path.join(config.get('directory'),config.get('run_dir'),config.get('dir_shell','submits'))
    log_path = os.path.join(config.get('directory'),config.get('run_dir'),'logs')
    
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(location):
        os.makedirs(location)

    filenames = config.get('shell_filenames')

    gpaw_cores = config.get('gpaw_cores',config.get('cores'))

    for file in filenames:
        fname = os.path.join(location,file)
        
        slurm_config = slurm_config_from_config(config,file)
        

        if 'train' in file:
            commands = [
                "module load anaconda3",
                'conda activate nequip',
                'export LD_LIBRARY_PATH=/opt/packages/anaconda3/lib:$LD_LIBRARY_PATH',
                'rm results/processed*/ -r',
                # 'python ${1}scripts/'+f'{branch}/combine_datasets.py --config_file $2 --MLP_config_file $3',
                # 'nequip-train $3',
                # 'python ${1}scripts/'+f'{branch}/train_NN.py --config_file $2 --MLP_config_file $3',
                'REDGEN-combine-datasets --config_file $2 --MLP_config_file $3',
                'REDGEN-train-NN --config_file $2 --MLP_config_file $3',
            ]
            
        elif 'restart' in file:
            commands = [
                "module load anaconda3",
                'conda activate nequip',
                'export LD_LIBRARY_PATH=/opt/packages/anaconda3/lib:$LD_LIBRARY_PATH',
                # 'python ${1}scripts/'+f'{branch}/restart.py --config_file $2 --MLP_config_file $3',
                'REDGEN-restart --config_file $2 --MLP_config_file $3'
            ]
            
        else:
            commands = ["module load anaconda3", "conda activate nequip",'export LD_LIBRARY_PATH=/opt/packages/anaconda3/lib:$LD_LIBRARY_PATH']

            if 'MLP' in file:
                commands += ['REDGEN-MLP-MD --config_file $2  --MLP_config_file $3 --loop_learning_count $4']
                
            elif 'MD' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_MD.py')
                commands += [f'mpiexec -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3']
                
            elif 'active' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_active.py')
                commands += [f'mpiexec -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3 --loop_learning_count $4']
                
            elif 'array' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_active_array.py')
                commands += [f'mpiexec -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3 --loop_learning_count $4' + " --array_index ${SLURM_ARRAY_TASK_ID}"]
                
            elif 'summary' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_summary_array.py')
                # commands += [f'mpiexec -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3 --loop_learning_count $4']
                commands += ['REDGEN-gpaw-summary --config_file $2  --MLP_config_file $3 --loop_learning_count $4']
            
        if 'array' in file:
            gen_job_array(commands,'',slurm_config,fname=fname)
        else:
            gen_job_script(commands,slurm_config,fname=fname)
            
def slurm_config_from_config(config, file):

    slurm_config = {
            't': '02-00:00',
            'A': False,
            'J': file.split('.')[0],
            'o': 'logs/output.%j',
            'e': 'logs/error.%j'
        }

    if ('summary' in file or
        'restart' in file):
        slurm_config['p'] = 'RM-shared'
        cores = 1
        slurm_config['N'] = 1
        slurm_config['--ntasks'] = 1
    elif ('train' in file or
        'MLP' in file):
        slurm_config['p'] = config.get('MLP_queue',config.get('queue','RM-shared'))
        cores = config.get('MLP_cores',config.get('cores'))
        slurm_config['N'] = config.get('MLP_nodes',config.get('nodes',1))
        slurm_config['--ntasks'] = 1
    elif ('MD' in file or 
        'active' in file or 
        'array' in file):
        slurm_config['p'] = config.get('gpaw_queue',config.get('queue','RM-shared'))
        cores = config.get('gpaw_cores',config.get('cores'))
        slurm_config['N'] = config.get('gpaw_nodes',config.get('nodes',1))

    
    if 'RM' in slurm_config['p']:
        slurm_config['n'] = cores
        slurm_config['--mem-per-cpu'] = config.get('memory_per_core',2000)
        if '--ntasks' in slurm_config.keys():
            slurm_config['--cpus-per-task'] = cores
    elif 'GPU' in slurm_config['p']:
        slurm_config['--gpus'] = cores
        slurm_config['--mem-per-gpu'] = config.get('memory_per_core',2000)
        if '--ntasks' in slurm_config.keys():
            slurm_config['--gpus-per-task'] = cores

    return slurm_config

