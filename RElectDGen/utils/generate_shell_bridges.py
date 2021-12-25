from slurm_tools import gen_job_array, gen_job_script
import os
import argparse
import yaml

def shell_from_config(config):

    location = config.get('dir_shell','submits')

    if not os.path.isdir('logs'):
        os.makedirs('logs')
    if not os.path.isdir(location):
        os.makedirs(location)

    filenames = config.get('shell_filenames')

    
    MLP_cores = config.get('MLP_cores',config.get('cores'))
    MLP_nodes = config.get('MLP_nodes',config.get('nodes',1))
    MLP_queue = config.get('MLP_queue',config.get('queue','RM-shared'))

    gpaw_cores = config.get('gpaw_cores',config.get('cores'))
    gpaw_nodes = config.get('gpaw_nodes',config.get('nodes',1))
    gpaw_queue = config.get('gpaw_queue',config.get('queue','RM-shared'))

    for file in filenames:
        fname = os.path.join(location,file)
        
        slurm_config = {
            'n': 18,
            't': '02-00:00',
            '--mem-per-cpu': 2000,
            'p': 'RM-shared',
            'A': False,
            'J': file.split('.')[0],
            'o': 'logs/output.%j',
            'e': 'logs/error.%j'
        }
        

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
            slurm_config['n'] = MLP_cores
            slurm_config['N'] = MLP_nodes
            slurm_config['p'] = MLP_queue
            slurm_config['--ntasks'] = 1
            slurm_config['--cpus-per-task'] = MLP_cores
        elif 'restart' in file:
            commands = [
                "module load anaconda3",
                'conda activate nequip',
                'export LD_LIBRARY_PATH=/opt/packages/anaconda3/lib:$LD_LIBRARY_PATH',
                # 'python ${1}scripts/'+f'{branch}/restart.py --config_file $2 --MLP_config_file $3',
                'REDGEN-restart --config_file $2 --MLP_config_file $3'
            ]
            slurm_config['n'] = MLP_cores
            slurm_config['N'] = MLP_nodes
            slurm_config['p'] = MLP_queue
            slurm_config['--ntasks'] = 1
            slurm_config['--cpus-per-task'] = MLP_cores
        else:
            commands = ["module load anaconda3", "conda activate nequip",'export LD_LIBRARY_PATH=/opt/packages/anaconda3/lib:$LD_LIBRARY_PATH']

            if 'MLP' in file:
                commands += ['REDGEN-MLP-MD --config_file $2  --MLP_config_file $3 --loop_learning_count $4']
                slurm_config['n'] = MLP_cores
                slurm_config['N'] = MLP_nodes
                slurm_config['p'] = MLP_queue
                slurm_config['--ntasks'] = 1
                slurm_config['--cpus-per-task'] = MLP_cores
            elif 'MD' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_MD.py')
                commands += [f'mpiexec -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3']
                slurm_config['n'] = gpaw_cores
                slurm_config['N'] = gpaw_nodes
                slurm_config['p'] = gpaw_queue
            elif 'active' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_active.py')
                commands += [f'mpiexec -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3 --loop_learning_count $4']
                slurm_config['n'] = gpaw_cores
                slurm_config['N'] = gpaw_nodes
                slurm_config['p'] = gpaw_queue
            elif 'array' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_active_array.py')
                commands += [f'mpiexec -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3 --loop_learning_count $4' + " --array_index ${SLURM_ARRAY_TASK_ID}"]
                slurm_config['n'] = gpaw_cores
                slurm_config['N'] = gpaw_nodes
                slurm_config['p'] = gpaw_queue
            elif 'summary' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_summary_array.py')
                # commands += [f'mpiexec -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3 --loop_learning_count $4']
                commands += ['REDGEN-gpaw-summary --config_file $2  --MLP_config_file $3 --loop_learning_count $4']
                slurm_config['n'] = MLP_cores
                slurm_config['N'] = MLP_nodes
                slurm_config['p'] = MLP_queue
                slurm_config['--ntasks'] = 1
                slurm_config['--cpus-per-task'] = MLP_cores
            
        if 'array' in file:
            gen_job_array(commands,'',slurm_config,fname=fname)
        else:
            gen_job_script(commands,slurm_config,fname=fname)
            

