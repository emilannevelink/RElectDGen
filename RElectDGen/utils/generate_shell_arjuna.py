from .slurm_tools import gen_job_array, gen_job_script
import os

def shell_from_config(config):

    location = os.path.join(config.get('directory'),config.get('run_dir'),config.get('dir_shell','submits'))
    log_path = os.path.join(config.get('directory'),config.get('run_dir'),'logs')

    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(location):
        os.makedirs(location)

    filenames = config.get('shell_filenames')

    if config.get('gpaw_cores') is not None:
        gpaw_cores = config.get('gpaw_cores')
        gpaw_nodes = config.get('gpaw_nodes',1)
    else:
        gpaw_cores = config.get('cores')
        gpaw_nodes = config.get('nodes',1)

    python_cores = config.get('cores')
    python_nodes = config.get('nodes',1)

    conda_environment = config.get('conda_env', 'nequip')
    spack_environment = config.get('spack_env', 'py-gpaw')

    for file in filenames:
        fname = os.path.join(location,file)
        
        slurm_config = slurm_config_from_config(config,file)

        if 'train' in file:
            commands = [
                "spack unload -a",
                "strings /usr/lib64/libstdc++.so.6 | grep CXXABI",
                "strings ~/.conda/pkgs/libstdcxx-ng-11.1.0-h56837e0_8/lib/libstdc++.so.6 | grep CXXABI",
                "source /home/spack/.spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/miniconda3-4.9.2-et7ujxrrzevxewx65fnmzqkftwwkrsyc/etc/profile.d/conda.sh",
                f'conda activate {conda_environment}',
                'rm results/processed*/ -r',
                'REDGEN-combine-datasets --config_file $2 --MLP_config_file $3',
                # 'python ${1}scripts/'+f'{branch}/combine_datasets.py --config_file $2 --MLP_config_file $3',
                # 'nequip-train $3',
                'REDGEN-train-NN --config_file $2 --MLP_config_file $3',
                # 'python ${1}scripts/'+f'{branch}/train_NN.py --config_file $2 --MLP_config_file $3',
            ]
            # slurm_config['n'] = python_cores
            # slurm_config['N'] = python_nodes
            # slurm_config['--ntasks'] = 1
            # slurm_config['--cpus-per-task'] = python_cores
        elif 'restart' in file:
            commands = [
                "spack unload -a",
                "source /home/spack/.spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/miniconda3-4.9.2-et7ujxrrzevxewx65fnmzqkftwwkrsyc/etc/profile.d/conda.sh",
                f'conda activate {conda_environment}',
                'REDGEN-restart --config_file $2 --MLP_config_file $3'
                # 'python ${1}scripts/'+f'{branch}/restart.py --config_file $2 --MLP_config_file $3',
            ]
            # slurm_config['n'] = python_cores
            # slurm_config['N'] = python_nodes
            # slurm_config['--ntasks'] = 1
            # slurm_config['--cpus-per-task'] = python_cores

        elif 'MLP' in file:
            commands = [
                "spack unload -a",
                "source /home/spack/.spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/miniconda3-4.9.2-et7ujxrrzevxewx65fnmzqkftwwkrsyc/etc/profile.d/conda.sh",
                f'conda activate {conda_environment}',
                'REDGEN-MLP-MD --config_file $2  --MLP_config_file $3 --loop_learning_count $4'
                # 'python ${1}scripts/'+f'{branch}/restart.py --config_file $2 --MLP_config_file $3',
            ]
            # slurm_config['n'] = python_cores
            # slurm_config['N'] = python_nodes
            # slurm_config['--ntasks'] = 1
            # slurm_config['--cpus-per-task'] = python_cores

        elif 'summary' in file:
                commands = [
                    "spack unload -a",
                    "source /home/spack/.spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/miniconda3-4.9.2-et7ujxrrzevxewx65fnmzqkftwwkrsyc/etc/profile.d/conda.sh",
                    f'conda activate {conda_environment}',
                    'REDGEN-gpaw-summary --config_file $2  --MLP_config_file $3 --loop_learning_count $4',
                    'REDGEN-log --config_file $2'
                ]

        else:
            commands = [f'spack load -r {spack_environment}']

            if 'MD' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_MD.py')
                commands += [f'srun -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3']
                # commands += [f'srun -n {gpaw_cores}' + ' gpaw python ${1}scripts/'+f'{branch}/slabmol_gpaw_MD.py --config_file $2 --MLP_config_file $3']
                # slurm_config['n'] = gpaw_cores
                # slurm_config['N'] = gpaw_nodes
            elif 'active' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_active.py')
                commands += [f'srun -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3 --loop_learning_count $4']
                # commands += [f'srun -n {gpaw_cores}' + ' gpaw python ${1}scripts/'+f'{branch}/slabmol_gpaw_active.py --config_file $2 --MLP_config_file $3 --loop_learning_count $4']
                # slurm_config['n'] = gpaw_cores
                # slurm_config['N'] = gpaw_nodes
            elif 'array' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_active_array.py')
                commands += [f'srun -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3 --loop_learning_count $4' + " --array_index ${SLURM_ARRAY_TASK_ID}"]
                # commands += [f'srun -n {gpaw_cores}' + ' gpaw python ${1}scripts/'+f'{branch}/gpaw_active_array.py --config_file $2 --MLP_config_file $3 --loop_learning_count $4' + " --array_index ${SLURM_ARRAY_TASK_ID}"]
                # slurm_config['n'] = gpaw_cores
                # slurm_config['N'] = gpaw_nodes
            
            
        if 'array' in fname:
            gen_job_array(commands,'',slurm_config,fname=fname)
        else:
            gen_job_script(commands,slurm_config,fname=fname)
            

def slurm_config_from_config(config, file):

    slurm_config = {
            'n': 18,
            't': '03-00:00',
            '--mem-per-cpu': 2000,
            'J': file.split('.')[0],
            'o': 'logs/output.%j',
            'e': 'logs/error.%j'
        }

    if ('summary' in file or
        'restart' in file):
        slurm_config['p'] = config.get('queue','cpu')
        slurm_config['t'] = '00-00:15'
        slurm_config['n'] = 1
        slurm_config['--cpus-per-task'] = 1
        slurm_config['N'] = 1
        slurm_config['--ntasks'] = 1
    elif ('train' in file or
        'MLP' in file):
        slurm_config['p'] = config.get('MLP_queue',config.get('queue','cpu'))
        cores = config.get('MLP_cores',config.get('cores'))
        slurm_config['n'] = cores
        slurm_config['--cpus-per-task'] = cores
        slurm_config['N'] = config.get('MLP_nodes',config.get('nodes',1))
        slurm_config['--ntasks'] = 1
    elif ('MD' in file or #note MLP has already taken out the MLP_MD
        'active' in file or 
        'array' in file):
        slurm_config['p'] = config.get('gpaw_queue',config.get('queue','cpu'))
        slurm_config['n'] = config.get('gpaw_cores',config.get('cores'))
        slurm_config['N'] = config.get('gpaw_nodes',config.get('nodes',1))

    if 'gpu' in slurm_config['p']:
        slurm_config['A'] = 'venkvis_gpu'
        n_gpu = min(1,max(4,int(slurm_config['n']/16.)))
        slurm_config[f'--gres'] = f'gpu:{n_gpu}'
    else:
        slurm_config['A'] = 'venkvis'
    
    if 'mem' in slurm_config['p']:
        slurm_config['--mem-per-cpu'] = 13000

    return slurm_config
