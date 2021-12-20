from slurm_tools import gen_job_array, gen_job_script
import os

def shell_from_config(config):

    location = config.get('dir_shell','submits')

    if not os.path.isdir('logs'):
        os.makedirs('logs')
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

    for file in filenames:
        fname = os.path.join(location,file)
        
        slurm_config = {
            'n': 18,
            't': '03-00:00',
            '--mem-per-cpu': 2000,
            'p': 'cpu',
            'A': 'venkvis',
            'J': file.split('.')[0],
            'o': 'logs/output.%j',
            'e': 'logs/error.%j'
        }

        if config.get('queue','cpu') == 'cpu':
            slurm_config['p'] = 'cpu'
            slurm_config['A'] = 'venkvis'
        elif config.get('queue') == 'gpu':
            slurm_config['p'] = 'gpu'
            slurm_config['A'] = 'venkvis_gpu'
        elif config.get('queue') == 'idle':
            slurm_config['p'] = 'idle'
            slurm_config['A'] = 'venkvis'
        elif config.get('queue') == 'highmem':
            slurm_config['p'] = 'highmem'
            slurm_config['A'] = 'venkvis'
            slurm_config['--mem-per-cpu'] = 13000

        

        if 'train' in file:
            commands = [
                "spack unload -a",
                "strings /usr/lib64/libstdc++.so.6 | grep CXXABI",
                "strings ~/.conda/pkgs/libstdcxx-ng-11.1.0-h56837e0_8/lib/libstdc++.so.6 | grep CXXABI",
                "source /home/spack/.spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/miniconda3-4.9.2-et7ujxrrzevxewx65fnmzqkftwwkrsyc/etc/profile.d/conda.sh",
                'conda activate nequip',
                'rm results/processed*/ -r',
                'REDGEN-combine-datasets $2 $3',
                # 'python ${1}scripts/'+f'{branch}/combine_datasets.py --config_file $2 --MLP_config_file $3',
                # 'nequip-train $3',
                'REDGEN-train-NN --config_file $2 --MLP_config_file $3',
                # 'python ${1}scripts/'+f'{branch}/train_NN.py --config_file $2 --MLP_config_file $3',
            ]
            slurm_config['n'] = python_cores
            slurm_config['N'] = python_nodes
            slurm_config['--ntasks'] = 1
            slurm_config['--cpus-per-task'] = python_cores
        elif 'restart' in file:
            commands = [
                "spack unload -a",
                "source /home/spack/.spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/miniconda3-4.9.2-et7ujxrrzevxewx65fnmzqkftwwkrsyc/etc/profile.d/conda.sh",
                'conda activate nequip',
                'REDGEN-restart --config_file $2 --MLP_config_file $3'
                # 'python ${1}scripts/'+f'{branch}/restart.py --config_file $2 --MLP_config_file $3',
            ]
            slurm_config['n'] = python_cores
            slurm_config['N'] = python_nodes
            slurm_config['--ntasks'] = 1
            slurm_config['--cpus-per-task'] = python_cores
        else:
            commands = ['spack load -r py-gpaw']

            if 'MLP' in file:
                commands += ['REDGEN-MLP-MD --config_file $2  --MLP_config_file $3 --loop_learning_count $4']
                # commands += ['python3 ${1}scripts/'+f'{branch}/slabmol_MLP_MD.py --config_file $2  --MLP_config_file $3 --loop_learning_count $4']
                slurm_config['n'] = python_cores
                slurm_config['N'] = python_nodes
                slurm_config['--ntasks'] = 1
                slurm_config['--cpus-per-task'] = python_cores
            elif 'MD' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_MD.py')
                commands += [f'srun -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3']
                # commands += [f'srun -n {gpaw_cores}' + ' gpaw python ${1}scripts/'+f'{branch}/slabmol_gpaw_MD.py --config_file $2 --MLP_config_file $3']
                slurm_config['n'] = gpaw_cores
                slurm_config['N'] = gpaw_nodes
            elif 'active' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_active.py')
                commands += [f'srun -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3 --loop_learning_count $4']
                # commands += [f'srun -n {gpaw_cores}' + ' gpaw python ${1}scripts/'+f'{branch}/slabmol_gpaw_active.py --config_file $2 --MLP_config_file $3 --loop_learning_count $4']
                slurm_config['n'] = gpaw_cores
                slurm_config['N'] = gpaw_nodes
            elif 'array' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_active_array.py')
                commands += [f'srun -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3 --loop_learning_count $4' + " --array_index ${SLURM_ARRAY_TASK_ID}"]
                # commands += [f'srun -n {gpaw_cores}' + ' gpaw python ${1}scripts/'+f'{branch}/gpaw_active_array.py --config_file $2 --MLP_config_file $3 --loop_learning_count $4' + " --array_index ${SLURM_ARRAY_TASK_ID}"]
                slurm_config['n'] = gpaw_cores
                slurm_config['N'] = gpaw_nodes
            elif 'summary' in file:
                file = os.path.join(config.get('scripts_path'),'gpaw_summary_array.py')
                commands += [f'srun -n {gpaw_cores}' + f' gpaw python {file} --config_file $2 --MLP_config_file $3 --loop_learning_count $4']
                # commands += [f'srun -n {gpaw_cores}' + ' gpaw python ${1}scripts/'+f'{branch}/gpaw_summary_array.py --config_file $2 --MLP_config_file $3 --loop_learning_count $4']
                slurm_config['n'] = gpaw_cores
                slurm_config['N'] = gpaw_nodes
            
        if 'array' in file:
            gen_job_array(commands,'',slurm_config,fname=fname)
        else:
            gen_job_script(commands,slurm_config,fname=fname)
            

