import os
from .slurm_tools import gen_job_array, gen_job_script


def shell_from_config(config):

    location = os.path.join(config.get('directory'), config.get(
        'run_dir'), config.get('dir_shell', 'submits'))
    log_path = os.path.join(config.get('directory'),
                            config.get('run_dir'), 'logs')

    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(location):
        os.makedirs(location)

    filenames = config.get('shell_filenames')

    
    gpaw_cores = config.get('gpaw_cores',config.get('cores'))
    gpaw_nodes = config.get('gpaw_nodes',config.get('nodes',1))
    
    gpaw_cores *= gpaw_nodes

    python_cores = config.get('cores')
    python_nodes = config.get('nodes',1)

    conda_environment = config.get('conda_env', 'nequip2')

    for file in filenames:
        fname = os.path.join(location, file)

        slurm_config = slurm_config_from_config(config, file)

        libstdc_location = '/home/eannevel/.conda/pkgs/libstdcxx-ng-12.2.0-h46fd767_19/lib'
        commands = [
            "spack unload -a",
            "source /home/spack/.spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/miniconda3-4.9.2-et7ujxrrzevxewx65fnmzqkftwwkrsyc/etc/profile.d/conda.sh",
            f'conda activate {conda_environment}',
            f'export LD_LIBRARY_PATH={libstdc_location}:$LD_LIBRARY_PATH',
            'spack load /pteuooj #openmpi',
        ]

        if 'train_prep' in file:
            commands += [
                'rm results/processed*/ -r',
                'REDGEN-combine-datasets --config_file $1 --MLP_config_file $2',
                'REDGEN-train-prep --config_file $1 --MLP_config_file $2',
            ]
        elif 'train_array' in file:
            commands += [
                'REDGEN-train-array --config_file $1 --MLP_config_file $2' +
                " --array_index ${SLURM_ARRAY_TASK_ID}"
            ]
        elif 'calibrate' in file:
            commands += [
                'REDGEN-calibrate-UQ --config_file $1 --MLP_config_file $2'
            ]
        elif 'sample' in file:
            commands += [
                'REDGEN-sample --config_file $1 --MLP_config_file $2'
            ]
        elif 'gpaw_array' in file.lower():
            file = os.path.join(config.get('scripts_path'),'gpaw_active_array_db.py')
            commands += [
                f'srun  --mpi=pmix -n {gpaw_cores}' + f' gpaw python {file} $1 $2' + " ${SLURM_ARRAY_TASK_ID}"
            ]
        elif 'gpaw_md' in file.lower():
            file = os.path.join(config.get('scripts_path'),'gpaw_MD_db.py')
            commands += [
                f'srun  --mpi=pmix -n {gpaw_cores}' + f' gpaw python {file} $1 $2'
            ]
        elif 'restart' in file:
            commands += [
                'REDGEN-restart --config_file $1 --MLP_config_file $2'
            ]

        if 'array' in fname:
            gen_job_array(commands, '', slurm_config, fname=fname)
        else:
            gen_job_script(commands, slurm_config, fname=fname)


def slurm_config_from_config(config, file):

    slurm_config = {
        'n': 18,
        't': '03-00:00',
        '--mem-per-cpu': 2000,
        'J': file.split('.')[0],
        'o': 'logs/output.%j',
        'e': 'logs/error.%j'
    }

    if (
        'restart' in file or
        'train_prep' in file
    ):
        slurm_config['p'] = config.get('queue', 'cpu')
        slurm_config['t'] = '00-00:30'
        slurm_config['n'] = 1
        slurm_config['--cpus-per-task'] = 1
        slurm_config['N'] = 1
        slurm_config['--ntasks'] = 1
    elif (
        'train' in file or
        'UQ' in file or
        'sample' in file
    ):
        slurm_config['p'] = config.get('MLP_queue', config.get('queue', 'cpu'))
        cores = config.get('MLP_cores', config.get('cores'))
        slurm_config['n'] = cores
        slurm_config['--cpus-per-task'] = cores
        slurm_config['N'] = config.get('MLP_nodes', config.get('nodes', 1))
        slurm_config['--ntasks'] = 1

        train_time_limit = config.get('train_time_limit')
        if 'train' in file and train_time_limit is not None:
            slurm_config['t'] = train_time_limit
        sampling_time_limit = config.get('sampling_time_limit')
        if ('MLP' in file or 'adv' in file) and sampling_time_limit is not None:
            slurm_config['t'] = sampling_time_limit
    elif ('gpaw' in file):
        slurm_config['p'] = config.get('gpaw_queue',config.get('queue','cpu'))
        slurm_config['n'] = config.get('gpaw_cores',config.get('cores'))
        slurm_config['N'] = config.get('gpaw_nodes',config.get('nodes',1))

        initial_time_limit = config.get('initial_time_limit')
        if 'MD' in file and initial_time_limit is not None:
            slurm_config['t'] = initial_time_limit
        active_time_limit = config.get('active_time_limit')
        if ('active' in file or 
            'array' in file) and active_time_limit is not None:
            slurm_config['t'] = active_time_limit

    if 'gpu' in slurm_config['p']:
        slurm_config['A'] = 'venkvis_gpu'
        n_gpu = max(1, min(4, int(slurm_config['n']/16.)))
        n_gpu = int(config.get('gpu_cores', n_gpu))
        slurm_config['--gres'] = f'gpu:{n_gpu}'
    else:
        slurm_config['A'] = 'venkvis'

    if 'mem' in slurm_config['p']:
        slurm_config['--mem-per-cpu'] = 13000

    return slurm_config
