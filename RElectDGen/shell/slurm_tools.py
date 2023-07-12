'''
Tools for using slurm in python scripts
'''
import os
import glob

def check_if_job_running(id):
    command = f"squeue | grep '{id}.*R'"
    if os.system(command)==0:
        return True
    
    return False
def _gen_sbatch_config(config):
    '''
    Helper to write sbatch config
    '''

    
    config['--gres'] = config.get('--gres', False)
    config['x'] = config.get('x', False)
    config['w'] = config.get('w', False)
    config['e'] = config.get('e', False)
    config['--mail-type'] = config.get('--mail-type', False)
    config['--mail-user'] = config.get('--mail-user', 'eannevel@andrew.cmu.edu')

    txt = '#!/usr/bin/bash\n\n'
    for job_par in config:
        if config[job_par]:
            if job_par.startswith('--'):
                txt += f'#SBATCH {job_par}={config[job_par]}\n'
            else:
                txt += f'#SBATCH -{job_par} {config[job_par]}\n'
    return txt


def gen_job_script(commands, config={}, write_file=True, fname='job.sh'):
    '''
    Generates a slurm job script using given config or using default
    values otherwise

    Parameters
    ----------
        commands (list of str): List of commands to run
        config (dict): Slurm configuration with keys as options, values
            as dict values
        write_file (bool): Write job script or not
        fname (str): name of job script

    Returns
    -------
        str
            job script text
    '''

    txt = _gen_sbatch_config(config)

    txt += '\n\n'
    txt += 'echo "Job started on `hostname` at `date`"\n'
    for command in commands:
        txt += f'{command}\n'

    txt += 'echo " "\n'
    txt += 'echo "Job Ended at `date`"'

    if write_file:
        with open(fname, 'w') as f:
            f.write(txt)

    return txt


def gen_job_array(commands, flstring, config={},
                  write_file=True, fname='job_array.sh'):
    '''
    Generates a slurm job array script using given config or using default
    values otherwise

    Parameters
    ----------
        commands (list of str): List of commands to run
        fls (str): Matching string fo input files e.g. `input.*`
        config (dict): Slurm configuration with keys as options, values
            as dict values
        write_file (bool): Write job script or not
        fname (str): name of job script

    Returns
    -------
        str
            job script text
    '''

    txt = _gen_sbatch_config(config)

    txt += '\n\n'
    txt += 'if [[ ! -z ${SLURM_ARRAY_TASK_ID} ]]; then\n'
    txt += f'    fls=( {flstring} )\n'
    txt += '    F_NAME=${fls[${SLURM_ARRAY_TASK_ID}]}\n'
    txt += 'fi\n\n'

    txt += '# The action:\n'
    txt += 'echo "Job started on `hostname` at `date`"\n'
    for command in commands:
        txt += f'{command}\n'

    txt += 'echo " "\n'
    txt += 'echo "Job Ended at `date`"'

    if write_file:
        with open(fname, 'w') as f:
            f.write(txt)

    return txt


def submit_job(job_script_name):
    ''' Submits the job_script_name to slurm'''
    os.system(f'sbatch {job_script_name}')


def submit_jobs(job_names):
    ''' Submits jobs for each file matching job_names to slurm'''
    job_scripts = glob.glob('job_names')
    for job_script in job_scripts:
        os.system(f'sbatch {job_script}')


def submit_job_array(job_script_name, max_jobs=5, start=1, stop=5):
    ''' Submits the job_script_name to slurm'''
    os.system(f'sbatch --array={start}-{stop}%{max_jobs} {job_script_name}')