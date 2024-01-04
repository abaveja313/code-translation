import logging

# Configure the root logger to DEBUG. This will apply to all loggers unless overridden.
logging.basicConfig(level=logging.DEBUG)

# Disable ptpython logger
logging.getLogger('ptpython').disabled = True

# Disable parso logger to avoid verbose debug logs
logging.getLogger('parso').setLevel(logging.WARNING)

from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(
    queue='gpu',  # replace with your SLURM queue/partition name
    cores=10,  # number of cores per job (vCPUs)
    memory='64GB',  # memory per job
    interface='em1',
    walltime='08:00:00',  # job time limit (3 hours)
    job_extra=[
        '-C GPU_BRD:GEFORCE',
        '--gpus=1',  # request 2 GPUs
        '--nodes=1',
        '--gres=gpu:1',
        '--gpu_cmode exclusive',
    ],  # specify GPU resources
    processes=1,
    job_script_prologue=[
        'module load gcc/12.1.0',
        'module load cuda/12.2.0',
        'source /home/users/abaveja/venv/bin/activate'
    ]
)
