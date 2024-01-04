from dask_cloudprovider.aws import EC2Cluster

cluster: EC2Cluster = EC2Cluster(
    region='us-east-1',
    # worker_options={
    #     'worker_module': "dask_cuda.cli.dask_cuda_worker"
    # },
    worker_instance_type="g5.8xlarge",
    scheduler_instance_type="r6i.large",
    key_name="yogabbagabba",
    iam_instance_profile={
        "Arn": "arn:aws:iam::604465838084:instance-profile/cluster-instance-profile"
    },
    n_workers=1,
    docker_image="public.ecr.aws/x6j7a9j4/dask-gpu-base:latest",
    name="dask-cuda-cluster",
    ami="ami-02b9e30e585f05b11",
    docker_args="$(lspci | grep -i nvidia > /dev/null 2>&1 && echo \"--gpus all\")  --log-driver=json-file "
                "--log-opt max-size=10m --log-opt max-file=3",
    security=False,
    debug=True,
    filesystem_size=128,
    env_vars={
        "PARAMS": "13",
        "MODEL": "codellama-13b-instruct.Q4_K_M"
    }
)
