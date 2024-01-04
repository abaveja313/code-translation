import multiprocessing
from pathlib import Path
from typing import Optional

import typer as typer
from prefect import flow
from prefect.filesystems import S3
from prefect_dask import DaskTaskRunner, get_dask_client

from sampler import sample, LlamaInitializePlugin
from utils import create_test_set, compress_code

app = typer.Typer()
N_THREADS: int = multiprocessing.cpu_count()


@flow(persist_result=True)
def generate_flow(
    *,
    experiment_id: str,
    llama_model_path: str,
    test_path: str,
    testcase_path: str,
    num_gpu_layers: int,
    num_threads_batch: int,
    upper_token_limit: int,
    num_samples: int,
    top_p: float = 0.95,
    top_k: int = 15,
    temp: float = 0.5,
    batch_size: int = 10,
    prompt_context_size: int = 1024,
    seed: int = 42,
):
    dataset = create_test_set(test_path=test_path, testcase_path=testcase_path)
    samples = {}
    for index, row in dataset.head(10).iterrows():
        samples[index] = sample(
            experiment_id=experiment_id,
            submission_id=str(index),
            model_path=llama_model_path,
            gpu_layers=num_gpu_layers,
            python_code=compress_code(row["python_code"]),
            num_threads_batch=num_threads_batch,
            num_samples=num_samples,
            top_k=top_k,
            top_p=top_p,
            temp=temp,
            batch_size=batch_size,
            prompt_context_size=prompt_context_size,
            seed=seed,
            upper_token_limit=upper_token_limit
        )

    return samples


@app.command()
def main(
    llama_model_path: Optional[Path],
    experiment_name: str,
    test_path: Optional[Path] = "test.java-python.id_code",
    dask_scheduler: str = "localhost:36515",
    task_retries: int = 3,
    task_backoff_secs: int = 10,
    gpu_layers: int = 20,
    testcase_path: Optional[Path] = "combined_id2tests.jsonl",
    batch_size: int = 20,
    threads_per_batch: int = N_THREADS,
    n_samples: int = 75,
    upper_token_limit: int = 950,
    temp: float = 0.4,
    top_k: int = 10,
    top_p: float = 0.9,
    seed: int = 42,
    prompt_context_size: int = 1024,
    s3_bucket_name: str = "llmsemantics",
):
    runner = DaskTaskRunner(address=dask_scheduler)
    experiment_flow = generate_flow.with_options(
        task_runner=runner,
        retries=task_retries,
        result_serializer="json",
        retry_delay_seconds=task_backoff_secs,
        result_storage=S3(
            bucket_path=f"{s3_bucket_name}/results/samples/{experiment_name}"
        ),
    )

    plugin = LlamaInitializePlugin(model_path=str(llama_model_path.absolute()), gpu_layers=gpu_layers)

    with get_dask_client(address=dask_scheduler) as client:
        client.register_worker_plugin(plugin, name='initialize')

    experiment_flow(
        experiment_id=experiment_name,
        llama_model_path=str(llama_model_path.absolute()),
        test_path=str(test_path.absolute()),
        testcase_path=str(testcase_path.absolute()),
        num_gpu_layers=gpu_layers,
        num_threads_batch=threads_per_batch,
        num_samples=n_samples,
        top_p=top_p,
        top_k=top_k,
        upper_token_limit=upper_token_limit,
        temp=temp,
        batch_size=batch_size,
        seed=seed,
        prompt_context_size=prompt_context_size,
    )


if __name__ == "__main__":
    app()
