import multiprocessing
from pathlib import Path
from typing import Annotated, Optional

import typer as typer
from prefect import flow
from prefect_dask import DaskTaskRunner
from prefect.filesystems import S3

from utils import create_test_set
from sampler import Sampler

app = typer.Typer()

N_THREADS: int = multiprocessing.cpu_count()


@flow(task_runner=DaskTaskRunner(), persist_result=True)
def generate_flow(
    *,
    experiment_id: str,
    model_path: str,
    test_path: str,
    testcase_path: str,
    num_gpu_layers: int,
    num_threads_batch: int,
    num_samples: int,
    top_p: float = 0.95,
    top_k: int = 15,
    temp: float = 0.5,
    batch_size: int = 10,
    prompt_context_size: int = 1024,
    seed: int = 42,
):
    dataset = create_test_set(test_path=test_path, testcase_path=testcase_path)
    sampler: Sampler = Sampler(gpu_layers=num_gpu_layers, model_path=model_path)
    samples = {}

    for index, row in dataset.iterrows():
        samples[index] = sampler.sample_async.submit(
            experiment_id=experiment_id,
            submission_id=index,
            python_code=row["python_code"],
            num_threads_batch=num_threads_batch,
            num_samples=num_samples,
            top_k=top_k,
            top_p=top_p,
            temp=temp,
            batch_size=batch_size,
            prompt_context_size=prompt_context_size,
            seed=seed,
        )

    return samples


@app.command()
def main(
    llama_model_path: Annotated[Optional[Path], typer.Option()],
    experiment_name: Annotated[str, typer.Option()],
    test_path: Annotated[Optional[Path], typer.Option()] = "test.java-python.id_code",
    task_retries: Annotated[
        Optional[int], typer.Option(help="How many retries before giving up on task")
    ] = 3,
    task_backoff_secs: Annotated[
        Optional[int], typer.Option(help="Sampling retry backoff")
    ] = 10,
    gpu_layers: Annotated[
        Optional[int], typer.Option(help="Number of NN layers to offload to GPU")
    ] = 20,
    testcase_path: Annotated[
        Optional[Path], typer.Option()
    ] = "combined_id2tests.jsonl",
    batch_size: Annotated[
        int, typer.Option(help="How many samples to run in parallel")
    ] = 20,
    threads_per_batch: Annotated[
        int, typer.Option(help="Threads per batch")
    ] = N_THREADS,
    n_samples: Annotated[
        int, typer.Option(help="Number of Samples to Generate for each Input")
    ] = 75,
    temp: Annotated[
        float, typer.Option(help="Stochasticity of Generated Outputs")
    ] = 0.4,
    top_k: Annotated[int, typer.Option(help="Top K Value")] = 10,
    top_p: Annotated[float, typer.Option(help="Top P Threshold")] = 0.9,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    prompt_context_size: Annotated[
        int, typer.Option(help="prompt_context_size")
    ] = 1024,
):
    experiment_flow = generate_flow.with_options(
        retries=task_retries,
        retry_delay_seconds=task_backoff_secs,
        result_storage=S3(bucket_path=f"results/samples/{experiment_name}"),
    )

    experiment_flow(
        experiment_id=experiment_name,
        model_path=str(llama_model_path.absolute()),
        test_path=str(test_path.absolute()),
        testcase_path=str(testcase_path.absolute()),
        num_gpu_layers=gpu_layers,
        num_threads_batch=threads_per_batch,
        num_samples=n_samples,
        top_p=top_p,
        top_k=top_k,
        temp=temp,
        batch_size=batch_size,
        seed=seed,
        prompt_context_size=prompt_context_size,
    )


if __name__ == "__main__":
    app()
