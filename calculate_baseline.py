import base64
import concurrent
import hashlib
import json
import os
import pprint
import re
import subprocess
import tempfile
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import tqdm as tqdm
import tqdm.contrib.concurrent
import typer
from typing_extensions import Annotated

app = typer.Typer()


class DataSource:
    ATCODER = "atcoder"
    CODEFORCES = "codeforces"


class TranslationStatus:
    COMPILE_ERROR = "compile_error"
    PASSED = "passed"
    TESTCASE_ERROR = "testcase_error"
    UNPARSEABLE = "unparseable"


class TestcaseStatus:
    PASSED = "passed"
    INCORRECT_OUTPUT = "incorrect_output"
    TIMEOUT = "timeout"
    RUNTIME_ERROR = "runtime_error"


def create_test_set(*, test_path: str, testcase_path: str) -> pd.DataFrame:
    test_df = pd.read_json(test_path, lines=True)
    # Use vectorized string concatenation
    test_df['sample_id'] = test_df['id'].astype(str) + '_' + test_df['sol'].astype(str)

    print(f"Test Dataset has {len(test_df)} entries")

    testcase_df = pd.read_json(testcase_path, lines=True).set_index('avatar_id')
    test_filtered = test_df[test_df.id.isin(testcase_df.index)]

    print(f"Testcase Dataset has {len(testcase_df)} entries")

    # Perform a left merge on the 'id' column of test_df and the index of testcase_df
    merged_df = pd.merge(test_filtered, testcase_df, left_on='id', right_index=True, how='left')

    print(f"Merged dataset has {len(merged_df)} entries")
    print(merged_df.head(20))
    return merged_df.set_index('sample_id')


def split_dataframe(df: pd.DataFrame, num_splits: int) -> List[pd.DataFrame]:
    split_size = len(df) // num_splits
    remainder = len(df) % num_splits
    splits = []

    for i in range(num_splits):
        start = i * split_size
        # For the last split, add the remainder
        end = start + split_size + (1 if i == num_splits - 1 and remainder != 0 else 0)
        splits.append(df.iloc[start:end])

    return splits

def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


class GenerateUtils:
    @staticmethod
    def generate(
        *,
        batch: pd.DataFrame,
        batch_id: int,
        output_path: str,
        binary: str,
        model: str,
        max_tokens: int,
        n_samples: int,
        temp: float,
        gpus: List[int],
        top_k: int,
        top_p: float,
        batch_size: int,
    ):
        batch_env = os.environ.copy()
        batch_env["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in gpus])
        print(
            f"Batch {batch_id}... visible devices:", batch_env["CUDA_VISIBLE_DEVICES"]
        )
        stats = defaultdict(lambda: 0)

        with open(f"log_batch_{batch_id}.log", "w") as log:
            for index, row in batch.iterrows():
                output_file = f"{output_path}/{index}.txt"
                command = list(
                    map(
                        str,
                        [
                            binary,
                            "-model",
                            model,
                            "-outputfile",
                            output_file,
                            # "-max-tokens",
                            # max_tokens,
                            "-batch",
                            batch_size,
                            "-samples",
                            n_samples,
                            "-temp",
                            temp,
                            "-top-k",
                            top_k,
                            "-top-p",
                            top_p,
                            "-b64",
                            row["b64"],
                            "-main-gpu",
                            gpus[0],
                        ],
                    )
                )
                print("Command:", command)

                with subprocess.Popen(
                    command, stdout=log, stderr=log, env=batch_env, text=True
                ) as process:
                    process.wait()
                    stats[process.returncode] += 1

    @staticmethod
    def generate_wrapper(split, args):
        return GenerateUtils.generate(batch=split, **args)

    @staticmethod
    def generate_from_dataframe(
        *,
        model_path: Path,
        output_path: Path,
        codellama_executable: Path,
        dataset: pd.DataFrame,
        batch_size: int,
        num_instances: int,
        max_tokens: int,
        n_samples: int,
        temp: float,
        top_k: int,
        top_p: float,
    ):
        splits: List[pd.DataFrame] = split_dataframe(dataset, num_instances)
        cuda_assignments: Dict[int, List[int]] = distribute_gpus_evenly(num_instances)
        print(f"CUDA Assigments: {cuda_assignments}")

        status = []
        with ProcessPoolExecutor() as executor:
            args = [
                dict(
                    batch_id=i,
                    output_path=str(output_path.absolute()),
                    binary=str(codellama_executable.absolute()),
                    model=str(model_path.absolute()),
                    max_tokens=max_tokens,
                    n_samples=n_samples,
                    temp=temp,
                    batch_size=batch_size,
                    gpus=cuda_assignments[i],
                    top_k=top_k,
                    top_p=top_p,
                )
                for i, split in enumerate(splits)
            ]

            futures = [
                executor.submit(GenerateUtils.generate_wrapper, splits[i], args[i])
                for i in range(len(splits))
            ]

            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                status.append(future.result())

        print("Final Stats:", status)


class SampleScoringUtils:
    @staticmethod
    def score_samples_wrapper(samples, args):
        return SampleScoringUtils.score_samples(samples, *args)

    @staticmethod
    def preprocess_sample(sample: str) -> Optional[str]:
        if match := re.search("\[JAVA](.*?)\[/JAVA]", sample, re.DOTALL):
            return match.group(1)

    @staticmethod
    def parse_samples(samples_file: str, delim: str):
        with open(samples_file, "r") as samples:
            data = samples.read()
            partitioned = data.split(delim)
            filtered = [s for s in partitioned if "[JAVA]" in s and "[/JAVA]" in s]
            return filtered

    @staticmethod
    def get_testcase_path(
        ds: str,
        tc_path: str,
        codeforces_testcase_dir: Path = Path("/data/codeforces_test_cases"),
        atcoder_testcase_dir: Path = Path("/data/atcoder_test_cases"),
    ) -> str:
        if ds == DataSource.CODEFORCES:
            tc_dir = codeforces_testcase_dir
        elif ds == DataSource.ATCODER:
            tc_dir = atcoder_testcase_dir
        else:
            raise Exception(f"Unable to find test case directory for {tc_path}")

        return str(tc_dir.joinpath(tc_path).absolute())

    @staticmethod
    def run_single_testcase(
        java_path: str, tc_input: str, expected_out: str
    ):
        result = {}

        with open(tc_input, "r") as input_file:
            try:
                run_process = subprocess.run(
                    ["java", java_path],
                    stdin=input_file,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if run_process.returncode != 0:
                    result["status"] = TestcaseStatus.RUNTIME_ERROR
                    return tc_input, result

                with open(expected_out, "r") as output_file:
                    tc_stdout = output_file.read().strip().rstrip('\n')
                    eval_stdout = run_process.stdout.strip().rstrip('\n')
                    if tc_stdout == eval_stdout:
                        result["status"] = TestcaseStatus.PASSED
                    else:
                        result["status"] = TestcaseStatus.INCORRECT_OUTPUT
                        result["info"] = str([tc_stdout, eval_stdout])

            except subprocess.TimeoutExpired:
                traceback.print_exc()
                result["status"] = TestcaseStatus.TIMEOUT
                print("Code: ", open(java_path, "r").read())

            return tc_input, result

    @staticmethod
    def run_testcases(
        java_path: str, testcases: List[Tuple[str, str]]
    ):
        status_to_testcases: Dict[str, List[str]] = defaultdict(list)
        missed = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    SampleScoringUtils.run_single_testcase,
                    java_path,
                    tc_input,
                    expected_out
                )
                for tc_input, expected_out in testcases
            ]
            for future in concurrent.futures.as_completed(futures):
                tc_input, result = future.result()
                status = result.get("status")
                if not status == TestcaseStatus.PASSED:
                    status_to_testcases[status].append(tc_input)

                    if not status == TestcaseStatus.TIMEOUT:
                        missed += 1

        return dict(missed=missed, tc=status_to_testcases, passed=missed == 0)

    @staticmethod
    def process_sample(sample_id, sample, testcases):
        sample_status: Dict[str, Any] = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            java_file_path = os.path.join(temp_dir, "Solution.java")
            with open(java_file_path, "w") as java_file:
                if res := SampleScoringUtils.preprocess_sample(sample):
                    java_file.write(res)
                else:
                    sample_status["passed"] = False
                    sample_status["status"] = TranslationStatus.UNPARSEABLE
                    return sample_id, sample_status

            compile_process = subprocess.run(
                ["javac", java_file_path], capture_output=True, text=True
            )
            if compile_process.returncode != 0:
                sample_status["passed"] = False
                sample_status["status"] = TranslationStatus.COMPILE_ERROR
                return sample_id, sample_status

            results = SampleScoringUtils.run_testcases(
                java_file_path, testcases
            )

            return sample_id, results

    @staticmethod
    def score_samples(
        samples: List[str], testcases: List[Tuple[str, str]]
    ):
        samples_status = {}
        passed_tcs = []
        c = 0

        # Figure out which samples are identical so we only need to evaluate them once
        hashes = []
        duplicates: Dict[str, List[str]] = defaultdict(list)

        for idx, sample in enumerate(samples):
            md5_hash = hashlib.md5(sample.encode()).hexdigest()
            if md5_hash not in hashes:
                hashes.append(md5_hash)
                duplicates[f"sample_{idx}"].append(f"sample_{idx}")
            else:
                hashes.append(None)
                source = hashes.index(md5_hash)
                duplicates[f"sample_{source}"].append(f"sample_{idx}")

        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    SampleScoringUtils.process_sample,
                    sid,
                    samples[int(sid.split('_')[1].split('_')[0])],
                    testcases
                )
                for sid in duplicates.keys()
            ]
            with tqdm.tqdm(total=len(duplicates)) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    sample_id, sample_status = future.result()

                    for ident_sample_id in duplicates[sample_id]:
                        samples_status.update({ident_sample_id: sample_status})
                        if sample_status['passed']:
                            passed_tcs.append(ident_sample_id)
                            c += 1

        return dict(
            samples=samples_status,
            passed=list(sorted(passed_tcs)),
            metric_pass_at_1=pass_at_k(len(samples), c, 1),
            metric_pass_at_5=pass_at_k(len(samples), c, 5),
            metric_pass_at_10=pass_at_k(len(samples), c, 10),
        )


@app.command()
def generate_samples(
    model_path: Annotated[Optional[Path], typer.Option()],
    output_path: Annotated[Optional[Path], typer.Option()],
    codellama_executable: Annotated[Optional[Path], typer.Option()],
    test_path: Annotated[Optional[Path], typer.Option()] = "test.java-python.id_code",
    testcase_path: Annotated[
        Optional[Path], typer.Option()
    ] = "combined_id2tests.jsonl",
    batch_size: Annotated[
        int, typer.Option(help="How many samples to run in parallel")
    ] = 20,
    num_instances: Annotated[
        int, typer.Option(help="Number of Parallel Instances to Run")
    ] = 4,
    max_tokens: Annotated[int, typer.Option(help="Max Tokens to Generate")] = 1256,
    n_samples: Annotated[
        int, typer.Option(help="Number of Samples to Generate for each Input")
    ] = 75,
    temp: Annotated[
        float, typer.Option(help="Stochasticity of Generated Outputs")
    ] = 0.4,
    top_k: Annotated[int, typer.Option(help="Top K Value")] = 10,
    top_p: Annotated[float, typer.Option(help="Top P Threshold")] = 0.9,
):
    dataset = create_test_set(
        test_path=str(test_path.absolute()), testcase_path=str(testcase_path.absolute())
    )

    samples = 




@app.command()
def eval_samples(
    data_path: Annotated[Optional[Path], typer.Option()] = "test.java-python.id_code",
    testcase_path: Annotated[
        Optional[Path], typer.Option()
    ] = "combined_id2tests.jsonl",
    sample_dir: Annotated[Optional[Path], typer.Option()] = "/data/output",
    processed_log_path: Annotated[Optional[Path], typer.Option()] = "/data/processed_files.log",
    debug_path: Annotated[Optional[Path], typer.Option()] = "/data/results/debug.jsonl",
    results_path: Annotated[
        Optional[Path], typer.Option()
    ] = "/data/results/results.json",
    interval: Annotated[int, typer.Option(help="Interval in seconds to check for new files")] = 60,
):
    dataset = create_test_set(
        test_path=str(data_path.absolute()), testcase_path=str(testcase_path.absolute())
    )

    results: Dict[str, Any] = defaultdict(list)
    if processed_log_path.exists():
        with open(processed_log_path, "r") as file:
            processed_files = set(file.read().splitlines())
    else:
        processed_files = set()

    while True:
        files = set(sample_dir.iterdir()) - processed_files
        if not files:
            print(f"No new files found. Checking again in {interval} seconds.")
            time.sleep(interval)
            continue

        for sample_file in tqdm.tqdm(files, total=len(files)):
            submission_id = Path(sample_file).stem.split("_batch")[0]
            samples = SampleScoringUtils.parse_samples(
                samples_file=str(sample_file.absolute()), delim="======"
            )

            record = dataset.loc[submission_id]  # get first match
            test_cases = list(zip(record["inputs"], record["outputs"]))
            valid_testcases: List[Tuple[str, str]] = []
            for _in, _out in test_cases:
                out_path = SampleScoringUtils.get_testcase_path(ds=submission_id.split("_")[0], tc_path=_out)
                in_path = SampleScoringUtils.get_testcase_path(ds=submission_id.split("_")[0], tc_path=_in)
                valid_testcases.append((in_path, out_path))

            result = SampleScoringUtils.score_samples(
                samples, valid_testcases
            )
            result['id'] = submission_id
            with open(str(debug_path.absolute()), "a") as debug_file:
                debug_file.write(json.dumps(result) + "\n\n")

            for key in result:
                if key.startswith("metric_"):
                    results[key].append(result[key])

            processed_files.add(sample_file)
            with open(str(processed_log_path.absolute()), "a") as log_file:
                log_file.write(submission_id + "\n")

            agg_metrics = {}
            for metric in results:
                agg_metrics["expected_" + metric] = np.mean(results[metric])
                agg_metrics["stddev_" + metric] = np.std(results[metric])

            print("Results so far: ", pprint.pprint(results | agg_metrics))
            with open(results_path, "w") as metric_path:
                metric_path.write(json.dumps(results | agg_metrics, indent=2))

    print(f"Waiting for {interval} seconds before next check.")
    time.sleep(interval)


@app.command()
def generate_expanded_context(
    model_path: Annotated[Optional[Path], typer.Option()],
    output_path: Annotated[Optional[Path], typer.Option()],
    codellama_executable: Annotated[Optional[Path], typer.Option()],
    test_path: Annotated[Optional[Path], typer.Option()] = "test.java-python.id_code",
    testcase_path: Annotated[
        Optional[Path], typer.Option()
    ] = "combined_id2tests.jsonl",
    batch_size: Annotated[
        int, typer.Option(help="How many samples to run in parallel")
    ] = 20,
    num_instances: Annotated[
        int, typer.Option(help="Number of Parallel Instances to Run")
    ] = 4,
    expanded_max_tokens: Annotated[
        int, typer.Option(help="Max Tokens to Generate")
    ] = 1800,
    n_samples: Annotated[
        int, typer.Option(help="Number of Samples to Generate for each Input")
    ] = 20,
    temp: Annotated[
        float, typer.Option(help="Stochasticity of Generated Outputs")
    ] = 0.4,
    top_k: Annotated[int, typer.Option(help="Top K Value")] = 10,
    top_p: Annotated[float, typer.Option(help="Top P Threshold")] = 0.9,
):
    dataset = create_test_set(
        test_path=str(test_path.absolute()), testcase_path=str(testcase_path.absolute())
    )

    targets: List[str] = []
    for sample_file in output_path.iterdir():
        sample_id: str = sample_file.stem.split("_batch")[0]

        samples = SampleScoringUtils.parse_samples(
            samples_file=str(sample_file.absolute()), delim="======"
        )
        if len(samples) < n_samples:
            print(f"{sample_file} sample Count: {len(samples)} < {n_samples}")
            targets.append(sample_id)

    print(f"Identified {len(targets)} targets for reevaluation.")
    reeval_dataset = dataset.loc[targets]

    GenerateUtils.generate_from_dataframe(
        dataset=reeval_dataset,
        model_path=model_path,
        output_path=output_path,
        codellama_executable=codellama_executable,
        batch_size=batch_size,
        num_instances=num_instances,
        max_tokens=expanded_max_tokens,
        temp=temp,
        top_k=top_k,
        top_p=top_p,
        n_samples=n_samples,
    )


if __name__ == "__main__":
    app()
