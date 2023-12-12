import json
import random
from functools import partial
from pathlib import Path
from typing import Annotated, Optional, Generator, Dict
from datasets import Dataset, DatasetDict
import pandas as pd
import typer
import re

app = typer.Typer()


def create_train_set(*, train_path: str, testcase_path: str) -> pd.DataFrame():
    train_df = pd.read_json(train_path, lines=True).set_index("id")
    print(f"Train Dataset has {len(train_df)} entries")
    testcase_df = pd.read_json(testcase_path, lines=True).set_index("avatar_id")
    print(f"Testcase Dataset has {len(testcase_df)} entries")
    train_filtered = train_df[train_df.index.isin(testcase_df.index)]
    print(f"Filtered dataset has {len(train_filtered)} entries")
    return train_filtered.join(other=testcase_df)


def preprocess_sample(sample: str) -> Optional[str]:
    if match := re.search("\[JAVA](.*?)\[/JAVA]", sample, re.DOTALL):
        return match.group(1)


def parse_samples(samples_file: str, delim: str):
    with open(samples_file, "r") as samples:
        data = samples.read()
        partitioned = data.split(delim)
        filtered = [s for s in partitioned if "[JAVA]" in s and "[/JAVA]" in s]
        return filtered


def build_dataset_gen(
    train_file: Path,
    testcases_file: Path,
    samples_dir: Path,
    eval_debug_file: Path,
    unique_pairs: int
) -> Generator[Dict[str, str]]:
    dataset = create_train_set(
        train_path=str(train_file.absolute()),
        testcase_path=str(testcases_file.absolute())
    )

    eval_results = {}
    with open(eval_debug_file, 'r') as edf:
        lines = [json.loads(line) for line in edf.read().split('\n\n')[:-1]]
        for parsed_line in lines:
            eval_results[parsed_line['id']] = parsed_line

    for problem_id in eval_results:
        samples_file = samples_dir.joinpath(problem_id + '.txt')
        parsed = parse_samples(str(samples_file.absolute()), '======')
        samples = {f'sample_{i}': preprocess_sample(s) for i, s in enumerate(parsed)}
        sample_evals = eval_results['problem_id']

        candidates = set()
        while len(candidates) < unique_pairs:
            i = random.randint(0, len(parsed) - 1)
            j = random.randint(0, len(parsed) - 1)
            missed_i = sample_evals[f'sample_{i}']['missed']
            missed_j = sample_evals[f'sample_{j}']['missed']

            if missed_i < missed_j:
                candidates.add((f'sample_{i}', f'sample_{j}'))

        for better_id, worse_id in candidates:
            record = {'python': dataset.loc[problem_id]['python_code'],
                      'python_tokenized': dataset.loc[problem_id]['python_code_tokenized'], 'sample_i': better_id,
                      'sample_j': worse_id, 'sample_i_code': samples[better_id], 'sample_j_code': samples[worse_id]}
            yield record


@app.command()
def build_dataset(
    train_file: Annotated[Optional[Path], typer.Option()],
    testcases_file: Annotated[Optional[Path], typer.Option()],
    samples_dir: Annotated[Optional[Path], typer.Option()],
    eval_debug_file: Annotated[Optional[Path], typer.Option()],
    unique_pairs: Annotated[int, typer.Option()] = 10,
    hf_dataset_name: str = 'abaveja313/trl_java_pairs'
):
    gen = partial(build_dataset_gen, train_file, testcases_file, samples_dir, eval_debug_file, unique_pairs)
    ds = Dataset.from_generator(gen)
    partitioned = ds.train_test_split(train_size=0.6, test_size=0.4)
    final = DatasetDict({
        'rl': partitioned['train'],
        'ft': partitioned['test']
    })
    final.push_to_hub(hf_dataset_name)


if __name__ == "__main__":
    app()
