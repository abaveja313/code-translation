from typing import List
import re

import pandas as pd
from loguru import logger


def compress_code(code):
    # Remove spaces around parentheses and operators
    compressed = re.sub(r'\s*([(),=+\-*/<>%&|^])\s*', r'\1', code)

    # Remove spaces after '[' and before ']'
    compressed = re.sub(r'\[\s+', '[', compressed)
    compressed = re.sub(r'\s+\]', ']', compressed)

    # Handle spaces around curly braces if needed
    # compressed = re.sub(r'\{\s+', '{', compressed)
    # compressed = re.sub(r'\s+\}', '}', compressed)

    # Remove unnecessary spaces at the beginning of lines
    compressed = re.sub(r'^\s+', '', compressed, flags=re.MULTILINE)

    return compressed


def create_test_set(*, test_path: str, testcase_path: str) -> pd.DataFrame:
    test_df = pd.read_json(test_path, lines=True)
    # Use vectorized string concatenation
    test_df["sample_id"] = test_df["id"].astype(str) + "_" + test_df["sol"].astype(str)

    logger.debug(f"Test Dataset has {len(test_df)} entries")

    testcase_df = pd.read_json(testcase_path, lines=True).set_index("avatar_id")
    test_filtered = test_df[test_df.id.isin(testcase_df.index)]

    logger.debug(f"Testcase Dataset has {len(testcase_df)} entries")

    # Perform a left merge on the 'id' column of test_df and the index of testcase_df
    merged_df = pd.merge(
        test_filtered, testcase_df, left_on="id", right_index=True, how="left"
    )

    logger.info(f"Merged dataset has {len(merged_df)} entries")
    logger.debug(merged_df.head(2))
    return merged_df.set_index("sample_id")


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
