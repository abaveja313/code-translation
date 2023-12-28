from typing import List

import pandas as pd


def create_test_set(*, test_path: str, testcase_path: str) -> pd.DataFrame:
    test_df = pd.read_json(test_path, lines=True)
    # Use vectorized string concatenation
    test_df["sample_id"] = test_df["id"].astype(str) + "_" + test_df["sol"].astype(str)

    print(f"Test Dataset has {len(test_df)} entries")

    testcase_df = pd.read_json(testcase_path, lines=True).set_index("avatar_id")
    test_filtered = test_df[test_df.id.isin(testcase_df.index)]

    print(f"Testcase Dataset has {len(testcase_df)} entries")

    # Perform a left merge on the 'id' column of test_df and the index of testcase_df
    merged_df = pd.merge(
        test_filtered, testcase_df, left_on="id", right_index=True, how="left"
    )

    print(f"Merged dataset has {len(merged_df)} entries")
    print(merged_df.head(20))
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
