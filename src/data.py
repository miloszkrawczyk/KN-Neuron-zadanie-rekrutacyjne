"""Reading the raw CSV and checking it is what the rest of the code assumes.

Every later stage, from filtering to the train/test split, starts here. A
broken file caught at this point is an error message; caught later, it is a
model quietly trained on the wrong thing.
"""

import pandas as pd

from src.config import (
    CHANNELS,
    CLASS_LABELS,
    LABEL_COLUMN,
    RAW_CSV,
    SUBJECT_COLUMN,
    TARGET_COLUMN,
)


def load_eeg(path=RAW_CSV):
    """Read the CSV, check it matches config, and add the 0/1 label column."""
    # float32 halves the memory and loses nothing: the file holds whole
    # numbers up to about 14 000, and float32 stores every whole number up to
    # 16 million exactly. It is also what PyTorch computes in.
    df = pd.read_csv(path, dtype={channel: "float32" for channel in CHANNELS})

    expected = [*CHANNELS, LABEL_COLUMN, SUBJECT_COLUMN]
    if list(df.columns) != expected:
        missing = [column for column in expected if column not in df.columns]
        unexpected = [column for column in df.columns if column not in expected]
        raise ValueError(
            f"columns do not match config: missing {missing}, unexpected {unexpected}"
            " (both empty means the order differs)"
        )

    if df.isna().any().any():
        with_gaps = list(df.columns[df.isna().any()])
        raise ValueError(f"missing values in columns {with_gaps}")

    # map() turns a label it does not know into NaN instead of guessing, so
    # an unexpected value such as "adhd" cannot slip through as a 0.
    df[TARGET_COLUMN] = df[LABEL_COLUMN].map(CLASS_LABELS)
    if df[TARGET_COLUMN].isna().any():
        unknown = sorted(df.loc[df[TARGET_COLUMN].isna(), LABEL_COLUMN].unique())
        raise ValueError(f"unknown labels {unknown}, expected {list(CLASS_LABELS)}")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    labels_per_subject = df.groupby(SUBJECT_COLUMN)[TARGET_COLUMN].nunique()
    if (labels_per_subject > 1).any():
        mixed = list(labels_per_subject.index[labels_per_subject > 1])
        raise ValueError(f"subjects with more than one label: {mixed}")

    # The filter later runs over each subject's recording as one continuous
    # signal, so their rows have to sit together in the file. A new block
    # starts wherever the ID differs from the row above.
    blocks = (df[SUBJECT_COLUMN] != df[SUBJECT_COLUMN].shift()).sum()
    if blocks != df[SUBJECT_COLUMN].nunique():
        raise ValueError("a subject's rows are split across the file")

    return df


def subject_table(df):
    """One row per subject: its ID and its 0/1 label, in file order.

    The train/test split is made on this table, so that it deals out whole
    subjects and never single rows.
    """
    return df.groupby(SUBJECT_COLUMN, sort=False)[TARGET_COLUMN].first().reset_index()
