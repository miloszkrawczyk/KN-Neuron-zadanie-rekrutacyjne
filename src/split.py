"""Splitting subjects, never windows, into train and test.

Windows from one child are far more alike than windows from two different
children. Put the same child on both sides and the model can score well by
recognising the child rather than ADHD. So the split is made on the table of
subjects, and every window follows its subject.
"""

from sklearn.model_selection import train_test_split

from src.config import SEED, SUBJECT_COLUMN, TARGET_COLUMN, TEST_SIZE


def split_subjects(subjects, test_size=TEST_SIZE, seed=SEED):
    """Deal whole subjects into train and test, with the same ADHD share in both.

    Takes the table from subject_table() and returns two sorted lists of IDs.
    A window belongs to train when np.isin(window_subjects, train_ids).
    """
    # stratify keeps the ADHD share equal on both sides; without it, chance
    # alone could leave a 25-subject test set noticeably lopsided.
    train, test = train_test_split(
        subjects,
        test_size=test_size,
        stratify=subjects[TARGET_COLUMN],
        random_state=seed,
    )
    return sorted(train[SUBJECT_COLUMN]), sorted(test[SUBJECT_COLUMN])
