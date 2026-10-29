"""Downloads the ADHD EEG dataset from Kaggle into data/raw/.

kagglehub normally keeps downloads in its own cache under the home
directory. output_dir puts the file inside the project instead, so the rest
of the code only ever looks in one place. A second run finds the file
already there and skips the download.

Uses the Kaggle API token in ~/.kaggle/kaggle.json if there is one.
"""

import kagglehub

from src.config import KAGGLE_DATASET, RAW_CSV, RAW_DIR

print(f"fetching {KAGGLE_DATASET}: 47 MB zipped, 267 MB unpacked")
kagglehub.dataset_download(KAGGLE_DATASET, output_dir=str(RAW_DIR))

# The file name in config comes from Kaggle's file listing. If the dataset
# is ever re-uploaded under another name, better to stop here than to fail
# later with a confusing error in the loading code.
if not RAW_CSV.exists():
    found = sorted(path.name for path in RAW_DIR.iterdir())
    raise SystemExit(f"expected {RAW_CSV.name} in {RAW_DIR}, found: {found}")

# Decimal megabytes, the unit Kaggle uses on the dataset page.
size_mb = RAW_CSV.stat().st_size / 1e6
print(f"done: {RAW_CSV} ({size_mb:.0f} MB)")
