# Load EEG datasets
# This script loads and preprocesses EEG datasets from the specified directories.



# Global variables
PATH_DATASET = "./datasets"
PATH_SCRIPTS = "./scripts"
PATH_RESULTS = "./results"


#import libraries

import glob
import os
import zipfile
import numpy as np 
from pathlib import Path
from scipy.io import loadmat

# function to load datasets

def extract_bonn_data(dataset):
    """
    Extract signals of the Bonn dataset.

    Args:
      dataset: The dataset dictionary to populate.

    Returns:
      The updated dataset dictionary.
    """
    bonn_path = Path(PATH_DATASET) / "Bonn"
    labels = ["F", "N", "O", "S", "Z"]
    for lbl in labels:
        folder = bonn_path / lbl
        txt_files = sorted(
            list(folder.glob(f"{lbl}[0-9][0-9][0-9].txt")) +
            list(folder.glob(f"{lbl}[0-9][0-9][0-9].TXT"))
        )
        if lbl not in dataset:
            dataset[lbl] = []
        for f in txt_files:
            try:
                # Each .txt contains only numbers (a single column) ──> 1‑D np.ndarray
                data = np.loadtxt(f, dtype=float)  # shape: (n_samples,)
                dataset[lbl].append(np.asarray(data).squeeze())
            except Exception as e:
                print(f"Error reading {f.name}: {e}")
    

def extract_delhi_data(dataset):
    """
    Extract signals of the Delhi dataset.

    Args:
      dataset: The dataset dictionary to populate.

    Returns:
      The updated dataset dictionary.
    """

    delhi_path = Path(PATH_DATASET) / "Delhi"
    labels = ["interictal", "preictal", "ictal"]
    for lbl in labels:
        folder = delhi_path / lbl
        mat_files = sorted(folder.glob(f"{lbl}*.mat"))
        if lbl not in dataset:
            dataset[lbl] = []
        for f in mat_files:
            try:
                data = loadmat(f)
                dataset[lbl].append(np.asarray(data[lbl]).squeeze())
            except Exception as e:
                print(f"  Error reading {f.name}: {e}")

def extract_data(path):
    """
    Extract signals of the Bonn and Delhi datasets.

    Args:
      path: The path to the zip file.

    Returns:
      dataset: The dataset dictionary to populate.
    """
    with zipfile.ZipFile(path, 'r') as zip_ref:
      extract_path = os.path.dirname(path)
      zip_ref.extractall(extract_path)
    dataset={}
    extract_bonn_data(dataset)
    extract_delhi_data(dataset)

    return dataset

raw_path = PATH_DATASET +"/Raw_data.zip"

dataset= extract_data(raw_path)
print(dataset)