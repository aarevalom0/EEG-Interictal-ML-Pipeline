

# Global variables
PATH_DATASET = "../datasets"
PATH_SCRIPTS = "../scripts"
PATH_RESULTS = "../results"

#import libraries
import pandas as pd
import numpy as np
from scipy.stats import entropy as shannon_entropy
from scipy.stats import skew, kurtosis
import pywt
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import pycatch22

# function to processing of Bonn and Delhi datasets
def hjorth_parameters(signal: np.ndarray):
    """
    Computes Hjorth parameters for a 1-D signal:
      • Activity   = var(y)
      • Mobility   = √(var(y') / var(y))
      • Complexity = √(var(y'') / var(y')) / Mobility
    """
    activity = np.var(signal)
    dy = np.diff(signal)
    var_dy = np.var(dy)
    mobility = np.sqrt(var_dy / activity)
    ddy = np.diff(dy)
    var_ddy = np.var(ddy)
    mobility_dy = np.sqrt(var_ddy / var_dy)
    complexity = mobility_dy / mobility
    return activity, mobility, complexity
def window_energy(signal: np.ndarray) -> float:
    """Computes the energy of the signal: sum of squares."""
    return np.sum(signal**2)

def window_entropy(signal: np.ndarray, bins: int = 50) -> float:
    """Computes Shannon entropy from the signal histogram."""
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    return shannon_entropy(hist, base=2)
def dwt_features(signal, wavelet='db4', level=3):
    """
    Aplica DWT y extrae energía y estadísticas de los coeficientes
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = {}
    for i, coeff in enumerate(coeffs):
        prefix = f"DWT_L{i}" if i != 0 else "DWT_A"  # A: approximation, D: details
        features[f"{prefix}_Energy"] = np.sum(coeff ** 2)
        features[f"{prefix}_Mean"] = np.mean(coeff)
        features[f"{prefix}_Std"] = np.std(coeff)
        features[f"{prefix}_Kurtosis"] = kurtosis(coeff)
        features[f"{prefix}_Skewness"] = skew(coeff)
    return features

def extract_classic_features(dataset):
    true_events = ["F", "N", "interictal"]
    bonn_keys = ["F", "N", "O", "S", "Z"]

    all_features = []

    for key, value in dataset.items():
        for signal in value:
            activity, mobility, complexity = hjorth_parameters(signal)
            energy = window_energy(signal)
            entropy = window_entropy(signal)
            std = np.std(signal)
            kurt = kurtosis(signal)
            zero_crossings = ((signal[:-1] * signal[1:]) < 0).sum()
            skewness = skew(signal)
            label = 1 if key in true_events else 0
            origin = "Bonn" if key in bonn_keys else "Delhi"
            dwt_feats = dwt_features(signal, wavelet='db4', level=3)

            features = {
                "Origin": origin,
                "Activity": activity,
                "Mobility": mobility,
                "Complexity": complexity,
                "Energy": energy,
                "Entropy": entropy,
                "Std": std,
                "Kurtosis": kurt,
                "ZeroCrossings": zero_crossings,
                "Skewness": skewness,
                "Label": label

            }
            features.update(dwt_feats)
            all_features.append(features)

            features_df = pd.DataFrame(all_features)
    return features_df

def extract_tsfresh_features(dataset):
    """
    Extracts features using tsfresh from the dataset.
    """
    all_samples = []
    all_labels = []
    true_events = ["F", "N", "interictal"]
    sample_id = 0

    for key, signals in dataset.items():
        for signal in signals:
            label = 1 if key in true_events else 0
            df_signal = pd.DataFrame({
                "id": sample_id,
                "time": np.arange(len(signal)),
                "value": signal
            })

            all_samples.append(df_signal)

            all_labels.append((sample_id, label))
            sample_id += 1

    tsfresh_input = pd.concat(all_samples, ignore_index=True)

    label_df = pd.DataFrame(all_labels, columns=["id", "Label"])
    features = extract_features(
                                    tsfresh_input,
                                    column_id="id",
                                    column_sort="time",
                                    disable_progressbar=False
                                )
    features = impute(features)
    features = select_features(features, label_df["Label"], ml_task="classification")
    
    threshold = 0.25
    label_corr = selected_features.corr()['level_0']
    selected_features = label_corr[abs(label_corr) >= threshold].index.tolist()
    filtered_df = features[selected_features]
    cols = [col for col in filtered_df.columns if col != 'level_0']
    
    filtered_df = filtered_df[cols]

    return filtered_df

def extract_pycatch22_features(dataset):
    """
    Extracts features using pycatch22 from the dataset.
    """
    all_features = []
    true_events = ["F", "N", "interictal"]

    for key, signals in dataset.items():
        for signal in signals:
            results = pycatch22.catch22_all(signal)
            feature_row = dict(zip(results['names'], results['values']))


            all_features.append(feature_row)


    features_df = pd.DataFrame(all_features)
    label_corr = features_df.corr()['Label']
    threshold = 0.2
    selected_features = label_corr[abs(label_corr) >= threshold].index.tolist()
    filtered_df = features_df[selected_features]
    cols = [col for col in filtered_df.columns if col != 'Label'] + ['Label']
    filtered_df = filtered_df[cols]
    
    return filtered_df

def extract_features_from_datasets(dataset):
    """
    Extracts features from the Bonn and Delhi datasets.
    
    Args:
      dataset: The dataset dictionary to populate.
    
    Returns:
      A DataFrame containing the extracted features.
    """
    classic_features_df = extract_classic_features(dataset)
    tsfresh_features_df = extract_tsfresh_features(dataset)
    pycatch22_features_df = extract_pycatch22_features(dataset)
    combined_df = pd.concat([classic_features_df, tsfresh_features_df, pycatch22_features_df], axis=1)
    
    return combined_df