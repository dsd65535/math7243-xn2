# pylint:disable=invalid-name,fixme
"""This module provides some common functions"""
import numpy as np
import pandas as pd


def get_data(y_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Get X and y data"""

    crispr_gene_effect = pd.read_csv("cache/22Q2/CRISPR_gene_effect.csv")
    sample_info = pd.read_csv("cache/22Q2/sample_info.csv")

    crispr_gene_effect_nona = crispr_gene_effect.dropna()
    crispr_gene_effect_meta = pd.merge(
        crispr_gene_effect_nona, sample_info, on="DepMap_ID", how="inner"
    )

    labels = []
    indices = []
    for _, row in crispr_gene_effect_meta.iterrows():
        # TODO: don't iterate through rows
        lineage = row[y_name]
        if lineage not in labels:
            labels.append(lineage)
        indices.append(labels.index(lineage))

    X_data = crispr_gene_effect_nona.select_dtypes(np.number).to_numpy()
    y_data = np.array(indices)

    return X_data, y_data, labels


def process_data(
    X_data: np.ndarray, y_data: np.ndarray, test_fraction: float = 1.0 / 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data up into training and testing"""

    # Do this first to capture all categories
    y_data_dummy = pd.get_dummies(y_data).to_numpy()

    test_count = round(len(y_data) * test_fraction)

    X_train = X_data[test_count:]
    y_train = y_data[test_count:]
    y_train_dummy = y_data_dummy[test_count:]

    X_test = X_data[:test_count]
    y_test = y_data[:test_count]
    y_test_dummy = y_data_dummy[:test_count]

    return X_train, y_train, y_train_dummy, X_test, y_test, y_test_dummy


def filter_singles(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove categories with a single member"""

    label_counts: dict[float, int] = {}
    for label in y_train:
        if label in label_counts:
            label_counts[float(label)] += 1
        else:
            label_counts[float(label)] = 1

    train_mult = pd.DataFrame(y_train).apply(
        lambda x: label_counts.get(float(x.to_numpy()), 0) > 1, axis=1
    )
    X_train_mult = X_train[train_mult]
    y_train_mult = y_train[train_mult]
    test_mult = pd.DataFrame(y_test).apply(
        lambda x: label_counts.get(float(x.to_numpy()), 0) > 1, axis=1
    )
    X_test_mult = X_test[test_mult]
    y_test_mult = y_test[test_mult]

    return X_train_mult, y_train_mult, X_test_mult, y_test_mult
