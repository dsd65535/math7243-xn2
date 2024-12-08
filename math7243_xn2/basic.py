# pylint:disable=invalid-name
"""This module provides some common functions"""
import numpy as np
import pandas as pd


def get_data(
    y_name: str = "primary_disease",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Get X and y data"""

    crispr_gene_effect = pd.read_csv("cache/22Q2/CRISPR_gene_effect.csv")
    sample_info = pd.read_csv("cache/22Q2/sample_info.csv")
    crispr_gene_effect.dropna(inplace=True)
    crispr_gene_effect = crispr_gene_effect.merge(
        sample_info[["DepMap_ID", y_name]], on="DepMap_ID", how="inner"
    )
    labels, y_data = np.unique(crispr_gene_effect[y_name], return_inverse=True)
    crispr_gene_effect = crispr_gene_effect.select_dtypes(include=[np.number])

    return (
        crispr_gene_effect.to_numpy(),
        y_data,
        labels.tolist(),
        crispr_gene_effect.columns.tolist(),
    )


def process_data(
    X_data: np.ndarray,
    y_data: np.ndarray,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    # pylint:disable=too-many-locals
    """Split data up into training, validation and testing"""

    valid_and_test_fraction = valid_fraction + test_fraction

    if valid_fraction < 0:
        raise ValueError("Validation Fraction can't be less than 0")
    if test_fraction < 0:
        raise ValueError("Testing Fraction can't be less than 0")
    if valid_and_test_fraction > 1:
        raise ValueError("Validation and Testing Fraction can't add to more than 1")

    # Do this first to capture all categories
    y_data_dummy = pd.get_dummies(y_data).to_numpy()

    test_count = round(len(y_data) * test_fraction)
    valid_and_test_count = round(len(y_data) * valid_and_test_fraction)

    X_train = X_data[valid_and_test_count:]
    y_train = y_data[valid_and_test_count:]
    y_train_dummy = y_data_dummy[valid_and_test_count:]

    X_valid = X_data[test_count:valid_and_test_count]
    y_valid = y_data[test_count:valid_and_test_count]
    y_valid_dummy = y_data_dummy[test_count:valid_and_test_count]

    X_test = X_data[:test_count]
    y_test = y_data[:test_count]
    y_test_dummy = y_data_dummy[:test_count]

    return (
        (X_train, y_train, y_train_dummy),
        (X_valid, y_valid, y_valid_dummy),
        (X_test, y_test, y_test_dummy),
    )


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
