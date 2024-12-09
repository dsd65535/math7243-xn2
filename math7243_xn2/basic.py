# pylint:disable=invalid-name
"""This module provides some common functions"""
from pathlib import Path

import numpy as np
import pandas as pd


def get_data(
    use_ccle: bool = True, cachedirpath: Path = Path("cache")
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Get X and y data"""

    y_name = "primary_disease"
    to_remove = ["Unknown", "Teratoma", "Adrenal Cancer", "Embryonal Cancer"]
    sample_info_csv_filepath = Path("22Q2/sample_info.csv")
    sample_info_on = "DepMap_ID"
    if use_ccle:
        df_csv_filepath = "22Q2/CCLE_expression.csv"
        df_on = "Unnamed: 0"
    else:
        df_csv_filepath = "22Q2/CRISPR_gene_effect.csv"
        df_on = "DepMap_ID"

    df = pd.read_csv(cachedirpath / df_csv_filepath)
    sample_info = pd.read_csv(cachedirpath / sample_info_csv_filepath)

    df.dropna(inplace=True)
    df = df.merge(
        sample_info[[sample_info_on, y_name]],
        left_on=df_on,
        right_on=sample_info_on,
        how="inner",
    )
    for label in to_remove:
        df = df[df[y_name] != label]
    labels, y_data = np.unique(df[y_name], return_inverse=True)
    df = df.select_dtypes(include=[np.number])

    return df.to_numpy(), y_data, labels.tolist(), df.columns.tolist()


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
