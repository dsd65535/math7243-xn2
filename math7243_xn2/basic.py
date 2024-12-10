# pylint:disable=invalid-name
"""This module provides some common functions"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_accuracy_from_cm(cm: np.ndarray) -> float:
    """Get the accuracy from a confusion matrix"""

    return cm.diagonal().sum() / cm.sum()


def get_feature_count_from_coefs(coefs: np.ndarray) -> int:
    """Get the non-zero feature count from coefficients"""

    return len(set(idx for row in coefs for idx, point in enumerate(row) if point))


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


def train_valid_test_split(
    X_data: np.ndarray,
    y_data: np.ndarray,
    test_size: float,
    valid_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training, validation and testing"""

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=random_state
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=valid_size / (1 - test_size),
        random_state=random_state,
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test
