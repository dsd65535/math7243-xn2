# pylint:disable=invalid-name,logging-fstring-interpolation
"""This module provides the Basic Models"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

from math7243_xn2.basic import get_accuracy_from_cm


class BasicResults:
    """Basic Classifications"""

    def __init__(self, data: dict[str, dict[str, np.ndarray]]):
        self.data = data

    @classmethod
    def run(
        cls,
        X_train: np.ndarray,
        X_valid: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        y_test: np.ndarray,
    ) -> "BasicResults":
        # pylint:disable=too-many-arguments,too-many-positional-arguments
        """Run Tests"""

        one_hot_encoder = OneHotEncoder(
            sparse_output=False,
            categories=[np.unique(np.concatenate([y_train, y_valid, y_test]))],
        )
        y_train_dummy = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
        y_valid_dummy = one_hot_encoder.fit_transform(y_valid.reshape(-1, 1))
        y_test_dummy = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))

        data = {}

        logging.info("Running LinearRegression...")
        model = LinearRegression()
        model.fit(X_train, y_train_dummy)
        print(f"R2 Training Score: {model.score(X_train, y_train_dummy):.3f}")
        print(f"R2 Testing Score: {model.score(X_valid, y_valid_dummy):.3f}")
        print(f"R2 Testing Score: {model.score(X_test, y_test_dummy):.3f}")

        for solver in ["lbfgs", "liblinear", "saga"]:
            logging.info(f"Running LogisticRegression ({solver})...")
            model = LogisticRegression(solver=solver)
            model.fit(X_train, y_train)
            data[f"logistic_{solver}"] = {
                "train": confusion_matrix(y_train, model.predict(X_train)),
                "valid": confusion_matrix(y_valid, model.predict(X_valid)),
                "test": confusion_matrix(y_test, model.predict(X_test)),
            }

        logging.info("Running LinearDiscriminantAnalysis...")
        model = LinearDiscriminantAnalysis(store_covariance=True)
        model.fit(X_train, y_train)
        data["lda"] = {
            "train": confusion_matrix(y_train, model.predict(X_train)),
            "valid": confusion_matrix(y_valid, model.predict(X_valid)),
            "test": confusion_matrix(y_test, model.predict(X_test)),
        }

        logging.info("Running QuadraticDiscriminantAnalysis...")
        model = QuadraticDiscriminantAnalysis(store_covariance=True)
        model.fit(X_train, y_train)
        data["qda"] = {
            "train": confusion_matrix(y_train, model.predict(X_train)),
            "valid": confusion_matrix(y_valid, model.predict(X_valid)),
            "test": confusion_matrix(y_test, model.predict(X_test)),
        }

        for kernel in ["linear", "poly", "rbf", "sigmoid"]:
            logging.info(f"Running SVC ({kernel})...")
            model = SVC(kernel=kernel)
            model.fit(X_train, y_train)
            data[f"svc_{kernel}"] = {
                "train": confusion_matrix(y_train, model.predict(X_train)),
                "valid": confusion_matrix(y_valid, model.predict(X_valid)),
                "test": confusion_matrix(y_test, model.predict(X_test)),
            }

        return cls(data)

    def dump(self, outfilepath: Path) -> None:
        """Dump to file"""

        with outfilepath.open("w", encoding="UTF-8") as outfile:
            json.dump(
                {
                    model: {
                        dataset: result.tolist()
                        for dataset, result in model_results.items()
                    }
                    for model, model_results in self.data.items()
                },
                outfile,
            )

    @classmethod
    def load(cls, infilepath: Path) -> "BasicResults":
        """Load from file"""

        with infilepath.open("r", encoding="UTF-8") as infile:
            data = {
                model: {
                    dataset: np.array(result)
                    for dataset, result in model_results.items()
                }
                for model, model_results in json.load(infile).items()
            }

        return cls(data)

    def print_accuracies(self) -> None:
        """Print Accuracies from Confusion Matrices"""

        for model, model_results in self.data.items():
            for dataset, result in model_results.items():
                accuracy = get_accuracy_from_cm(result)
                print(f"{model},{dataset},{accuracy}")

    def dump_cms(self, outdirpath: Path, labels: list[str]) -> None:
        """Dump PNGs of the confusion matrices"""

        for model, model_results in self.data.items():
            for dataset, result in model_results.items():
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=result, display_labels=labels
                )
                _, ax = plt.subplots(figsize=(10.0, 10.0))
                disp.plot(ax=ax, colorbar=False)
                plt.xticks(rotation=90.0)
                plt.tight_layout()
                plt.savefig(str(outdirpath / f"{model}_{dataset}.png"))

    @classmethod
    def load_or_run(
        cls,
        infilepath: Path,
        X_train: np.ndarray,
        X_valid: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        y_test: np.ndarray,
    ) -> "BasicResults":
        # pylint:disable=too-many-arguments,too-many-positional-arguments
        """Load from file if it exists, otherwise run"""

        if not infilepath.exists():
            cls.run(X_train, X_valid, X_test, y_train, y_valid, y_test).dump(infilepath)

        return cls.load(infilepath)
