# pylint:disable=invalid-name,logging-fstring-interpolation
"""This module provides functions related to L1 Regularization Sweeps"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from math7243_xn2.basic import get_accuracy_from_cm
from math7243_xn2.basic import get_feature_count_from_coefs
from math7243_xn2.basic_models import BasicResults


class L1Sweep:
    """Sweep of L1 Regularization"""

    def __init__(self, data: dict[float, tuple[dict[str, np.ndarray], np.ndarray]]):
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
        min_C: float,
        max_C: float,
        count_C: int,
        log_C: bool,
    ) -> "L1Sweep":
        # pylint:disable=too-many-arguments,too-many-positional-arguments
        """Run Tests"""

        data = {}

        for C in (
            np.logspace(min_C, max_C, count_C)
            if log_C
            else np.linspace(min_C, max_C, count_C)
        ):
            logging.info(f"Running C={C}...")
            model = LogisticRegression(solver="saga", penalty="l1", C=C)
            model.fit(X_train, y_train)
            data[C] = (
                {
                    "train": confusion_matrix(y_train, model.predict(X_train)),
                    "valid": confusion_matrix(y_valid, model.predict(X_valid)),
                    "test": confusion_matrix(y_test, model.predict(X_test)),
                },
                model.coef_,
            )

        return cls(data)

    def dump(self, outfilepath: Path) -> None:
        """Dump to file"""

        with outfilepath.open("w", encoding="UTF-8") as outfile:
            json.dump(
                {
                    model: (
                        {
                            dataset: result.tolist()
                            for dataset, result in model_results.items()
                        },
                        coefs.tolist(),
                    )
                    for model, (model_results, coefs) in self.data.items()
                },
                outfile,
            )

    @classmethod
    def load(cls, infilepath: Path) -> "L1Sweep":
        """Load from file"""

        with infilepath.open("r", encoding="UTF-8") as infile:
            data = {
                model: (
                    {
                        dataset: np.array(result)
                        for dataset, result in model_results.items()
                    },
                    np.array(coefs),
                )
                for model, (model_results, coefs) in json.load(infile).items()
            }

        return cls(data)

    def print_accuracies(self) -> None:
        """Print Accuracies from Confusion Matrices"""

        for model, (model_results, _) in self.data.items():
            for dataset, result in model_results.items():
                accuracy = get_accuracy_from_cm(result)
                print(f"{model},{dataset},{accuracy}")

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
        min_C: float,
        max_C: float,
        count_C: int,
        log_C: bool,
    ) -> "L1Sweep":
        # pylint:disable=too-many-arguments,too-many-positional-arguments
        """Load from file if it exists, otherwise run"""

        if not infilepath.exists():
            cls.run(
                X_train,
                X_valid,
                X_test,
                y_train,
                y_valid,
                y_test,
                min_C,
                max_C,
                count_C,
                log_C,
            ).dump(infilepath)

        return cls.load(infilepath)


def plot_l1_sweep(
    results: list[L1Sweep],
    basic_results: BasicResults,
    outfilepath: Path,
) -> None:
    """Plot the results of the L1 Sweeps"""

    combined_data = {}
    for result in results:
        for C, contents in result.data.items():
            combined_data[float(C)] = contents

    accuracies = {
        dataset: {
            C: get_accuracy_from_cm(results[0][dataset])
            for C, results in sorted(combined_data.items())
        }
        for dataset in ["train", "valid", "test"]
    }
    feature_counts = {
        C: get_feature_count_from_coefs(results[1])
        for C, results in sorted(combined_data.items())
    }

    _, ax1 = plt.subplots(figsize=(10.0, 10.0))

    plt.plot(
        *zip(*accuracies["train"].items()), "--", label="Training Accuracy", color="C0"
    )
    plt.plot(
        *zip(*accuracies["valid"].items()), "-", label="Validation Accuracy", color="C0"
    )
    plt.plot(
        *zip(*accuracies["test"].items()), "-.", label="Testing Accuracy", color="C0"
    )
    y_value = get_accuracy_from_cm(basic_results.data["logistic_saga"]["valid"])
    plt.semilogx(
        [min(accuracies["valid"].keys()), max(accuracies["valid"].keys())],
        [y_value, y_value],
        ":",
        label="Validation Accuracy (no reg.)",
        color="C0",
    )
    ax1.set_xlabel("C")
    ax1.set_ylabel("Accuracy")
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    plt.plot(*zip(*feature_counts.items()), "-", label="Feature Count", color="C1")
    ax2.set_ylabel("Feature Count")
    ax1.legend()
    ax2.legend()
    plt.title("L1 Regularization with SAGA Solver")
    plt.savefig(str(outfilepath))
