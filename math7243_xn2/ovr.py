# pylint:disable=invalid-name,logging-fstring-interpolation
"""This module provides the OneVsReset Sweep"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC

from math7243_xn2.basic import get_accuracy_from_cm


class OneVsRest:
    """One-Versus-Rest Sweep"""

    def __init__(self, data: list[dict[str, np.ndarray]]):
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
        labels: list[str],
    ) -> "OneVsRest":
        # pylint:disable=too-many-arguments,too-many-positional-arguments
        """Run Tests"""

        data = []

        for idx, label in enumerate(labels):
            logging.info(f"Running {label}...")
            model = SVC(kernel="linear")
            model.fit(X_train, y_train == idx)
            data.append(
                {
                    "train": confusion_matrix(y_train == idx, model.predict(X_train)),
                    "valid": confusion_matrix(y_valid == idx, model.predict(X_valid)),
                    "test": confusion_matrix(y_test == idx, model.predict(X_test)),
                }
            )

        return cls(data)

    def dump(self, outfilepath: Path) -> None:
        """Dump to file"""

        with outfilepath.open("w", encoding="UTF-8") as outfile:
            json.dump(
                [
                    {dataset: result.tolist() for dataset, result in results.items()}
                    for results in self.data
                ],
                outfile,
            )

    @classmethod
    def load(cls, infilepath: Path) -> "OneVsRest":
        """Load from file"""

        with infilepath.open("r", encoding="UTF-8") as infile:
            data = [
                {dataset: np.array(result) for dataset, result in results.items()}
                for results in json.load(infile)
            ]

        return cls(data)

    def print_accuracies(self, labels: list[str]) -> None:
        """Print Accuracies from Confusion Matrices"""

        for label, results in zip(labels, self.data):
            for dataset, result in results.items():
                accuracy = get_accuracy_from_cm(result)
                print(f"{label},{dataset},{accuracy}")

    def dump_cms(self, outdirpath: Path, labels: list[str], dataset: str) -> None:
        """Dump PNG of the confusion matrices"""

        rows = round(len(labels) ** 0.5)
        cols = (len(labels) - 1) // rows + 1
        _, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows), dpi=300)
        for label, results, ax in zip(labels, self.data, axes.flatten()):
            result = results[dataset]
            ax.set_title(label, fontsize=10)
            try:
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=result, display_labels=["Neg.", "Pos."]
                )
                disp.plot(ax=ax, colorbar=False)
            except Exception:  # pylint:disable=broad-exception-caught
                ax.axis("off")
                continue

            ax.set_aspect("auto")
            ax.tick_params(axis="x", labelrotation=45, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)

        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        plt.tight_layout()
        plt.savefig(str(outdirpath / f"ovr_{dataset}.png"))

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
        labels: list[str],
    ) -> "OneVsRest":
        # pylint:disable=too-many-arguments,too-many-positional-arguments
        """Load from file if it exists, otherwise run"""

        if not infilepath.exists():
            cls.run(X_train, X_valid, X_test, y_train, y_valid, y_test, labels).dump(
                infilepath
            )

        return cls.load(infilepath)
