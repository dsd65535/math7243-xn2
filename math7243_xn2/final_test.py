# pylint:disable=invalid-name,logging-fstring-interpolation
"""This script outputs the data used in the Modeling section of the final report"""
import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

from math7243_xn2.basic import get_data


class BasicResults:
    """Basic Classifications"""

    def __init__(self, data: dict[str, dict[str, np.ndarray]]):
        self.data = data

    @classmethod
    def run(
        cls,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> "BasicResults":
        """Run Tests"""

        one_hot_encoder = OneHotEncoder(
            sparse_output=False,
            categories=[np.unique(np.concatenate([y_train, y_test]))],
        )
        y_train_dummy = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
        y_test_dummy = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))

        data = {}

        logging.info("Running LinearRegression...")
        model = LinearRegression()
        model.fit(X_train, y_train_dummy)
        print(f"R2 Training Score: {model.score(X_train, y_train_dummy):.3f}")
        print(f"R2 Testing Score: {model.score(X_test, y_test_dummy):.3f}")

        logging.info("Running LogisticRegression (lbfgs)...")
        model = LogisticRegression(solver="lbfgs")
        model.fit(X_train, y_train)
        data["logistic_lbfgs"] = {
            "train": confusion_matrix(y_train, model.predict(X_train)),
            "test": confusion_matrix(y_test, model.predict(X_test)),
        }

        logging.info("Running LogisticRegression (saga)...")
        model = LogisticRegression(solver="saga")
        model.fit(X_train, y_train)
        data["logistic_saga"] = {
            "train": confusion_matrix(y_train, model.predict(X_train)),
            "test": confusion_matrix(y_test, model.predict(X_test)),
        }

        logging.info("Running LinearDiscriminantAnalysis...")
        model = LinearDiscriminantAnalysis(store_covariance=True)
        model.fit(X_train, y_train)
        data["lda"] = {
            "train": confusion_matrix(y_train, model.predict(X_train)),
            "test": confusion_matrix(y_test, model.predict(X_test)),
        }

        logging.info("Running QuadraticDiscriminantAnalysis...")
        model = QuadraticDiscriminantAnalysis(store_covariance=True)
        model.fit(X_train, y_train)
        data["qda"] = {
            "train": confusion_matrix(y_train, model.predict(X_train)),
            "test": confusion_matrix(y_test, model.predict(X_test)),
        }

        logging.info("Running SVC (linear)...")
        model = SVC(kernel="linear")
        model.fit(X_train, y_train)
        data["svc_linear"] = {
            "train": confusion_matrix(y_train, model.predict(X_train)),
            "test": confusion_matrix(y_test, model.predict(X_test)),
        }

        return cls(data)

    def dump(self, outfilepath: Path) -> None:
        """Dump to file"""

        with outfilepath.open("w", encoding="UTF-8") as outfile:
            json.dump(
                {
                    test: {
                        dataset: result.tolist()
                        for dataset, result in test_results.items()
                    }
                    for test, test_results in self.data.items()
                },
                outfile,
            )

    @classmethod
    def load(cls, infilepath: Path) -> "BasicResults":
        """Load from file"""

        with infilepath.open("r", encoding="UTF-8") as infile:
            data = {
                test: {
                    dataset: np.array(result)
                    for dataset, result in test_results.items()
                }
                for test, test_results in json.load(infile).items()
            }

        return cls(data)

    def print_accuracies(self) -> None:
        """Print Accuracies from Confusion Matrices"""

        for test, test_results in self.data.items():
            for dataset, result in test_results.items():
                accuracy = result.diagonal().sum() / result.sum()
                print(f"{test},{dataset},{accuracy}")

    def dump_cms(self, outdirpath: Path, labels: list[str]) -> None:
        """Dump PNGs of the confusion matrices"""

        for test, test_results in self.data.items():
            for dataset, result in test_results.items():
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=result, display_labels=labels
                )
                _, ax = plt.subplots(figsize=(10.0, 10.0))
                disp.plot(ax=ax, colorbar=False)
                plt.xticks(rotation=90.0)
                plt.tight_layout()
                plt.savefig(str(outdirpath / f"{test}_{dataset}.png"))

    @classmethod
    def load_or_run(
        cls,
        infilepath: Path,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> "BasicResults":
        # pylint:disable=too-many-arguments,too-many-positional-arguments
        """Load from file if it exists, otherwise run"""

        if not infilepath.exists():
            cls.run(X_train, X_test, y_train, y_test).dump(infilepath)

        return cls.load(infilepath)


class L1Sweep:
    """Sweep of L1 Regularization"""

    def __init__(self, data: dict[str, tuple[dict[str, np.ndarray], np.ndarray]]):
        self.data = data

    @classmethod
    def run(
        cls,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
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
                    test: (
                        {
                            dataset: result.tolist()
                            for dataset, result in test_results.items()
                        },
                        coefs.tolist(),
                    )
                    for test, (test_results, coefs) in self.data.items()
                },
                outfile,
            )

    @classmethod
    def load(cls, infilepath: Path) -> "L1Sweep":
        """Load from file"""

        with infilepath.open("r", encoding="UTF-8") as infile:
            data = {
                test: (
                    {
                        dataset: np.array(result)
                        for dataset, result in test_results.items()
                    },
                    np.array(coefs),
                )
                for test, (test_results, coefs) in json.load(infile).items()
            }

        return cls(data)

    def print_accuracies(self) -> None:
        """Print Accuracies from Confusion Matrices"""

        for test, (test_results, _) in self.data.items():
            for dataset, result in test_results.items():
                accuracy = result.diagonal().sum() / result.sum()
                print(f"{test},{dataset},{accuracy}")

    @classmethod
    def load_or_run(
        cls,
        infilepath: Path,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
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
                X_train, X_test, y_train, y_test, min_C, max_C, count_C, log_C
            ).dump(infilepath)

        return cls.load(infilepath)


def main(outdirpath: Path = Path("results"), seed: int = 43) -> None:
    """CLI Entry Point"""

    logging.basicConfig(level=logging.INFO)
    random.seed(seed)
    np.random.seed(seed)

    X_data, y_data, labels, _ = get_data()

    outdirpath.mkdir(parents=True, exist_ok=True)

    logging.info("Running No PCA...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=seed
    )
    basic_results = BasicResults.load_or_run(
        outdirpath / "basic_results.json", X_train, X_test, y_train, y_test
    )
    basic_results.print_accuracies()
    basic_results.dump_cms(outdirpath, labels)

    for n_components in np.logspace(0.0, 11.0, 12, base=2.0).astype(int):
        pca = PCA(n_components=n_components)
        try:
            pca.fit(X_data)
        except Exception:  # pylint:disable=broad-exception-caught
            logging.exception("Failed a PCA fit")
            continue
        X_data_pca = pca.transform(X_data)

        logging.info(f"Running {n_components}-PCA...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_data_pca, y_data, test_size=0.2, random_state=41
        )
        try:
            basic_results = BasicResults.load_or_run(
                outdirpath / f"{n_components}_pca.json",
                X_train,
                X_test,
                y_train,
                y_test,
            )
        except Exception:  # pylint:disable=broad-exception-caught
            logging.exception("Failed a PCA run")
            continue
        basic_results.print_accuracies()

    logging.info("Running L1 Sweeps...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=41
    )
    l1_sweep = L1Sweep.load_or_run(
        outdirpath / "l1_sweep.json",
        X_train,
        X_test,
        y_train,
        y_test,
        -3.0,
        3.0,
        7,
        True,
    )
    l1_sweep.print_accuracies()
    l1_sweep = L1Sweep.load_or_run(
        outdirpath / "l1_sweep_1.json",
        X_train,
        X_test,
        y_train,
        y_test,
        -1.0,
        1.0,
        7,
        True,
    )
    l1_sweep.print_accuracies()
    l1_sweep = L1Sweep.load_or_run(
        outdirpath / "l1_sweep_2.json",
        X_train,
        X_test,
        y_train,
        y_test,
        1.0,
        2.5,
        16,
        False,
    )
    l1_sweep.print_accuracies()


if __name__ == "__main__":
    main()
