# pylint:disable=invalid-name
"""This script outputs the data used in the Modeling section of the final report"""
import json
import logging
import random
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
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

        data = {}

        logging.info("Running LogisticRegression (lbfgs)...")
        clf = LogisticRegression(solver="lbfgs")
        clf.fit(X_train, y_train)
        data["logistic_lbfgs"] = {
            "train": confusion_matrix(y_train, clf.predict(X_train)),
            "test": confusion_matrix(y_test, clf.predict(X_test)),
        }

        logging.info("Running LogisticRegression (saga)...")
        clf = LogisticRegression(solver="saga")
        clf.fit(X_train, y_train)
        data["logistic_saga"] = {
            "train": confusion_matrix(y_train, clf.predict(X_train)),
            "test": confusion_matrix(y_test, clf.predict(X_test)),
        }

        logging.info("Running LinearDiscriminantAnalysis...")
        clf = LinearDiscriminantAnalysis(store_covariance=True)
        clf.fit(X_train, y_train)
        data["lda"] = {
            "train": confusion_matrix(y_train, clf.predict(X_train)),
            "test": confusion_matrix(y_test, clf.predict(X_test)),
        }

        logging.info("Running QuadraticDiscriminantAnalysis...")
        clf = QuadraticDiscriminantAnalysis(store_covariance=True)
        clf.fit(X_train, y_train)
        data["qda"] = {
            "train": confusion_matrix(y_train, clf.predict(X_train)),
            "test": confusion_matrix(y_test, clf.predict(X_test)),
        }

        logging.info("Running LinearRegression...")
        clf = LinearRegression()
        clf.fit(X_train, y_train)
        data["linear"] = {
            "train": confusion_matrix(y_train, clf.predict(X_train)),
            "test": confusion_matrix(y_test, clf.predict(X_test)),
        }

        logging.info("Running SVR (linear)...")
        clf = SVC(kernel="linear")
        clf.fit(X_train, y_train)
        data["svc_linear"] = {
            "train": confusion_matrix(y_train, clf.predict(X_train)),
            "test": confusion_matrix(y_test, clf.predict(X_test)),
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


def main(seed: int = 42) -> None:
    """CLI Entry Point"""

    logging.basicConfig(level=logging.INFO)
    random.seed(seed)
    np.random.seed(seed)

    X_data, y_data, _, _ = get_data()

    logging.info("Running No PCA...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=41
    )
    basic_results = BasicResults.run(X_train, X_test, y_train, y_test)
    basic_results.dump(Path("basic_results.json"))
    basic_results.print_accuracies()

    for n_components_exp in range(14):
        n_components = int(2**n_components_exp)
        pca = PCA(n_components=n_components)
        pca.fit(X_data)
        X_data_pca = pca.transform(X_data)

        logging.info("Running {n_components}-PCA...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_data_pca, y_data, test_size=0.2, random_state=41
        )
        try:
            basic_results = BasicResults.run(X_train, X_test, y_train, y_test)
        except Exception:  # pylint:disable=broad-exception-caught
            logging.exception("Failed a run")
            continue
        basic_results.dump(Path(f"{n_components}_pca.json"))
        basic_results.print_accuracies()


if __name__ == "__main__":
    main()
