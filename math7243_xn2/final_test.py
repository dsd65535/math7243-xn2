# pylint:disable=invalid-name
"""This script outputs the data used in the Modeling section of the final report"""
import json
import logging
import random
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from math7243_xn2.basic import get_data


def run_basic_tests(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """Run basic classifications"""

    results = {}

    logging.info("Running LogisticRegression (lbfgs)...")
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(X_train, y_train)
    results["logistic_lbfgs"] = {
        "train": confusion_matrix(y_train, clf.predict(X_train)),
        "test": confusion_matrix(y_test, clf.predict(X_test)),
    }

    logging.info("Running LogisticRegression (saga)...")
    clf = LogisticRegression(solver="saga")
    clf.fit(X_train, y_train)
    results["logistic_saga"] = {
        "train": confusion_matrix(y_train, clf.predict(X_train)),
        "test": confusion_matrix(y_test, clf.predict(X_test)),
    }

    logging.info("Running LinearDiscriminantAnalysis...")
    clf = LinearDiscriminantAnalysis(store_covariance=True)
    clf.fit(X_train, y_train)
    results["lda"] = {
        "train": confusion_matrix(y_train, clf.predict(X_train)),
        "test": confusion_matrix(y_test, clf.predict(X_test)),
    }

    logging.info("Running QuadraticDiscriminantAnalysis...")
    clf = QuadraticDiscriminantAnalysis(store_covariance=True)
    clf.fit(X_train, y_train)
    results["qda"] = {
        "train": confusion_matrix(y_train, clf.predict(X_train)),
        "test": confusion_matrix(y_test, clf.predict(X_test)),
    }

    logging.info("Running LinearRegression...")
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    results["linear"] = {
        "train": confusion_matrix(y_train, clf.predict(X_train)),
        "test": confusion_matrix(y_test, clf.predict(X_test)),
    }

    logging.info("Running SVR (linear)...")
    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)
    results["svc_linear"] = {
        "train": confusion_matrix(y_train, clf.predict(X_train)),
        "test": confusion_matrix(y_test, clf.predict(X_test)),
    }

    return results


def dump_basic_results(
    basic_results: dict[str, dict[str, np.ndarray]], outfilepath: Path
) -> None:
    """Dump results from basic tests to file"""

    with outfilepath.open("w", encoding="UTF-8") as outfile:
        json.dump(
            {
                test: {
                    dataset: result.tolist() for dataset, result in test_results.items()
                }
                for test, test_results in basic_results.items()
            },
            outfile,
        )


def load_basic_results(infilepath: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load results from basic tests from file"""

    with infilepath.open("r", encoding="UTF-8") as infile:
        basic_results = {
            test: {
                dataset: np.array(result) for dataset, result in test_results.items()
            }
            for test, test_results in json.load(infile).items()
        }

    return basic_results


def print_accuracies(basic_results: dict[str, dict[str, np.ndarray]]) -> None:
    """Print Accuracies from Confusion Matrices"""

    for test, test_results in basic_results.items():
        for dataset, result in test_results.items():
            accuracy = result.diagonal().sum() / result.sum()
            print(f"{test},{dataset},{accuracy}")


def main(outfilepath: Path = Path("basic_results.json"), seed: int = 42) -> None:
    """CLI Entry Point"""

    logging.basicConfig(level=logging.INFO)
    random.seed(seed)
    np.random.seed(seed)

    X_data, y_data, _, _ = get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=41
    )

    basic_results = run_basic_tests(X_train, X_test, y_train, y_test)
    dump_basic_results(basic_results, outfilepath)
    print_accuracies(basic_results)


if __name__ == "__main__":
    main()
