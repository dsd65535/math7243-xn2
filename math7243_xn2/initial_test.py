# pylint:disable=invalid-name
"""This script fits some basic models to CRISPR_gene_effect vs lineage"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from .basic import filter_singles
from .basic import get_data
from .basic import process_data


def do_clf(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """Perform Logistic Regression"""

    clf = LogisticRegression(solver="lbfgs", max_iter=100)
    clf.fit(X_train, y_train)
    print(f"Logistic Regression Training Score: {clf.score(X_train, y_train):.3f}")
    print(f"Logistic Regression Testing Score: {clf.score(X_test, y_test):.3f}")


def do_lda(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """Perform Linear Discriminant Analysis"""

    lda = LinearDiscriminantAnalysis(store_covariance=True)
    lda.fit(X_train, y_train)
    print(f"LDA Training Score: {lda.score(X_train, y_train):.3f}")
    print(f"LDA Testing Score: {lda.score(X_test, y_test):.3f}")


def do_qda(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """Perform Quadratic Discriminant Analysis"""

    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(X_train, y_train)
    print(f"QDA Training Score: {qda.score(X_train, y_train):.3f}")
    print(f"QDA Testing Score: {qda.score(X_test, y_test):.3f}")


def do_lr(
    X_train: np.ndarray,
    y_train_dummy: np.ndarray,
    X_test: np.ndarray,
    y_test_dummy: np.ndarray,
) -> None:
    """Perform Linear Regression"""

    lr = LinearRegression()
    lr.fit(X_train, y_train_dummy)
    print(f"R2 Training Score: {lr.score(X_train, y_train_dummy):.3f}")
    print(f"R2 Testing Score: {lr.score(X_test, y_test_dummy):.3f}")


def run_initial_test(
    y_name: str, valid_fraction: float = 0.0, test_fraction: float = 1.0 / 7
) -> None:
    """Run an initial test"""

    X_data, y_data, _ = get_data(y_name)
    (X_train, y_train, y_train_dummy), _, (X_test, y_test, y_test_dummy) = process_data(
        X_data, y_data, valid_fraction, test_fraction
    )
    X_train_mult, y_train_mult, X_test_mult, y_test_mult = filter_singles(
        X_train, y_train, X_test, y_test
    )

    print(f"Training Samples: {len(y_train)}")
    print(f"Testing Samples: {len(y_test)}")

    do_clf(X_train, y_train, X_test, y_test)
    do_lda(X_train, y_train, X_test, y_test)
    do_qda(X_train_mult, y_train_mult, X_test_mult, y_test_mult)
    do_lr(X_train, y_train_dummy, X_test, y_test_dummy)


def main() -> None:
    """Entry Point"""

    run_initial_test("lineage")
    run_initial_test("primary_disease")


if __name__ == "__main__":
    main()
