# pylint:disable=invalid-name
"""This script explores PCA and L1 regularization"""
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from .basic import filter_singles
from .basic import get_data
from .basic import process_data
from .initial_test import do_clf
from .initial_test import do_lda
from .initial_test import do_lr
from .initial_test import do_qda


def run_l1_test(
    y_name: str,
    C: float,
    valid_fraction: float = 0.0,
    test_fraction: float = 1.0 / 7,
) -> None:
    """Run an L1 test"""

    X_data, y_data, _ = get_data(y_name)
    (X_train, y_train, _), _, (X_test, y_test, _) = process_data(
        X_data, y_data, valid_fraction, test_fraction
    )

    clf = LogisticRegression(solver="saga", max_iter=100, penalty="l1", C=C)
    clf.fit(X_train, y_train)
    print(f"Logistic Regression Training Score: {clf.score(X_train, y_train):.3f}")
    print(f"Logistic Regression Testing Score: {clf.score(X_test, y_test):.3f}")


def run_pca_test(
    y_name: str,
    pca_components: int,
    valid_fraction: float = 0.0,
    test_fraction: float = 1.0 / 7,
) -> None:
    # pylint:disable=too-many-locals
    """Run a PCA test"""

    X_data, y_data, _ = get_data(y_name)

    pca = PCA(n_components=pca_components)
    pca.fit(X_data)
    X_data = pca.transform(X_data)

    (X_train, y_train, y_train_dummy), _, (X_test, y_test, y_test_dummy) = process_data(
        X_data, y_data, valid_fraction, test_fraction
    )
    X_train_mult, y_train_mult, X_test_mult, y_test_mult = filter_singles(
        X_train, y_train, X_test, y_test
    )

    do_clf(X_train, y_train, X_test, y_test)
    do_lda(X_train, y_train, X_test, y_test)
    do_qda(X_train_mult, y_train_mult, X_test_mult, y_test_mult)
    do_lr(X_train, y_train_dummy, X_test, y_test_dummy)


def main() -> None:
    """Entry Point"""

    for C_e in range(-3, 4):
        C = 10**C_e
        print(f"C: {C}")
        run_l1_test("primary_disease", C)

    for pca_components_e in range(14):
        pca_components = int(2**pca_components_e)
        print(f"PCA: {pca_components}")
        run_pca_test("primary_disease", pca_components)


if __name__ == "__main__":
    main()
