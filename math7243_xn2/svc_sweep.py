# pylint:disable=invalid-name
"""This script sweeps SVC parameters"""
from sklearn.svm import SVC

from math7243_xn2.basic import get_data
from math7243_xn2.basic import process_data


def main() -> None:
    """Entry Point"""

    valid_fraction = 0.0
    test_fraction = 1.0 / 7

    X_data, y_data, _, _ = get_data()
    (X_train, y_train, _), _, (X_test, y_test, _) = process_data(
        X_data, y_data, valid_fraction, test_fraction
    )

    for decision_function_shape in ["ovo", "ovr"]:
        for kernel in ["linear", "poly", "rbf", "sigmoid", "precomputed"]:
            for C10 in range(1, 11):
                svc = SVC(
                    C=C10 / 10,
                    kernel=kernel,
                    decision_function_shape=decision_function_shape,
                )
                svc.fit(X_train, y_train)
                print(
                    f"{decision_function_shape},{kernel},{C10/10},"
                    f"{svc.score(X_train, y_train)},{svc.score(X_test, y_test)}"
                )


if __name__ == "__main__":
    main()
