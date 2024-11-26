# pylint:disable=invalid-name,fixme
"""This script fits some basic models to CRISPR_gene_effect vs lineage"""
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


def get_data(y_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Get X and y data"""

    crispr_gene_effect = pd.read_csv("cache/22Q2/CRISPR_gene_effect.csv")
    sample_info = pd.read_csv("cache/22Q2/sample_info.csv")

    crispr_gene_effect_nona = crispr_gene_effect.dropna()
    crispr_gene_effect_meta = pd.merge(
        crispr_gene_effect_nona, sample_info, on="DepMap_ID", how="inner"
    )

    labels = []
    indices = []
    for _, row in crispr_gene_effect_meta.iterrows():
        # TODO: don't iterate through rows
        lineage = row[y_name]
        if lineage not in labels:
            labels.append(lineage)
        indices.append(labels.index(lineage))

    X_data = crispr_gene_effect_nona.select_dtypes(np.number).to_numpy()
    y_data = np.array(indices)

    return X_data, y_data, labels


def filter_singles(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove categories with a single member"""

    label_counts: dict[float, int] = {}
    for label in y_train:
        if label in label_counts:
            label_counts[float(label)] += 1
        else:
            label_counts[float(label)] = 1

    train_mult = pd.DataFrame(y_train).apply(
        lambda x: label_counts.get(float(x.to_numpy()), 0) > 1, axis=1
    )
    X_train_mult = X_train[train_mult]
    y_train_mult = y_train[train_mult]
    test_mult = pd.DataFrame(y_test).apply(
        lambda x: label_counts.get(float(x.to_numpy()), 0) > 1, axis=1
    )
    X_test_mult = X_test[test_mult]
    y_test_mult = y_test[test_mult]

    return X_train_mult, y_train_mult, X_test_mult, y_test_mult


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


def run_initial_test(y_name: str, test_fraction: float = 1.0 / 7) -> None:
    # pylint:disable=too-many-locals
    """Run an initial test"""

    X_data, y_data, _ = get_data(y_name)
    y_data_dummy = pd.get_dummies(y_data).to_numpy()

    test_count = round(len(y_data) * test_fraction)
    X_train = X_data[test_count:]
    y_train = y_data[test_count:]
    y_train_dummy = y_data_dummy[test_count:]
    X_test = X_data[:test_count]
    y_test = y_data[:test_count]
    y_test_dummy = y_data_dummy[:test_count]
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
