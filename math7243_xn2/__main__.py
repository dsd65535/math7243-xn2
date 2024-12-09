# pylint:disable=invalid-name,logging-fstring-interpolation
"""This script outputs the data used in the Modeling section of the final report"""
import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def _get_accuracy_from_cm(cm: np.ndarray) -> float:
    """Get the accuracy from a confusion matrix"""

    return cm.diagonal().sum() / cm.sum()


def _get_feature_count_from_coefs(coefs: np.ndarray) -> int:
    """Get the non-zero feature count from coefficients"""

    return len(set(idx for row in coefs for idx, point in enumerate(row) if point))


def get_data(
    use_ccle: bool = True, cachedirpath: Path = Path("cache")
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Get X and y data"""

    y_name = "primary_disease"
    to_remove = ["Unknown", "Teratoma", "Adrenal Cancer", "Embryonal Cancer"]
    sample_info_csv_filepath = Path("22Q2/sample_info.csv")
    sample_info_on = "DepMap_ID"
    if use_ccle:
        df_csv_filepath = "22Q2/CCLE_expression.csv"
        df_on = "Unnamed: 0"
    else:
        df_csv_filepath = "22Q2/CRISPR_gene_effect.csv"
        df_on = "DepMap_ID"

    df = pd.read_csv(cachedirpath / df_csv_filepath)
    sample_info = pd.read_csv(cachedirpath / sample_info_csv_filepath)

    df.dropna(inplace=True)
    df = df.merge(
        sample_info[[sample_info_on, y_name]],
        left_on=df_on,
        right_on=sample_info_on,
        how="inner",
    )
    for label in to_remove:
        df = df[df[y_name] != label]
    labels, y_data = np.unique(df[y_name], return_inverse=True)
    df = df.select_dtypes(include=[np.number])

    return df.to_numpy(), y_data, labels.tolist(), df.columns.tolist()


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
                accuracy = _get_accuracy_from_cm(result)
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


def plot_pca(
    pca_results: dict[int, BasicResults],
    basic_results: BasicResults,
    outfilepath_log: Path,
    outfilepath_gauss: Path,
    outfilepath_svc: Path,
) -> None:
    """Plot results from PCA Sweep"""

    xaxis = sorted(pca_results.keys())
    all_models = list(pca_results[xaxis[0]].data.keys())

    for outfilepath, models in zip(
        [outfilepath_log, outfilepath_gauss, outfilepath_svc],
        [all_models[:3], all_models[3:5], all_models[5:]],
    ):
        plt.figure(figsize=(10, 10))
        for idx, model in enumerate(models):
            for dataset, fmt in [("train", "--"), ("valid", "-"), ("test", "-.")]:
                plt.semilogx(
                    xaxis,
                    [
                        _get_accuracy_from_cm(
                            pca_results[n_components].data[model][dataset]
                        )
                        for n_components in xaxis
                    ],
                    fmt,
                    label=f"{model} ({dataset})",
                    color=f"C{idx}",
                )
            dataset = "valid"
            fmt = ":"
            y_value = _get_accuracy_from_cm(basic_results.data[model][dataset])
            plt.semilogx(
                [min(xaxis), max(xaxis)],
                [y_value, y_value],
                fmt,
                label=f"{model} ({dataset}, no PCA)",
                color=f"C{idx}",
            )
        plt.legend()
        plt.title("PCA Effect on Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Number of Components")
        plt.savefig(str(outfilepath))


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
                accuracy = _get_accuracy_from_cm(result)
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
            combined_data[C] = contents

    accuracies = {
        dataset: {
            float(C): _get_accuracy_from_cm(results[0][dataset])
            for C, results in sorted(combined_data.items())
        }
        for dataset in ["train", "valid", "test"]
    }
    feature_counts = {
        float(C): _get_feature_count_from_coefs(results[1])
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
    y_value = _get_accuracy_from_cm(basic_results.data["logistic_saga"]["valid"])
    plt.semilogx(
        [min(accuracies.keys()), max(accuracies.keys())],
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
                accuracy = _get_accuracy_from_cm(result)
                print(f"{label},{dataset},{accuracy}")

    def dump_cms(self, outdirpath: Path, labels: list[str], dataset: str) -> None:
        """Dump PNG of the confusion matrices"""

        rows = round(len(labels) ** 0.5)
        cols = (len(labels) - 1) // rows + 1
        _, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows))
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


def train_valid_test_split(
    X_data: np.ndarray,
    y_data: np.ndarray,
    test_size: float,
    valid_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training, validation and testing"""

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=random_state
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=valid_size / (1 - test_size),
        random_state=random_state,
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def main(use_ccle: bool = True) -> None:
    # pylint:disable=too-many-locals,too-many-statements
    """CLI Entry Point"""

    if use_ccle:
        outdirpath = Path("results_ccle")
        seed = 45
    else:
        outdirpath = Path("results_crispr")
        seed = 40

    logging.basicConfig(level=logging.INFO)
    random.seed(seed)
    np.random.seed(seed)

    X_data, y_data, labels, _ = get_data(use_ccle)

    outdirpath.mkdir(parents=True, exist_ok=True)

    logging.info("Running No PCA...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
        X_data, y_data, 0.2, 0.2, seed
    )

    basic_results = BasicResults.load_or_run(
        outdirpath / "basic_results.json",
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    )
    basic_results.print_accuracies()
    basic_results.dump_cms(outdirpath, labels)

    pca_results = {}
    for n_components in np.logspace(0.0, 10.0, 11, base=2.0).astype(int):
        logging.info(f"Running {n_components}-PCA...")
        pca = PCA(n_components=n_components)
        try:
            pca.fit(X_data)
        except Exception:  # pylint:disable=broad-exception-caught
            logging.exception("Failed a PCA fit")
            continue
        X_data_pca = pca.transform(X_data)

        X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
            X_data_pca, y_data, 0.2, 0.2, seed
        )
        try:
            pca_results[n_components] = BasicResults.load_or_run(
                outdirpath / f"{n_components}_pca.json",
                X_train,
                X_valid,
                X_test,
                y_train,
                y_valid,
                y_test,
            )
        except Exception:  # pylint:disable=broad-exception-caught
            logging.exception("Failed a PCA run")
            continue
        pca_results[n_components].print_accuracies()
    plot_pca(
        pca_results,
        basic_results,
        outdirpath / "pca_log.png",
        outdirpath / "pca_gauss.png",
        outdirpath / "pca_svc.png",
    )

    logging.info("Running L1 Sweeps...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
        X_data, y_data, 0.2, 0.2, seed
    )
    l1_sweeps = []
    l1_sweeps.append(
        L1Sweep.load_or_run(
            outdirpath / "l1_sweep.json",
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test,
            -3.0,
            3.0,
            7,
            True,
        )
    )
    l1_sweeps[-1].print_accuracies()
    l1_sweeps.append(
        L1Sweep.load_or_run(
            outdirpath / "l1_sweep_1.json",
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test,
            -1.0,
            1.0,
            7,
            True,
        )
    )
    l1_sweeps[-1].print_accuracies()
    l1_sweeps.append(
        L1Sweep.load_or_run(
            outdirpath / "l1_sweep_2.json",
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test,
            1.0,
            2.5,
            16,
            False,
        )
    )
    l1_sweeps[-1].print_accuracies()

    plot_l1_sweep(l1_sweeps, basic_results, outdirpath / "l1_sweep.png")

    logging.info("Running OVR Sweep...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
        X_data, y_data, 0.2, 0.2, seed
    )
    ovr_results = OneVsRest.load_or_run(
        outdirpath / "one_versus_rest.json",
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        labels,
    )
    ovr_results.print_accuracies(labels)
    ovr_results.dump_cms(outdirpath, labels, "train")
    ovr_results.dump_cms(outdirpath, labels, "valid")
    ovr_results.dump_cms(outdirpath, labels, "test")


if __name__ == "__main__":
    main()
