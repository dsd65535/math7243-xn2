"""This module provides functions related to PCA"""

from pathlib import Path

import matplotlib.pyplot as plt

from math7243_xn2.basic import get_accuracy_from_cm
from math7243_xn2.basic_models import BasicResults


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
                        get_accuracy_from_cm(
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
            y_value = get_accuracy_from_cm(basic_results.data[model][dataset])
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


def print_best_pca(pca_results: dict[int, BasicResults]) -> None:
    """Print the best results from PCA Sweep"""

    for model in list(pca_results.values())[0].data.keys():
        for dataset in ["train", "valid", "test"]:
            results = [
                (
                    n_components,
                    get_accuracy_from_cm(basic_results.data[model][dataset]),
                )
                for n_components, basic_results in pca_results.items()
            ]
            best = sorted(results, key=lambda x: x[1])[-1]
            print(f"{model},{dataset},{best[0]},{best[1]}")
