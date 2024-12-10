# pylint:disable=invalid-name,logging-fstring-interpolation,duplicate-code
"""This script outputs the data used in the Modeling section of the final report"""
import logging
import random
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from math7243_xn2.basic import get_data
from math7243_xn2.basic import train_valid_test_split
from math7243_xn2.basic_models import BasicResults
from math7243_xn2.l1_sweep import L1Sweep
from math7243_xn2.l1_sweep import plot_l1_sweep
from math7243_xn2.ovr import OneVsRest
from math7243_xn2.pca import plot_pca
from math7243_xn2.pca import print_best_pca


def main(use_ccle: bool = True) -> None:
    # pylint:disable=too-many-locals
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
    print_best_pca(pca_results)

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
            outdirpath / "l1_sweep_zoom.json",
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test,
            -2.0,
            1.0,
            13,
            True,
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
