#!/usr/bin/env python3
"""Run the 192 BERTopic configurations described in the manuscript.

The search varies 4 x 3 x 2 x 2 x 2 x 2 settings:
UMAP neighbours, UMAP components, UMAP min_dist, HDBSCAN min_samples,
HDBSCAN selection method, and BERTopic outlier reduction on/off.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=str(repo_root))
    parser.add_argument("--data_file", default="topic_modeling/meta_data.csv")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script = Path(__file__).with_name("bertopic_cluster.py")

    grid = list(
        itertools.product(
            [10, 15, 20, 25],  # UMAP n_neighbors
            [5, 8, 10],        # UMAP n_components
            [0.0, 0.1],        # UMAP min_dist
            [3, 5],            # HDBSCAN min_samples
            ["eom", "leaf"],  # cluster selection
            [False, True],     # reduce_outliers
        )
    )
    assert len(grid) == 192

    for index, (neighbors, components, min_dist, min_samples, selection, reduce) in enumerate(grid, 1):
        command = [
            sys.executable,
            str(script),
            "--base_dir", args.base_dir,
            "--data_file", args.data_file,
            "--n_topics", "auto",
            "--umap_n_neighbors", str(neighbors),
            "--umap_n_components", str(components),
            "--umap_min_dist", str(min_dist),
            "--hdbscan_min_cluster_size", "10",
            "--hdbscan_min_samples", str(min_samples),
            "--hdbscan_cluster_selection", selection,
            "--metrics_only",
            "--skip_visualization",
        ]
        if args.model_path:
            command.extend(["--model_path", args.model_path])
        if reduce:
            command.extend([
                "--reduce_outliers",
                "--outlier_strategy", "distributions",
                "--outlier_threshold", "0.1",
            ])

        print(f"[{index:03d}/192] {' '.join(command)}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode and not args.continue_on_error:
            return result.returncode

    compare_script = Path(__file__).with_name("compare_experiments.py")
    return subprocess.run(
        [sys.executable, str(compare_script), "--base_dir", args.base_dir],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
