"""Encode validation and test samples as BoW histograms."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances_argmin

from experiment_utils import (
    FRAME_FEATURE_COUNT,
    experiment_directory,
    load_experiment_config,
)


SPLITS = ("validation", "test")
SUBSETS = ("enrollment", "probe")
SUBSETS_BY_SPLIT = {
    "validation": ("enrollment", "probe"),
    "test": ("probe",),
}


def find_input_samples(
    samples_directory: Path,
    split: str,
    subset: str,
) -> list[Path]:
    """Find samples for one split and subset in deterministic order."""
    input_directory = samples_directory / split / subset
    if not input_directory.is_dir():
        return []
    return sorted(
        path
        for path in input_directory.rglob("*.npy")
        if path.is_file()
    )


def load_codebook(
    codebook_path: Path,
    codeword_count: int,
    expected_dimension: int,
) -> np.ndarray:
    """Load and validate the saved codebook."""
    if not codebook_path.is_file():
        raise FileNotFoundError(f"コードブックが見つかりません: {codebook_path}")

    try:
        codebook = np.load(codebook_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"コードブックを読み込めません: {codebook_path}"
        ) from error

    if codebook.ndim != 2:
        raise ValueError(
            f"コードブックは2次元配列である必要があります: "
            f"shape={codebook.shape}"
        )
    if codebook.shape != (codeword_count, expected_dimension):
        raise ValueError(
            f"コードブックの形状が不正です: {codebook.shape} "
            f"(期待値: {(codeword_count, expected_dimension)})"
        )

    try:
        numeric_codebook = codebook.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("コードブックに数値へ変換できない値があります") from error
    if not np.isfinite(numeric_codebook).all():
        raise ValueError("コードブックにNaNまたはinfが含まれています")
    return numeric_codebook


def load_sample(sample_path: Path, expected_dimension: int) -> np.ndarray:
    """Load and validate one temporal descriptor sample."""
    try:
        sample = np.load(sample_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"入力サンプルを読み込めません: {sample_path}"
        ) from error

    if sample.ndim != 2:
        raise ValueError(
            f"{sample_path} は2次元配列である必要があります: "
            f"shape={sample.shape}"
        )
    if sample.shape[1] != expected_dimension:
        raise ValueError(
            f"{sample_path} の特徴量次元が不正です: {sample.shape[1]} "
            f"(期待値: {expected_dimension})"
        )

    try:
        numeric_sample = sample.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{sample_path} に数値へ変換できない値があります"
        ) from error
    if len(numeric_sample) == 0:
        raise ValueError(f"{sample_path} に時間窓がありません")
    if not np.isfinite(numeric_sample).all():
        raise ValueError(f"{sample_path} にNaNまたはinfが含まれています")
    return numeric_sample


def encode_codewords(
    sample: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    """Assign each temporal descriptor to its nearest codeword."""
    return pairwise_distances_argmin(sample, codebook)


def encode_sample(
    sample: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    """Assign windows to codewords and return an L2-normalized histogram."""
    codeword_ids = encode_codewords(sample, codebook)
    counts = np.bincount(codeword_ids, minlength=len(codebook)).astype(float)
    return counts / np.linalg.norm(counts)


def histogram_output_path(
    sample_path: Path,
    samples_directory: Path,
    histograms_directory: Path,
    split: str,
    subset: str,
) -> Path:
    """Map one sample to the corresponding histogram path."""
    relative_path = sample_path.relative_to(samples_directory / split / subset)
    return histograms_directory.joinpath(split, subset, relative_path)


def select_groups(
    split: str,
    subset: str,
) -> list[tuple[str, str]]:
    """Return valid split/subset pairs and reject invalid combinations."""
    selected_splits = SPLITS if split == "all" else (split,)
    selected_groups = []
    for current_split in selected_splits:
        available_subsets = SUBSETS_BY_SPLIT[current_split]
        selected_subsets = (
            available_subsets if subset == "all" else (subset,)
        )
        invalid_subsets = set(selected_subsets) - set(available_subsets)
        if invalid_subsets:
            invalid_subset = sorted(invalid_subsets)[0]
            raise ValueError(
                f"{current_split}には{invalid_subset}がありません"
            )
        selected_groups.extend(
            (current_split, current_subset)
            for current_subset in selected_subsets
        )
    return selected_groups


def encode_histograms(
    experiment: int,
    split: str = "all",
    subset: str = "all",
    force: bool = False,
) -> list[Path]:
    """Encode selected validation/test samples for one experiment."""
    config = load_experiment_config(experiment)
    experiment_dir = experiment_directory(experiment)
    samples_directory = experiment_dir / "samples"
    histograms_directory = experiment_dir / "histograms"
    expected_dimension = FRAME_FEATURE_COUNT * config.window_length
    codebook = load_codebook(
        experiment_dir / "models" / "codebook.npy",
        config.codeword_count,
        expected_dimension,
    )

    groups = [
        (current_split, current_subset, sample_path)
        for current_split, current_subset in select_groups(split, subset)
        for sample_path in find_input_samples(
            samples_directory,
            current_split,
            current_subset,
        )
    ]
    if not groups:
        raise FileNotFoundError("対象のvalidation/test入力サンプルがありません")

    output_paths = []
    skipped_count = 0
    for current_split, current_subset, sample_path in groups:
        output_path = histogram_output_path(
            sample_path,
            samples_directory,
            histograms_directory,
            current_split,
            current_subset,
        )
        if output_path.exists() and not force:
            print(f"スキップ（既存）: {output_path}")
            skipped_count += 1
            continue

        sample = load_sample(sample_path, expected_dimension)
        histogram = encode_sample(sample, codebook)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, histogram)
        output_paths.append(output_path)
        print(f"{sample_path} -> {output_path}")

    print(
        f"{len(output_paths)}ファイルを保存し、"
        f"{skipped_count}ファイルをスキップしました"
    )
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="validation/testの入力サンプルをBoWヒストグラムに変換します。"
    )
    parser.add_argument(
        "experiment",
        type=int,
        help="実験番号（例: 1 -> experiments/exp01）",
    )
    parser.add_argument(
        "--split",
        choices=("all", *SPLITS),
        default="all",
        help="処理対象の分割（既定値: all）",
    )
    parser.add_argument(
        "--subset",
        choices=("all", *SUBSETS),
        default="all",
        help="処理対象のサブセット（既定値: all）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存のヒストグラムを上書きする",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        encode_histograms(
            experiment=args.experiment,
            split=args.split,
            subset=args.subset,
            force=args.force,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"エラー: {error}") from error


if __name__ == "__main__":
    main()
