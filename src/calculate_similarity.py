"""Calculate the dot-product similarity between two histograms."""

import argparse
from pathlib import Path

import numpy as np

from experiment_utils import (
    experiment_directory,
    load_experiment_config,
    validate_index,
)


SUBSETS_BY_SPLIT = {
    "validation": ("enrollment", "probe"),
    "test": ("probe",),
}


def load_histogram_dimension(experiment: int) -> int:
    """Load and validate D from the experiment config."""
    return load_experiment_config(experiment).codeword_count


def validate_location(split: str, subset: str) -> None:
    """Validate a split/subset pair."""
    if split not in SUBSETS_BY_SPLIT:
        raise ValueError(
            f"分割は{tuple(SUBSETS_BY_SPLIT)}から指定してください"
        )
    if subset not in SUBSETS_BY_SPLIT[split]:
        raise ValueError(
            f"{split}には{subset}がありません"
        )


def histogram_path(
    experiment: int,
    subject: int,
    prompt: int,
    split: str,
    subset: str,
) -> Path:
    """Build a histogram path from experiment and index values."""
    validate_index("被験者番号", subject)
    validate_index("プロンプト番号", prompt)
    validate_location(split, subset)
    filename = f"s{subject:02d}_p{prompt:02d}.npy"
    return (
        experiment_directory(experiment)
        / "histograms"
        / split
        / subset
        / filename
    )


def load_histogram(
    path: Path,
    expected_dimension: int,
) -> np.ndarray:
    """Load and validate one D-dimensional histogram."""
    if not path.is_file():
        raise FileNotFoundError(f"ヒストグラムが見つかりません: {path}")

    try:
        histogram = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"ヒストグラムを読み込めません: {path}"
        ) from error

    if histogram.ndim != 1:
        raise ValueError(
            f"{path} は1次元配列である必要があります: "
            f"shape={histogram.shape}"
        )
    if histogram.size != expected_dimension:
        raise ValueError(
            f"{path} のヒストグラム次元が不正です: "
            f"{histogram.size}（期待値: {expected_dimension}）"
        )

    try:
        numeric_histogram = histogram.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path} に数値へ変換できない値があります"
        ) from error

    if not np.isfinite(numeric_histogram).all():
        raise ValueError(
            f"{path} にNaNまたはinfが含まれています"
        )
    return numeric_histogram


def calculate_similarity_from_paths(
    experiment: int,
    histogram1_path: Path,
    histogram2_path: Path,
) -> float:
    """Load two explicitly selected histograms and return their dot product."""
    dimension = load_histogram_dimension(experiment)
    histogram1 = load_histogram(Path(histogram1_path), dimension)
    histogram2 = load_histogram(Path(histogram2_path), dimension)
    return float(np.dot(histogram1, histogram2))


def calculate_similarity(
    experiment: int,
    subject1: int,
    prompt1: int,
    subject2: int,
    prompt2: int,
    split1: str = "validation",
    subset1: str = "enrollment",
    split2: str = "validation",
    subset2: str = "probe",
) -> float:
    """Load two histograms and return their dot-product similarity."""
    dimension = load_histogram_dimension(experiment)

    histogram1_path = histogram_path(
        experiment,
        subject1,
        prompt1,
        split1,
        subset1,
    )
    histogram2_path = histogram_path(
        experiment,
        subject2,
        prompt2,
        split2,
        subset2,
    )

    histogram1 = load_histogram(
        histogram1_path,
        dimension,
    )
    histogram2 = load_histogram(
        histogram2_path,
        dimension,
    )
    return float(np.dot(histogram1, histogram2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2つのヒストグラムの内積類似度を計算します。",
    )
    parser.add_argument(
        "experiment",
        type=int,
        help="実験番号（例: 1 -> exp01）",
    )
    parser.add_argument("subject1", type=int, help="第1ヒストグラムの被験者番号")
    parser.add_argument("prompt1", type=int, help="第1ヒストグラムのプロンプト番号")
    parser.add_argument("subject2", type=int, help="第2ヒストグラムの被験者番号")
    parser.add_argument("prompt2", type=int, help="第2ヒストグラムのプロンプト番号")
    parser.add_argument(
        "--split1",
        choices=("validation", "test"),
        default="validation",
        help="第1ヒストグラムの分割（既定値: validation）",
    )
    parser.add_argument(
        "--subset1",
        choices=("enrollment", "probe"),
        default="enrollment",
        help="第1ヒストグラムのサブセット（既定値: enrollment）",
    )
    parser.add_argument(
        "--split2",
        choices=("validation", "test"),
        default="validation",
        help="第2ヒストグラムの分割（既定値: validation）",
    )
    parser.add_argument(
        "--subset2",
        choices=("enrollment", "probe"),
        default="probe",
        help="第2ヒストグラムのサブセット（既定値: probe）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        similarity = calculate_similarity(
            experiment=args.experiment,
            subject1=args.subject1,
            prompt1=args.prompt1,
            subject2=args.subject2,
            prompt2=args.prompt2,
            split1=args.split1,
            subset1=args.subset1,
            split2=args.split2,
            subset2=args.subset2,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"エラー: {error}") from error
    print(similarity)


if __name__ == "__main__":
    main()