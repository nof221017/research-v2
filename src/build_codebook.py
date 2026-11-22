"""Build a BoW codebook from training samples for one experiment."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from experiment_utils import (
    FRAME_FEATURE_COUNT,
    experiment_directory,
    load_experiment_config,
)


def find_training_samples(samples_directory: Path) -> list[Path]:
    """Find training sample files in deterministic order."""
    if not samples_directory.is_dir():
        raise FileNotFoundError(
            f"trainingサンプルのディレクトリが見つかりません: "
            f"{samples_directory}"
        )

    sample_paths = sorted(
        path
        for path in samples_directory.rglob("*.npy")
        if path.is_file()
    )
    if not sample_paths:
        raise FileNotFoundError(
            f"trainingサンプルが見つかりません: {samples_directory}"
        )
    return sample_paths


def load_training_descriptors(
    sample_paths: list[Path],
    expected_dimension: int,
) -> np.ndarray:
    """Load, validate, and concatenate all finite training descriptors."""
    descriptors = []
    for sample_path in sample_paths:
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
                f"{sample_path} の特徴量次元が不正です: "
                f"{sample.shape[1]}（期待値: {expected_dimension}）"
            )

        try:
            numeric_sample = sample.astype(float, copy=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{sample_path} に数値へ変換できない値があります"
            ) from error

        finite_sample = numeric_sample[np.isfinite(numeric_sample).all(axis=1)]
        if len(finite_sample):
            descriptors.append(finite_sample)

    if not descriptors:
        raise ValueError("有効なtrainingサンプルがありません")
    return np.concatenate(descriptors, axis=0)


def fit_codebook(
    descriptors: np.ndarray,
    codeword_count: int,
    random_state: int,
) -> np.ndarray:
    """Fit K-means and return codeword centers."""
    if len(descriptors) < codeword_count:
        raise ValueError(
            f"有効な時間窓が{len(descriptors)}個しかないため、"
            f"D={codeword_count}を学習できません"
        )

    model = KMeans(
        n_clusters=codeword_count,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    model.fit(descriptors)
    return model.cluster_centers_


def build_codebook(experiment: int, force: bool = False) -> Path:
    """Build and save the BoW codebook for one experiment."""
    config = load_experiment_config(experiment)
    if config.method != "bow":
        raise ValueError(
            f"実験設定の method は現在 bow のみ対応しています: "
            f"{config.method!r}"
        )
    experiment_dir = experiment_directory(experiment)
    training_dir = experiment_dir / "samples" / "training"
    output_path = experiment_dir / "models" / "codebook.npy"

    if output_path.exists() and not force:
        raise FileExistsError(
            f"コードブックが既に存在します: {output_path} "
            "(上書きするには --force を指定してください)"
        )

    sample_paths = find_training_samples(training_dir)
    expected_dimension = FRAME_FEATURE_COUNT * config.window_length
    descriptors = load_training_descriptors(
        sample_paths,
        expected_dimension,
    )
    codebook = fit_codebook(
        descriptors,
        config.codeword_count,
        config.random_state,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, codebook)
    print(
        f"{len(sample_paths)}ファイル、{len(descriptors)}時間窓から"
        f"D={config.codeword_count}のコードブックを作成しました: "
        f"{output_path}"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="実験のtrainingサンプルからBoWコードブックを作成します。"
    )
    parser.add_argument(
        "experiment",
        type=int,
        help="実験番号（例: 1 -> experiments/exp01）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存のコードブックを上書きする",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        build_codebook(args.experiment, force=args.force)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"エラー: {error}") from error


if __name__ == "__main__":
    main()
