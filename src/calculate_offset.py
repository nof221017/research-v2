"""Calculate codeword offset spaces for validation enrollment/probe pairs."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from encode_histogram import (
    encode_codewords,
    find_input_samples,
    load_codebook,
    load_sample,
)
from experiment_utils import (
    FRAME_FEATURE_COUNT,
    experiment_directory,
    load_experiment_config,
)


def frame_indices_path(
    sample_path: Path,
    samples_directory: Path,
    indices_directory: Path,
) -> Path:
    """Map a sample path to its corresponding frame-index array."""
    return indices_directory / sample_path.relative_to(samples_directory)


def load_frame_indices(
    path: Path,
    expected_count: int,
) -> np.ndarray:
    """Load and validate the original frame indices for one sample."""
    if not path.is_file():
        raise FileNotFoundError(
            f"フレーム番号が見つかりません: {path} "
            "(extract_sample.pyで再抽出してください)"
        )

    try:
        frame_indices = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"フレーム番号を読み込めません: {path}") from error

    if frame_indices.ndim != 1:
        raise ValueError(
            f"{path} は1次元配列である必要があります: "
            f"shape={frame_indices.shape}"
        )
    if frame_indices.size != expected_count:
        raise ValueError(
            f"{path} のフレーム番号数がサンプル数と一致しません: "
            f"{frame_indices.size} (期待値: {expected_count})"
        )
    if not np.issubdtype(frame_indices.dtype, np.integer):
        raise ValueError(f"{path} のフレーム番号は整数である必要があります")
    if np.any(frame_indices < 0):
        raise ValueError(f"{path} に負のフレーム番号があります")
    if np.any(np.diff(frame_indices) <= 0):
        raise ValueError(f"{path} のフレーム番号は昇順である必要があります")
    return frame_indices


def load_encoded_sample(
    sample_path: Path,
    samples_directory: Path,
    indices_directory: Path,
    expected_dimension: int,
    codebook: np.ndarray,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Load one sample and prepare its codewords and frame indices."""
    sample = load_sample(sample_path, expected_dimension)
    frame_indices = load_frame_indices(
        frame_indices_path(
            sample_path,
            samples_directory,
            indices_directory,
        ),
        len(sample),
    )
    return sample_path, encode_codewords(sample, codebook), frame_indices


def calculate_offset_space(
    codeword_ids: np.ndarray,
    frame_indices: np.ndarray,
    other_codeword_ids: np.ndarray,
    other_frame_indices: np.ndarray,
) -> dict[int, int]:
    """Count all frame offsets for pairs assigned to the same codeword."""
    arrays = (
        codeword_ids,
        frame_indices,
        other_codeword_ids,
        other_frame_indices,
    )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("コードワードIDとフレーム番号は1次元配列が必要です")
    if len(codeword_ids) != len(frame_indices) or len(
        other_codeword_ids
    ) != len(other_frame_indices):
        raise ValueError("コードワードID数とフレーム番号数が一致しません")

    positions_by_codeword: dict[int, list[int]] = {}
    for codeword_id, frame_index in zip(
        other_codeword_ids,
        other_frame_indices,
    ):
        positions_by_codeword.setdefault(int(codeword_id), []).append(
            int(frame_index)
        )

    counts = Counter(
        int(frame_index) - other_frame_index
        for codeword_id, frame_index in zip(codeword_ids, frame_indices)
        for other_frame_index in positions_by_codeword.get(
            int(codeword_id),
            (),
        )
    )
    return dict(sorted(counts.items()))


def calculate_validation_offsets(
    experiment: int,
    force: bool = False,
) -> list[Path]:
    """Write one offset-space CSV for every validation enrollment/probe pair."""
    config = load_experiment_config(experiment)
    experiment_dir = experiment_directory(experiment)
    samples_directory = experiment_dir / "samples"
    indices_directory = experiment_dir / "sample_indices"
    output_directory = experiment_dir / "offset" / "validation"
    expected_dimension = FRAME_FEATURE_COUNT * config.window_length
    codebook = load_codebook(
        experiment_dir / "models" / "codebook.npy",
        config.codeword_count,
        expected_dimension,
    )

    enrollment_samples = find_input_samples(
        samples_directory,
        "validation",
        "enrollment",
    )
    probe_samples = find_input_samples(
        samples_directory,
        "validation",
        "probe",
    )
    if not enrollment_samples:
        raise FileNotFoundError("validation/enrollmentの入力サンプルがありません")
    if not probe_samples:
        raise FileNotFoundError("validation/probeの入力サンプルがありません")

    output_paths = []
    encoded_samples: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    for enrollment_path in enrollment_samples:
        for probe_path in probe_samples:
            output_path = output_directory / (
                f"{enrollment_path.stem}__{probe_path.stem}.csv"
            )
            if output_path.exists() and not force:
                print(f"スキップ（既存）: {output_path}")
                continue

            for sample_path in (enrollment_path, probe_path):
                if sample_path not in encoded_samples:
                    _, codeword_ids, frame_indices = load_encoded_sample(
                        sample_path,
                        samples_directory,
                        indices_directory,
                        expected_dimension,
                        codebook,
                    )
                    encoded_samples[sample_path] = (
                        codeword_ids,
                        frame_indices,
                    )
            enrollment_codewords, enrollment_frame_indices = (
                encoded_samples[enrollment_path]
            )
            probe_codewords, probe_frame_indices = encoded_samples[probe_path]
            offset_space = calculate_offset_space(
                enrollment_codewords,
                enrollment_frame_indices,
                probe_codewords,
                probe_frame_indices,
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as output_file:
                writer = csv.writer(output_file)
                writer.writerow(("offset", "count"))
                writer.writerows(offset_space.items())
            output_paths.append(output_path)
            print(f"{enrollment_path} × {probe_path} -> {output_path}")

    print(f"{len(output_paths)}ファイルを保存しました")
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "validation/enrollmentとvalidation/probeの"
            "オフセット空間をCSVへ保存します。"
        )
    )
    parser.add_argument(
        "experiment",
        type=int,
        help="実験番号（例: 1 -> experiments/exp01）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存のオフセットCSVを上書きする",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        calculate_validation_offsets(
            experiment=args.experiment,
            force=args.force,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"エラー: {error}") from error


if __name__ == "__main__":
    main()
