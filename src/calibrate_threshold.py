"""Calibrate per-subject thresholds from validation histograms."""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from calculate_similarity import calculate_similarity_from_paths
from experiment_utils import experiment_directory


HISTOGRAM_FILENAME = re.compile(r"^s(?P<subject>\d+)_p(?P<prompt>\d+)\.npy$")


def histogram_indices(directory: Path) -> list[tuple[int, int, Path]]:
    """Return subject/prompt indices and paths found in a directory."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"ヒストグラムディレクトリが見つかりません: {directory}"
        )

    indices = []
    for path in sorted(directory.glob("*.npy")):
        match = HISTOGRAM_FILENAME.match(path.name)
        if match:
            indices.append(
                (
                    int(match.group("subject")),
                    int(match.group("prompt")),
                    path,
                )
            )
    return indices


def subject_indices(directory: Path) -> list[int]:
    """Return unique subject indices found in a directory."""
    return sorted({subject for subject, _, _ in histogram_indices(directory)})


def all_subject_indices(experiment: int) -> list[int]:
    """Return subjects available in validation/enrollment."""
    experiment_dir = experiment_directory(experiment)
    enrollment_dir = experiment_dir / "histograms" / "validation" / "enrollment"
    subjects = subject_indices(enrollment_dir)
    if not subjects:
        raise FileNotFoundError(
            f"登録用ヒストグラムが見つかりません: {enrollment_dir}"
        )
    return subjects


def calibrate_threshold(
    experiment: int,
    subject: int,
) -> dict:
    """Calibrate and return threshold metadata for one subject."""
    if isinstance(subject, bool) or not isinstance(subject, int):
        raise ValueError("被験者番号は整数で指定してください")
    if subject < 1:
        raise ValueError("被験者番号は1以上で指定してください")

    experiment_dir = experiment_directory(experiment)
    enrollment_dir = (
        experiment_dir / "histograms" / "validation" / "enrollment"
    )
    probe_dir = (
        experiment_dir / "histograms" / "validation" / "probe"
    )

    enrollment_histograms = histogram_indices(enrollment_dir)
    probe_histograms = histogram_indices(probe_dir)

    target_enrollment = [
        (prompt, path)
        for current_subject, prompt, path in enrollment_histograms
        if current_subject == subject
    ]
    if not target_enrollment:
        raise FileNotFoundError(
            f"被験者{subject}の登録用ヒストグラムがありません"
        )

    target_probe = [
        (prompt, path)
        for current_subject, prompt, path in probe_histograms
        if current_subject == subject
    ]
    if not target_probe:
        raise FileNotFoundError(
            f"被験者{subject}の本人probeヒストグラムがありません"
        )

    genuine_scores = []
    for enrollment_prompt, enrollment_path in target_enrollment:
        for probe_prompt, probe_path in target_probe:
            score = calculate_similarity_from_paths(
                experiment,
                enrollment_path,
                probe_path,
            )
            genuine_scores.append(
                {
                    "score": score,
                    "enrollment": str(enrollment_path),
                    "probe": str(probe_path),
                }
            )

    impostor_scores = []
    for enrollment_prompt, enrollment_path in target_enrollment:
        for impostor_subject, probe_prompt, probe_path in probe_histograms:
            if impostor_subject == subject:
                continue
            score = calculate_similarity_from_paths(
                experiment,
                enrollment_path,
                probe_path,
            )
            impostor_scores.append(
                {
                    "score": score,
                    "subject": impostor_subject,
                    "enrollment_prompt": enrollment_prompt,
                    "probe_prompt": probe_prompt,
                    "enrollment": str(enrollment_path),
                    "probe": str(probe_path),
                }
            )

    if not impostor_scores:
        raise ValueError(
            f"被験者{subject}を除く他人probeヒストグラムがありません"
        )

    genuine_min_entry = min(genuine_scores, key=lambda item: item["score"])
    impostor_max_entry = max(impostor_scores, key=lambda item: item["score"])
    genuine_min = float(genuine_min_entry["score"])
    impostor_max = float(impostor_max_entry["score"])

    if genuine_min >= impostor_max:
        threshold = (genuine_min + impostor_max) / 2.0
    else:
        threshold = genuine_min

    return {
        "experiment": experiment,
        "subject": subject,
        "threshold": float(threshold),
        "threshold_rule": (
            "midpoint"
            if genuine_min >= impostor_max
            else "genuine_min"
        ),
        "genuine_min": genuine_min,
        "impostor_max": impostor_max,
        "genuine_count": len(genuine_scores),
        "impostor_count": len(impostor_scores),
        "genuine_min_pair": genuine_min_entry,
        "impostor_max_pair": impostor_max_entry,
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
    }


def save_threshold(
    experiment: int,
    metadata: dict,
    force: bool = False,
) -> tuple[Path, Path]:
    """Save a threshold .npy and JSON metadata."""
    subject = metadata["subject"]
    threshold_dir = experiment_directory(experiment) / "threshold"
    threshold_path = threshold_dir / f"s{subject:02d}_threshold.npy"
    metadata_path = threshold_dir / f"s{subject:02d}_threshold.json"

    if (
        (threshold_path.exists() or metadata_path.exists())
        and not force
    ):
        raise FileExistsError(
            f"閾値またはメタデータが既に存在します: {threshold_path} / "
            f"{metadata_path}（上書きには --force を指定してください）"
        )

    threshold_dir.mkdir(parents=True, exist_ok=True)
    np.save(threshold_path, np.array(metadata["threshold"], dtype=float))
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
    return threshold_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="被験者ごとの本人・他人類似度から閾値を決定します。",
    )
    parser.add_argument(
        "experiment",
        type=int,
        help="実験番号（例: 1 -> exp01）",
    )
    parser.add_argument(
        "subject",
        type=int,
        nargs="?",
        default=None,
        help="対象被験者番号（省略時は全被験者）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存の閾値とメタデータを上書きする",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        subjects = (
            [args.subject]
            if args.subject is not None
            else all_subject_indices(args.experiment)
        )
        for subject in subjects:
            metadata = calibrate_threshold(args.experiment, subject)
            threshold_path, metadata_path = save_threshold(
                args.experiment,
                metadata,
                force=args.force,
            )
            print(
                f"subject{subject:02d}: threshold={metadata['threshold']:.12g} "
                f"genuine_min={metadata['genuine_min']:.12g} "
                f"impostor_max={metadata['impostor_max']:.12g}"
            )
            print(f"{threshold_path}")
            print(f"{metadata_path}")
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(f"エラー: {error}") from error


if __name__ == "__main__":
    main()