"""Authenticate test probe histograms using per-subject thresholds."""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calculate_similarity import (
    calculate_similarity_from_paths,
)
from experiment_utils import experiment_directory


HISTOGRAM_FILENAME = re.compile(
    r"^s(?P<subject>\d+)_p(?P<prompt>\d+)\.npy$"
)


@dataclass(frozen=True)
class ComparisonResult:
    probe_path: Path
    true_subject: int
    probe_prompt: int
    candidate_subject: int
    enrollment_path: Path
    enrollment_prompt: int
    similarity: float
    threshold: float
    decision: str


@dataclass(frozen=True)
class AuthenticationResult:
    probe_path: Path
    true_subject: int
    probe_prompt: int
    candidate_subject: int
    comparisons: tuple[ComparisonResult, ...]
    genuine_count: int
    impostor_count: int
    final_decision: str


def validate_subject(subject: int, name: str = "被験者番号") -> None:
    """Validate a subject index."""
    if isinstance(subject, bool) or not isinstance(subject, int):
        raise ValueError(f"{name}は整数で指定してください")
    if subject < 1:
        raise ValueError(f"{name}は1以上で指定してください")


def parse_histogram_filename(path: Path) -> tuple[int, int]:
    """Parse subject and prompt indices from sXX_pYY.npy."""
    match = HISTOGRAM_FILENAME.match(path.name)
    if not match:
        raise ValueError(
            f"ヒストグラム名が不正です: {path} "
            "(期待形式: sXX_pYY.npy)"
        )
    subject = int(match.group("subject"))
    prompt = int(match.group("prompt"))
    if subject < 1 or prompt < 1:
        raise ValueError(
            f"ヒストグラム名の番号は1以上で指定してください: {path}"
        )
    return subject, prompt


def find_histograms(directory: Path) -> list[tuple[int, int, Path]]:
    """Find valid histograms and return subject, prompt, and path."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"ヒストグラムディレクトリが見つかりません: {directory}"
        )

    histograms = []
    for path in sorted(directory.glob("*.npy")):
        subject, prompt = parse_histogram_filename(path)
        histograms.append((subject, prompt, path))
    if not histograms:
        raise FileNotFoundError(
            f"ヒストグラムが見つかりません: {directory}"
        )
    return histograms


def find_probe_histograms(
    experiment: int,
    probe_subject: int | None = None,
) -> list[tuple[int, int, Path]]:
    """Find test/probe histograms, optionally filtered by true subject."""
    if probe_subject is not None:
        validate_subject(probe_subject, "認証対象被験者番号")

    probe_directory = (
        experiment_directory(experiment)
        / "histograms"
        / "test"
        / "probe"
    )
    probes = find_histograms(probe_directory)
    if probe_subject is None:
        return probes

    filtered = [
        (subject, prompt, path)
        for subject, prompt, path in probes
        if subject == probe_subject
    ]
    if not filtered:
        raise FileNotFoundError(
            f"被験者{probe_subject}のtest/probeヒストグラムがありません"
        )
    return filtered


def find_candidate_subjects(
    experiment: int,
    candidate_subject: int | None = None,
) -> list[int]:
    """Find candidate subjects from validation/enrollment."""
    if candidate_subject is not None:
        validate_subject(candidate_subject, "照合対象被験者番号")
        return [candidate_subject]

    enrollment_directory = (
        experiment_directory(experiment)
        / "histograms"
        / "validation"
        / "enrollment"
    )
    enrollments = find_histograms(enrollment_directory)
    return sorted({subject for subject, _, _ in enrollments})


def find_enrollment_histograms(
    experiment: int,
    subject: int,
) -> list[tuple[int, int, Path]]:
    """Find all enrollment histograms for one subject."""
    validate_subject(subject)
    enrollment_directory = (
        experiment_directory(experiment)
        / "histograms"
        / "validation"
        / "enrollment"
    )
    enrollments = find_histograms(enrollment_directory)
    subject_histograms = [
        (current_subject, prompt, path)
        for current_subject, prompt, path in enrollments
        if current_subject == subject
    ]
    if not subject_histograms:
        raise FileNotFoundError(
            f"被験者{subject}の登録用ヒストグラムがありません"
        )
    return subject_histograms


def load_threshold(experiment: int, subject: int) -> float:
    """Load one subject threshold from threshold/sXX_threshold.npy."""
    validate_subject(subject)
    threshold_path = (
        experiment_directory(experiment)
        / "threshold"
        / f"s{subject:02d}_threshold.npy"
    )
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"被験者{subject}の閾値が見つかりません: {threshold_path}"
        )

    try:
        threshold_array = np.load(
            threshold_path,
            allow_pickle=False,
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"被験者{subject}の閾値を読み込めません: {threshold_path}"
        ) from error

    if threshold_array.size != 1:
        raise ValueError(
            f"被験者{subject}の閾値はスカラーである必要があります: "
            f"shape={threshold_array.shape}"
        )

    threshold = float(threshold_array.reshape(-1)[0])
    if not np.isfinite(threshold):
        raise ValueError(
            f"被験者{subject}の閾値が有限な数値ではありません"
        )
    return threshold


def authenticate(
    experiment: int,
    probe_subject: int | None = None,
    candidate_subject: int | None = None,
) -> list[AuthenticationResult]:
    """Authenticate probe histograms against one or all candidate subjects."""
    probes = find_probe_histograms(experiment, probe_subject)
    candidates = find_candidate_subjects(experiment, candidate_subject)

    thresholds = {
        subject: load_threshold(experiment, subject)
        for subject in candidates
    }
    enrollments = {
        subject: find_enrollment_histograms(experiment, subject)
        for subject in candidates
    }

    results = []
    for true_subject, probe_prompt, probe_path in probes:
        for candidate in candidates:
            threshold = thresholds[candidate]
            comparisons = []
            genuine_count = 0
            impostor_count = 0

            for _, enrollment_prompt, enrollment_path in enrollments[candidate]:
                similarity = calculate_similarity_from_paths(
                    experiment,
                    enrollment_path,
                    probe_path,
                )
                decision = (
                    "genuine"
                    if similarity >= threshold
                    else "impostor"
                )
                if decision == "genuine":
                    genuine_count += 1
                else:
                    impostor_count += 1

                comparisons.append(
                    ComparisonResult(
                        probe_path=probe_path,
                        true_subject=true_subject,
                        probe_prompt=probe_prompt,
                        candidate_subject=candidate,
                        enrollment_path=enrollment_path,
                        enrollment_prompt=enrollment_prompt,
                        similarity=similarity,
                        threshold=threshold,
                        decision=decision,
                    )
                )

            if genuine_count > impostor_count:
                final_decision = "genuine"
            elif impostor_count > genuine_count:
                final_decision = "impostor"
            else:
                final_decision = "pending"

            results.append(
                AuthenticationResult(
                    probe_path=probe_path,
                    true_subject=true_subject,
                    probe_prompt=probe_prompt,
                    candidate_subject=candidate,
                    comparisons=tuple(comparisons),
                    genuine_count=genuine_count,
                    impostor_count=impostor_count,
                    final_decision=final_decision,
                )
            )
    return results


def save_authentication_results(
    experiment: int,
    results: list[AuthenticationResult],
    output_path: Path | None = None,
) -> Path:
    """Save each probe/candidate/enrollment comparison to CSV."""
    if output_path is None:
        output_path = (
            experiment_directory(experiment)
            / "results"
            / "authentication.csv"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "probe_filename",
        "true_subject",
        "probe_prompt",
        "candidate_subject",
        "enrollment_filename",
        "enrollment_prompt",
        "similarity",
        "threshold",
        "comparison_decision",
        "genuine_count",
        "impostor_count",
        "final_decision",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for comparison in result.comparisons:
                writer.writerow(
                    {
                        "probe_filename": comparison.probe_path.name,
                        "true_subject": result.true_subject,
                        "probe_prompt": result.probe_prompt,
                        "candidate_subject": result.candidate_subject,
                        "enrollment_filename": comparison.enrollment_path.name,
                        "enrollment_prompt": comparison.enrollment_prompt,
                        "similarity": f"{comparison.similarity:.12g}",
                        "threshold": f"{comparison.threshold:.12g}",
                        "comparison_decision": comparison.decision,
                        "genuine_count": result.genuine_count,
                        "impostor_count": result.impostor_count,
                        "final_decision": result.final_decision,
                    }
                )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="test/probeヒストグラムを被験者閾値で認証します。",
    )
    parser.add_argument(
        "experiment",
        type=int,
        help="実験番号（例: 1 -> exp01）",
    )
    parser.add_argument(
        "--probe-subject",
        type=int,
        default=None,
        help="認証対象の被験者番号（省略時は全被験者）",
    )
    parser.add_argument(
        "--candidate-subject",
        type=int,
        default=None,
        help="照合対象の被験者番号（省略時は全被験者）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV出力先（既定値: experiments/expEE/results/authentication.csv）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        results = authenticate(
            experiment=args.experiment,
            probe_subject=args.probe_subject,
            candidate_subject=args.candidate_subject,
        )
        output_path = save_authentication_results(
            experiment=args.experiment,
            results=results,
            output_path=args.output,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
    ) as error:
        raise SystemExit(f"エラー: {error}") from error

    print(
        f"{len(results)}件の照合結果を保存しました: {output_path}"
    )


if __name__ == "__main__":
    main()