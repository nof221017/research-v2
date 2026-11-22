"""Extract temporal landmark samples from recorded videos."""

import argparse
import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from experiment_utils import (
    FRAME_FEATURE_COUNT,
    experiment_directory,
    load_experiment_config,
)


LANDMARK_COUNT = 21
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".wmv"}
VIDEO_NAME_PATTERN = re.compile(r"^s(?P<subject>\d+)_p(?P<prompt>\d+)$")


def load_window_length(experiment: int) -> int:
    """Load and return L from an experiment configuration."""
    return load_experiment_config(experiment).window_length


def create_temporal_descriptors(
    frame_features: np.ndarray,
    window_length: int,
) -> np.ndarray:
    """Concatenate L frames with 50% overlap."""
    if window_length < 2 or window_length % 2:
        raise ValueError("Lは2以上の偶数で指定してください")
    if frame_features.ndim != 2 or frame_features.shape[1] != FRAME_FEATURE_COUNT:
        raise ValueError(
            f"フレーム特徴量は{FRAME_FEATURE_COUNT}次元の2次元配列が必要です"
        )

    if len(frame_features) < window_length:
        return np.empty((0, FRAME_FEATURE_COUNT * window_length), dtype=float)

    stride = window_length // 2
    descriptors = [
        frame_features[start : start + window_length].reshape(-1)
        for start in range(0, len(frame_features) - window_length + 1, stride)
    ]
    descriptors_array = np.asarray(descriptors, dtype=float)
    valid = np.isfinite(descriptors_array).all(axis=1)
    return descriptors_array[valid]


def extract_frame_features(video_path: Path) -> np.ndarray:
    """Extract 84-dimensional left/right hand x-y features per frame."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    frame_features = []
    mp_hands = mp.solutions.hands

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                frame_feature = np.full((2, LANDMARK_COUNT, 2), np.nan)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness,
                    ):
                        hand_label = handedness.classification[0].label
                        hand_index = 0 if hand_label == "Left" else 1
                        for landmark_index, landmark in enumerate(
                            hand_landmarks.landmark
                        ):
                            frame_feature[
                                hand_index, landmark_index
                            ] = (landmark.x, landmark.y)

                frame_features.append(frame_feature.reshape(-1))
    finally:
        capture.release()

    return np.asarray(frame_features, dtype=float).reshape(
        -1, FRAME_FEATURE_COUNT
    )


def sample_output_path(
    video_path: Path,
    data_dir: Path,
    output_dir: Path,
) -> Path:
    """Map data/<split>/sXX_pYY.mp4 to samples/<split>/sXX_pYY.npy."""
    relative_path = video_path.relative_to(data_dir)
    return (output_dir / relative_path).with_suffix(".npy")


def find_videos(
    data_dir: Path,
    subject: int | None = None,
    prompt: int | None = None,
) -> list[Path]:
    """Find videos and optionally filter by subject and prompt indices."""
    videos = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue

        match = VIDEO_NAME_PATTERN.match(path.stem)
        if match is None:
            continue
        if subject is not None and int(match["subject"]) != subject:
            continue
        if prompt is not None and int(match["prompt"]) != prompt:
            continue
        videos.append(path)
    return videos


def extract_samples(
    data_dir: Path,
    output_dir: Path,
    window_length: int = 4,
    subject: int | None = None,
    prompt: int | None = None,
    force: bool = False,
) -> list[Path]:
    """Extract and save one temporal descriptor array per input video."""
    video_paths = find_videos(data_dir, subject=subject, prompt=prompt)
    if not video_paths:
        raise FileNotFoundError("条件に一致する動画が見つかりません")

    output_paths = []
    for video_index, video_path in enumerate(video_paths, start=1):
        output_path = sample_output_path(video_path, data_dir, output_dir)
        if output_path.exists() and not force:
            print(f"スキップ（既存）: {output_path}")
            continue

        print(
            f"Processing ({video_index}/{len(video_paths)}): {video_path}",
            flush=True,
        )
        frame_features = extract_frame_features(video_path)
        descriptors = create_temporal_descriptors(
            frame_features,
            window_length,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, descriptors)
        output_paths.append(output_path)
        print(
            f"{video_path}: {len(frame_features)}フレーム, "
            f"{len(descriptors)}サンプル -> {output_path}"
        )

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="動画からMediaPipeランドマークの時間窓サンプルを作成します。"
    )
    parser.add_argument(
        "experiment",
        type=int,
        help="実験番号（例: 1 -> experiments/exp01）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="入力動画のルートディレクトリ（既定値: data）",
    )
    parser.add_argument("--subject", type=int, help="被験者インデックス")
    parser.add_argument("--prompt", type=int, help="プロンプトインデックス")
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存の入力サンプルを上書きして再処理する",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"入力ディレクトリが見つかりません: {args.data_dir}")

    try:
        window_length = load_window_length(args.experiment)
        output_dir = experiment_directory(args.experiment) / "samples"
        extract_samples(
            data_dir=args.data_dir,
            output_dir=output_dir,
            window_length=window_length,
            subject=args.subject,
            prompt=args.prompt,
            force=args.force,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"エラー: {error}") from error


if __name__ == "__main__":
    main()
