"""Capture a prompt input and its corresponding webcam video."""

import argparse
from pathlib import Path

from prompt_input import load_prompts, read_answer, select_prompt
from video_recorder import CameraRecorder, PROJECT_ROOT
from experiment_utils import validate_index


SAVE_DIRECTORIES = (
    "training",
    "validation/enrollment",
    "validation/probe",
    "test/probe",
)


def video_output_path(
    subject_index: int,
    prompt_index: int,
    save_dir: str,
) -> Path:
    if save_dir not in SAVE_DIRECTORIES:
        raise ValueError(f"無効な保存先です: {save_dir}")

    filename = f"s{subject_index:02d}_p{prompt_index:02d}.mp4"
    return PROJECT_ROOT / "data" / save_dir / filename


def ensure_video_output_available(
    video_path: Path,
    force: bool = False,
) -> None:
    if video_path.exists() and not force:
        raise FileExistsError(
            f"警告: 保存予定の動画が既に存在します: {video_path} "
 "(上書きするには --force を指定してください)"
        )
    if video_path.exists() and force:
        print(f"警告: 既存の動画を上書きします: {video_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="プロンプト入力とWebカメラ映像を同時に収集します。",
    )
    parser.add_argument("subject_index", type=int, help="被験者のインデックス")
    parser.add_argument("prompt_index", type=int, help="プロンプトのインデックス")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="使用するカメラ番号（既定値: 0）",
    )
    parser.add_argument(
        "--save-dir",
        choices=SAVE_DIRECTORIES,
        default="training",
        help="動画の保存先（既定値: training）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="同名の既存動画を上書きする",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        validate_index("被験者番号", args.subject_index)
        validate_index("プロンプト番号", args.prompt_index)
        prompt = select_prompt(load_prompts(), args.prompt_index)
        video_path = video_output_path(
            args.subject_index,
            args.prompt_index,
            args.save_dir,
        )
        ensure_video_output_available(video_path, force=args.force)
    except (FileExistsError, ValueError) as error:
        raise SystemExit(f"エラー: {error}") from error

    recorder = CameraRecorder(
        output_path=video_path,
        camera_index=args.camera_index,
        show_window=False,
        quiet=True,
    )
    recorder.start()

    if not recorder.wait_until_ready():
        recorder.stop()
        recorder.join()
        raise recorder.error or RuntimeError(
            "カメラを起動できなかったため、入力を開始できません。"
        )

    try:
        answer = read_answer(prompt)
    finally:
        recorder.stop()
        recorder.join()

    if answer is None:
        print("Escキーで入力を終了しました。")
        return

    print(
        "入力が一致しました。"
        if answer == prompt
        else "入力が一致しませんでした。"
    )


if __name__ == "__main__":
    main()
