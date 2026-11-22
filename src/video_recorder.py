"""Webカメラ映像の録画処理。"""

import argparse
import threading
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data"


class CameraRecorder:
    """Run a webcam recording in a background thread."""

    def __init__(
        self,
        output_path: Path,
        camera_index: int = 0,
        show_window: bool = True,
        quiet: bool = False,
    ) -> None:
        self.output_path = Path(output_path)
        self.camera_index = camera_index
        self.show_window = show_window
        self.quiet = quiet
        self.stop_event = threading.Event()
        self.camera_ready_event = threading.Event()
        self.camera_error_event = threading.Event()
        self.error: RuntimeError | None = None
        self.thread = threading.Thread(
            target=self._record,
            daemon=True,
        )

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def join(self) -> None:
        self.thread.join()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        if timeout is None:
            while not self.camera_ready_event.is_set() and not self.camera_error_event.is_set():
                self.camera_ready_event.wait(0.05)
        else:
            end_time = time.monotonic() + timeout
            while not self.camera_ready_event.is_set() and not self.camera_error_event.is_set():
                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    break
                self.camera_ready_event.wait(min(0.05, remaining))
        return self.camera_ready_event.is_set()

    def __enter__(self) -> "CameraRecorder":
        self.start()
        if not self.wait_until_ready():
            self.stop()
            self.join()
            raise self.error or RuntimeError(
                f"カメラを起動できませんでした: index={self.camera_index}"
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
        self.join()

    def _record(self) -> None:
        import cv2

        capture = cv2.VideoCapture(self.camera_index)
        writer = None

        if not capture.isOpened():
            self.error = RuntimeError(
                f"カメラを開けませんでした: index={self.camera_index}"
            )
            self.camera_error_event.set()
            return

        try:
            while not self.stop_event.is_set():
                success, frame = capture.read()
                if not success:
                    self.error = RuntimeError(
                        f"カメラから映像を取得できませんでした: "
                        f"index={self.camera_index}"
                    )
                    self.camera_error_event.set()
                    if not self.quiet:
                        print("カメラから映像を取得できませんでした。")
                    break

                if writer is None:
                    height, width = frame.shape[:2]
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 30.0

                    writer = cv2.VideoWriter(
                        str(self.output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        self.error = RuntimeError(
                            f"動画ファイルを開けませんでした: {self.output_path}"
                        )
                        self.camera_error_event.set()
                        break
                    self.camera_ready_event.set()
                    if not self.quiet:
                        print(f"カメラ準備完了: {self.output_path}")
                    if not self.quiet:
                        print(f"録画開始: {self.output_path}")

                writer.write(frame)

                if self.show_window:
                    cv2.imshow("Webcam Recorder", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        self.stop_event.set()
        finally:
            if writer is not None:
                writer.release()
            capture.release()
            if self.show_window:
                import cv2
                cv2.destroyAllWindows()

            if writer is not None and not self.quiet:
                print(f"録画保存完了: {self.output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Webカメラ映像を録画します。"
    )
    parser.add_argument(
        "filename",
        help="保存する動画のファイル名（例: session01.mp4）"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="使用するカメラ番号（既定値: 0）"
    )
    return parser.parse_args()


def record_video(
    output_path: Path,
    camera_index: int = 0,
    stop_event: threading.Event | None = None,
    recording_event: threading.Event | None = None,
    camera_ready_event: threading.Event | None = None,
    camera_error_event: threading.Event | None = None,
    show_window: bool = True,
    quiet: bool = False,
):
    recorder = CameraRecorder(
        output_path=output_path,
        camera_index=camera_index,
        show_window=show_window,
        quiet=quiet,
    )

    # Preserve the existing event-based API for callers outside this module.
    if camera_ready_event is not None:
        recorder.camera_ready_event = camera_ready_event
    if camera_error_event is not None:
        recorder.camera_error_event = camera_error_event

    recorder.start()
    if not recorder.wait_until_ready():
        recorder.stop()
        recorder.join()
        raise recorder.error or RuntimeError(
            f"カメラを起動できませんでした: index={camera_index}"
        )

    if stop_event is not None:
        while not stop_event.is_set() and recorder.thread.is_alive():
            recorder.thread.join(0.05)
        recorder.stop()
    else:
        recorder.join()


def main():
    args = parse_args()
    filename = Path(args.filename).name
    if not Path(filename).suffix:
        filename += ".mp4"

    output_path = VIDEO_DIR / filename
    record_video(output_path, args.camera_index)


if __name__ == "__main__":
    main()
