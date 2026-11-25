import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prompt_capture import (
    SAVE_DIRECTORIES,
    ensure_video_output_available,
    parse_args,
    video_output_path,
)
from prompt_input import handle_enter, read_answer, select_prompt


class CaptureRefactorTests(unittest.TestCase):
    def test_collected_is_the_default_save_directory(self):
        with patch.object(
            sys,
            "argv",
            ["prompt_capture.py", "3", "7"],
        ):
            args = parse_args()

        self.assertEqual(args.save_dir, "collected")

    def test_collected_video_output_path_is_used(self):
        path = video_output_path(3, 7, "collected")

        self.assertEqual(
            path.parent,
            PROJECT_ROOT / "data" / "collected",
        )

    def test_video_output_path_uses_experiment_filename(self):
        path = video_output_path(3, 7, "validation/enrollment")

        self.assertEqual(path.name, "s03_p07.mp4")
        self.assertEqual(
            path.parent,
            PROJECT_ROOT / "data" / "validation" / "enrollment",
        )

    def test_invalid_save_directory_is_rejected(self):
        with self.assertRaises(ValueError):
            video_output_path(3, 7, "tmp")

    def test_save_directories_remain_explicit(self):
        self.assertEqual(
            SAVE_DIRECTORIES,
            (
                "collected",
                "training",
                "validation/enrollment",
                "validation/probe",
                "test/probe",
            ),
        )

    def test_select_prompt_uses_one_based_index(self):
        self.assertEqual(select_prompt(["first", "second"], 2), "second")

    def test_select_prompt_rejects_out_of_range_index(self):
        with self.assertRaises(ValueError):
            select_prompt(["only"], 2)

    def test_enter_inserts_prompt_newline(self):
        answer, cursor, action = handle_enter(
            "first\nsecond",
            "first",
            5,
        )

        self.assertEqual(answer, "first\n")
        self.assertEqual(cursor, 6)
        self.assertEqual(action, "insert_newline")

    def test_enter_at_middle_of_unfinished_prompt_is_ignored(self):
        answer, cursor, action = handle_enter(
            "abcdef",
            "ab",
            2,
        )

        self.assertEqual(answer, "ab")
        self.assertEqual(cursor, 2)
        self.assertEqual(action, "ignored")

    def test_enter_at_mismatch_end_finishes_input(self):
        answer, cursor, action = handle_enter(
            "abc",
            "axc",
            3,
        )

        self.assertEqual(answer, "axc")
        self.assertEqual(cursor, 3)
        self.assertEqual(action, "finish")

    def test_escape_returns_without_completing_input(self):
        with (
            patch("prompt_input.msvcrt.getwch", return_value="\x1b"),
            patch("prompt_input.render_screen"),
        ):
            answer = read_answer("abc")

        self.assertIsNone(answer)

    def test_existing_video_requires_force(self):
        existing_path = PROJECT_ROOT / "data" / "training" / "existing.mp4"
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.touch()
        try:
            with self.assertRaises(FileExistsError):
                ensure_video_output_available(existing_path)
        finally:
            existing_path.unlink()

    def test_existing_video_is_allowed_with_force(self):
        existing_path = PROJECT_ROOT / "data" / "training" / "existing.mp4"
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.touch()
        try:
            with patch("builtins.print") as print_mock:
                ensure_video_output_available(existing_path, force=True)
            print_mock.assert_called_once()
        finally:
            existing_path.unlink()


if __name__ == "__main__":
    unittest.main()
