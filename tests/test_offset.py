import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from calculate_offset import (
    calculate_offset_space,
    calculate_validation_offsets,
    load_frame_indices,
)
from experiment_utils import ExperimentConfig

with patch.dict(
    sys.modules,
    {"cv2": MagicMock(), "mediapipe": MagicMock()},
):
    from extract_sample import create_temporal_descriptors


class OffsetTests(unittest.TestCase):
    def test_temporal_descriptors_keep_start_frames_after_invalid_windows(self):
        frame_features = np.ones((8, 84), dtype=float)
        frame_features[0, 0] = np.nan

        descriptors, frame_indices = create_temporal_descriptors(
            frame_features,
            window_length=4,
        )

        np.testing.assert_array_equal(frame_indices, [2, 4])
        self.assertEqual(descriptors.shape, (2, 336))

    def test_offset_space_counts_every_matching_codeword_pair(self):
        offsets = calculate_offset_space(
            np.array([4, 4]),
            np.array([0, 2]),
            np.array([4, 4]),
            np.array([1, 3]),
        )

        self.assertEqual(offsets, {-3: 1, -1: 2, 1: 1})

    def test_frame_indices_must_match_sample_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "indices.npy"
            np.save(path, np.array([0, 2], dtype=np.int64))

            with self.assertRaises(ValueError):
                load_frame_indices(path, expected_count=1)

    def test_offset_space_rejects_mismatched_sequence_lengths(self):
        with self.assertRaises(ValueError):
            calculate_offset_space(
                np.array([4]),
                np.array([0, 2]),
                np.array([4]),
                np.array([1]),
            )

    def test_validation_enrollment_probe_pairs_are_written_as_csv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment_directory = Path(temporary_directory)
            samples_directory = experiment_directory / "samples"
            indices_directory = experiment_directory / "sample_indices"
            enrollment_path = (
                samples_directory / "validation" / "enrollment" / "s01_p01.npy"
            )
            probe_path = (
                samples_directory / "validation" / "probe" / "s02_p02.npy"
            )
            enrollment_indices_path = (
                indices_directory / "validation" / "enrollment" / "s01_p01.npy"
            )
            probe_indices_path = (
                indices_directory / "validation" / "probe" / "s02_p02.npy"
            )
            for path in (
                enrollment_path,
                probe_path,
                enrollment_indices_path,
                probe_indices_path,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)

            dimension = 84 * 4
            (experiment_directory / "models").mkdir()
            np.save(
                experiment_directory / "models" / "codebook.npy",
                np.stack((np.zeros(dimension), np.ones(dimension))),
            )
            np.save(enrollment_path, np.zeros((2, dimension)))
            np.save(probe_path, np.zeros((2, dimension)))
            np.save(enrollment_indices_path, np.array([0, 2]))
            np.save(probe_indices_path, np.array([1, 3]))

            config = ExperimentConfig(
                method="bow",
                window_length=4,
                codeword_count=2,
                random_state=0,
            )
            with (
                patch(
                    "calculate_offset.experiment_directory",
                    return_value=experiment_directory,
                ),
                patch("calculate_offset.load_experiment_config", return_value=config),
            ):
                output_paths = calculate_validation_offsets(1)

            self.assertEqual(len(output_paths), 1)
            self.assertEqual(
                output_paths[0].name,
                "s01_p01__s02_p02.csv",
            )
            with output_paths[0].open(
                encoding="utf-8",
                newline="",
            ) as output_file:
                rows = list(csv.reader(output_file))
            self.assertEqual(
                rows,
                [["offset", "count"], ["-3", "1"], ["-1", "2"], ["1", "1"]],
            )


if __name__ == "__main__":
    unittest.main()
