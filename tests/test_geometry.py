import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from virtual_pointer import (
    ArucoBoardPoseEstimator,
    MockStereoScene,
    _undistort_landmarks,
    calculate_intersection,
    calculate_pointing_direction,
    mock_stereo_calibration,
    to_screen_pixels,
    triangulate_landmarks,
    write_board_marker,
)


class GeometryTests(unittest.TestCase):
    def test_mock_stereo_pipeline_hits_expected_board_location(self):
        calibration = mock_stereo_calibration()
        scene = MockStereoScene(calibration)
        for _ in range(6):
            _, _, raw1, raw2 = scene.next_frames()
            points = triangulate_landmarks(
                _undistort_landmarks(raw1, calibration.K1, calibration.D1),
                _undistort_landmarks(raw2, calibration.K2, calibration.D2),
                calibration.P1, calibration.P2,
            )
            tip, direction = calculate_pointing_direction(points)
            hit = calculate_intersection(tip, direction, scene.board_geometry[3])
            self.assertIsNotNone(hit)
            actual = to_screen_pixels(hit, *scene.board_geometry[:3], (600, 400), (1280, 720))
            phase = scene.frame_count * 0.035
            expected = (
                round((np.sin(phase) * 300 * 0.82 + 300) / 600 * 1279),
                round((np.cos(phase * 0.73) * 200 * 0.78 + 200) / 400 * 719),
            )
            self.assertLessEqual(abs(actual[0] - expected[0]), 2)
            self.assertLessEqual(abs(actual[1] - expected[1]), 2)

    def test_ray_parallel_or_behind_board_has_no_hit(self):
        plane = (0.0, 0.0, 1.0, -1000.0)
        self.assertIsNone(calculate_intersection(np.array([0, 0, 0]), np.array([1, 0, 0]), plane))
        self.assertIsNone(calculate_intersection(np.array([0, 0, 0]), np.array([0, 0, -1]), plane))

    def test_generated_aruco_board_is_detectable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "board.png"
            write_board_marker(path, (600, 400), 80)
            frame = cv2.imread(str(path))
            estimator = ArucoBoardPoseEstimator((600, 400), 80)
            K = np.array([[1800, 0, 900], [0, 1800, 600], [0, 0, 1]], dtype=np.float64)
            geometry = estimator.estimate(frame, K, np.zeros(5))
            self.assertIsNotNone(geometry)
            origin, u_axis, v_axis, plane = geometry
            self.assertTrue(np.isfinite(origin).all())
            self.assertAlmostEqual(np.linalg.norm(u_axis), 1, places=5)
            self.assertAlmostEqual(np.linalg.norm(v_axis), 1, places=5)
            self.assertAlmostEqual(np.dot(np.array(plane[:3]), origin) + plane[3], 0, places=5)


if __name__ == "__main__":
    unittest.main()
