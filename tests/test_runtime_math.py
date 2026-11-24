import threading
import time
import unittest

import numpy as np

from camera_manager import SingleThreadStereoCapture
from projector_calibration import compute_projector_geometry
from virtual_pointer import KalmanFilter2D, to_screen_pixels


class StereoSnapshotTests(unittest.TestCase):
    def make_capture(self, left_timestamp: float, right_timestamp: float):
        capture = object.__new__(SingleThreadStereoCapture)
        capture._state_lock = threading.Lock()
        capture._latest_left = np.zeros((1, 1, 3), dtype=np.uint8)
        capture._latest_right = np.zeros((1, 1, 3), dtype=np.uint8)
        capture._left_timestamp = left_timestamp
        capture._right_timestamp = right_timestamp
        capture._left_online = True
        capture._right_online = True
        capture._fps_left = 30.0
        capture._fps_right = 30.0
        capture._left_idx = 0
        capture._right_idx = 1
        return capture

    def test_synchronized_snapshot_rejects_misaligned_frames(self):
        now = time.perf_counter()
        capture = self.make_capture(now - 0.01, now - 0.10)

        snapshot = capture.get_synchronized_snapshot(max_skew_sec=0.02)

        self.assertIsNone(snapshot["left_frame"])
        self.assertIsNone(snapshot["right_frame"])

    def test_synchronized_snapshot_accepts_recent_aligned_frames(self):
        now = time.perf_counter()
        capture = self.make_capture(now - 0.01, now - 0.015)

        snapshot = capture.get_synchronized_snapshot(max_skew_sec=0.02)

        self.assertIsNotNone(snapshot["left_frame"])
        self.assertIsNotNone(snapshot["right_frame"])


class PointerMathTests(unittest.TestCase):
    def test_kalman_reset_discards_previous_velocity(self):
        kalman = KalmanFilter2D()
        for point in ((0, 0), (100, 100), (200, 200)):
            kalman.update(*point)

        kalman.reset()

        self.assertEqual(kalman.update(500, 500), (500, 500))
        state = kalman.filter.statePost.reshape(-1)
        self.assertEqual(float(state[2]), 0.0)
        self.assertEqual(float(state[3]), 0.0)

    def test_homography_maps_plane_origin_and_scales_to_target_screen(self):
        homography = np.array(
            [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        origin = np.zeros(3, dtype=np.float64)
        u_axis = np.array([1.0, 0.0, 0.0])
        v_axis = np.array([0.0, 1.0, 0.0])

        self.assertEqual(
            to_screen_pixels(
                origin,
                origin,
                u_axis,
                v_axis,
                (1.0, 1.0),
                (201, 101),
                homography,
                (101, 101),
            ),
            (0, 0),
        )
        self.assertEqual(
            to_screen_pixels(
                u_axis + v_axis,
                origin,
                u_axis,
                v_axis,
                (1.0, 1.0),
                (201, 101),
                homography,
                (101, 101),
            ),
            (200, 100),
        )

    def test_projector_origin_extrapolates_to_image_edge_and_v_points_down(self):
        width, height = 1280, 720
        marker_size, margin = 180, 90
        x0 = margin + marker_size / 2
        x1 = width - margin - marker_size / 2
        y0 = margin + marker_size / 2
        y1 = height - margin - marker_size / 2
        marker_points = {
            0: np.array([x0, y0, 5.0]),
            1: np.array([x1, y0, 5.0]),
            2: np.array([x1, y1, 5.0]),
            3: np.array([x0, y1, 5.0]),
        }

        geometry = compute_projector_geometry(
            marker_points, width, height, marker_size, margin
        )

        np.testing.assert_allclose(geometry["P_origin"], [0.0, 0.0, 5.0], atol=1e-5)
        self.assertGreater(float(geometry["v_axis"][1]), 0.99)
        np.testing.assert_allclose(
            geometry["projector_corners_3d"],
            [[0, 0, 5], [width - 1, 0, 5], [width - 1, height - 1, 5], [0, height - 1, 5]],
            atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
