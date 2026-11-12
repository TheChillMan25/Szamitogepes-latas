import argparse
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stereo_calibrate import calibrate


class CalibrationTests(unittest.TestCase):
    def test_calibrates_synthetic_checkerboard_pairs(self):
        # Render a physical 10 x 7 square board through two known camera poses.
        square_pixels = 30
        pattern = np.full((7 * square_pixels, 10 * square_pixels), 255, np.uint8)
        for row in range(7):
            for col in range(10):
                if (row + col) % 2 == 0:
                    pattern[row * square_pixels:(row + 1) * square_pixels,
                            col * square_pixels:(col + 1) * square_pixels] = 0
        source = np.float32([[0, 0], [300, 0], [300, 210], [0, 210]])
        world = np.float32([[0, 0, 0], [250, 0, 0], [250, 175, 0], [0, 175, 0]])
        K = np.array([[1100, 0, 640], [0, 1100, 360], [0, 0, 1]], np.float64)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for side in ("left", "right"):
                (root / side).mkdir()
            for index in range(16):
                rvec = np.array([0.18 * np.sin(index * 0.7),
                                 0.25 * np.cos(index * 0.6),
                                 0.12 * np.sin(index * 0.9)], np.float64)
                translation = np.array([[-125 + 38 * np.sin(index * 1.1)],
                                        [-87 + 30 * np.cos(index * 0.8)],
                                        [850 + 100 * np.sin(index * 0.5)]], np.float64)
                for side, offset in (("left", 0), ("right", -120)):
                    tvec = translation + np.array([[offset], [0], [0]])
                    projected, _ = cv2.projectPoints(world, rvec, tvec, K, None)
                    target = projected.reshape(4, 2).astype(np.float32)
                    transform = cv2.getPerspectiveTransform(source, target)
                    frame = cv2.warpPerspective(pattern, transform, (1280, 720),
                                                borderValue=255)
                    cv2.imwrite(str(root / side / f"{index:04d}.png"), frame)
            output = root / "calibration.json"
            calibrate(argparse.Namespace(images=root, pattern=(9, 6), square_mm=25.0,
                                         output=output))
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["calibration_type"], "measured")
            self.assertEqual(data["image_size"], [1280, 720])
            self.assertGreaterEqual(data["usable_pairs"], 10)
            self.assertAlmostEqual(np.linalg.norm(data["T"]), 120, delta=35)


if __name__ == "__main__":
    unittest.main()
