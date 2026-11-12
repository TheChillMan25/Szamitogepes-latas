import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from virtual_pointer import LaserOverlay, VirtualPointerWindow, mock_stereo_calibration


class UiTests(unittest.TestCase):
    def test_mock_updates_overlay(self):
        app = QApplication.instance() or QApplication([])
        screen = app.primaryScreen()
        overlay = LaserOverlay()
        overlay.setGeometry(screen.geometry())
        window = VirtualPointerWindow(
            mock_stereo_calibration(), "Mock", screen, overlay,
            None, (None, None), "MOCK", (600, 400), 80, None,
        )
        try:
            window.timer.stop()
            window.update_demo()
            self.assertIsNotNone(overlay.position)
            self.assertIn("px", window.status.text())
        finally:
            window.close()
            overlay.close()


if __name__ == "__main__":
    unittest.main()
