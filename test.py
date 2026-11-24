import argparse
import faulthandler
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Enable Python & native fault tracebacks where supported
faulthandler.enable(all_threads=True)

# Import helpers from virtual_pointer.py without starting the GUI
import virtual_pointer as vp


def get_rss_mb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return kb / 1024.0
    except OSError:
        pass
    return -1.0


def log(msg: str) -> None:
    rss = get_rss_mb()
    rss_str = f" [RSS={rss:.1f} MB]" if rss >= 0 else ""
    print(f"[{time.strftime('%H:%M:%S')}]{rss_str} {msg}", flush=True)


def test_tracker() -> bool:
    log("Stage 1/3: Testing HandTrackerBackend in isolation...")
    model_path = Path(vp.HAND_LANDMARKER_MODEL_FILE)
    log(f"Model path: {model_path.resolve()} | exists={model_path.exists()}")
    if model_path.exists():
        log(f"Model size: {model_path.stat().st_size / (1024 * 1024):.2f} MB")

    t0 = time.perf_counter()
    tracker = vp.HandTrackerBackend()
    log(f"HandTrackerBackend created in {time.perf_counter() - t0:.2f}s")

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t1 = time.perf_counter()
    res = tracker.detect_index_landmarks(dummy_frame)
    log(f"Inference on blank 640x480 frame finished in {(time.perf_counter() - t1)*1000:.1f} ms | result={res}")

    tracker.close()
    log("PASS: HandTrackerBackend closed cleanly.")
    return True


def test_cameras(left_idx: int, right_idx: int, settings_path: Path) -> bool:
    log("Stage 2/3: Testing camera open & frame read...")
    settings = vp.load_camera_settings(settings_path)
    log(f"Camera settings: {settings}")

    for label, idx in (("LEFT", left_idx), ("RIGHT", right_idx)):
        log(f"Opening {label} camera index {idx}...")
        cap = vp._open_capture(idx)
        opened = bool(cap and cap.isOpened())
        log(f"{label} cam #{idx} isOpened={opened}")
        if not opened:
            if cap is not None:
                cap.release()
            log(f"FAIL: Could not open {label} camera #{idx}")
            return False

        vp._apply_camera_settings(cap, settings)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None or frame.size == 0:
            log(f"FAIL: {label} cam #{idx} opened, but cap.read() returned no frame.")
            return False

        log(f"PASS: {label} cam #{idx} frame shape={frame.shape}, dtype={frame.dtype}")

    return True


def test_qt(screen_idx: int | None) -> bool:
    log("Stage 3/3: Testing PySide6 QApplication & LaserOverlay...")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([sys.argv[0]])
    screens = app.screens()
    log(f"Detected {len(screens)} Qt screen(s)")
    if not screens:
        log("FAIL: No Qt screens detected.")
        return False

    idx = 1 if (screen_idx is None and len(screens) > 1) else (screen_idx or 0)
    idx = max(0, min(idx, len(screens) - 1))
    geom = screens[idx].geometry()
    log(f"Using screen #{idx} geometry={geom.x()},{geom.y()} {geom.width()}x{geom.height()}")

    overlay = vp.LaserOverlay()
    overlay.setGeometry(geom)
    overlay.set_position((geom.width() // 2, geom.height() // 2))
    overlay.show()

    QTimer.singleShot(1000, app.quit)
    app.exec()
    overlay.close()
    log("PASS: Qt overlay ran and exited cleanly.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated diagnostics for virtual_pointer.py")
    parser.add_argument(
        "--stage",
        choices=["all", "tracker", "cameras", "qt"],
        default="all",
        help="Which subsystem to test",
    )
    parser.add_argument("--left", type=int, default=0, help="Left camera index")
    parser.add_argument("--right", type=int, default=1, help="Right camera index")
    parser.add_argument("--settings", type=Path, default=vp.SETTINGS_FILE, help="Camera settings JSON")
    parser.add_argument("--screen", type=int, default=None, help="Target Qt screen index")
    args = parser.parse_args()

    log(f"PID={os.getpid()} | Python={sys.version.split()[0]}")

    if args.stage in ("all", "tracker"):
        if not test_tracker():
            return 1
        if args.stage == "all":
            return 0  # Stop after tracker in 'all' so we isolate tracker memory/crashes first

    if args.stage == "cameras":
        return 0 if test_cameras(args.left, args.right, args.settings) else 1

    if args.stage == "qt":
        return 0 if test_qt(args.screen) else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())