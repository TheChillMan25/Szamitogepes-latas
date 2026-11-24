import glob
import re
import sys
import threading
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

MAX_CONSECUTIVE_FAILURES_BEFORE_REOPEN = 5
RECONNECT_BACKOFF_SEC = 0.6
DEFAULT_CAPTURE_TIMEOUT_SEC = 2.0
MAX_FRAME_AGE_SEC = 0.3
MAX_STEREO_FRAME_SKEW_SEC = 0.025


def _safe_release(cap: Optional[cv2.VideoCapture]) -> None:
    if cap is None:
        return
    try:
        cap.release()
    except Exception:
        pass


def _windows_backends() -> List[int]:
    backends: List[int] = []
    for name in ("CAP_DSHOW", "CAP_MSMF"):
        if hasattr(cv2, name):
            backends.append(getattr(cv2, name))
    return backends


def _open_capture_with_backend(
    idx: int, backend: Optional[int], timeout_sec: Optional[float] = None
) -> Optional[cv2.VideoCapture]:
    params: List[int] = []
    if timeout_sec is not None:
        timeout_ms = max(100, int(timeout_sec * 1000))
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms])
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms])

    try:
        if backend is not None and params:
            cap = cv2.VideoCapture(idx, backend, params)
        elif backend is not None:
            cap = cv2.VideoCapture(idx, backend)
        elif params:
            cap = cv2.VideoCapture(idx, params)
        else:
            cap = cv2.VideoCapture(idx)
    except Exception:
        return None

    if cap is None:
        return None
    if not cap.isOpened():
        _safe_release(cap)
        return None
    return cap


def open_capture(idx: int, timeout_sec: Optional[float] = None) -> Optional[cv2.VideoCapture]:
    if timeout_sec is None:
        timeout_sec = DEFAULT_CAPTURE_TIMEOUT_SEC

    if sys.platform.startswith("win"):
        for backend in _windows_backends():
            cap = _open_capture_with_backend(idx, backend, timeout_sec=timeout_sec)
            if cap is not None:
                return cap
            cap = _open_capture_with_backend(idx, backend)
            if cap is not None:
                return cap
        return None

    backend = getattr(cv2, "CAP_V4L2", None) if hasattr(cv2, "CAP_V4L2") else None
    cap = _open_capture_with_backend(idx, backend, timeout_sec=timeout_sec)
    if cap is None:
        cap = _open_capture_with_backend(idx, backend)
    return cap


def _apply_auto_exposure(cap: cv2.VideoCapture, enabled: bool) -> None:
    if not hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
        return
    primary_val = 3 if enabled else 1
    fallback_val = 0.75 if enabled else 0.25
    try:
        if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, primary_val):
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, fallback_val)
    except Exception:
        pass


def parse_resolution(res_str: str) -> Tuple[int, int]:
    try:
        w, h = [int(part) for part in str(res_str).lower().split("x")]
        if w > 0 and h > 0:
            return w, h
    except ValueError:
        pass
    return 1280, 720


def apply_camera_settings(cap: Optional[cv2.VideoCapture], settings: dict) -> None:
    if cap is None:
        return
    try:
        if not cap.isOpened():
            return
    except Exception:
        return

    width, height = parse_resolution(settings.get("resolution", "1280x720"))
    fps = int(settings.get("fps", 30))

    try:
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            cap.set(
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                int(DEFAULT_CAPTURE_TIMEOUT_SEC * 1000),
            )

        autofocus = bool(settings.get("autofocus", False))
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        if not autofocus:
            cap.set(cv2.CAP_PROP_FOCUS, int(settings.get("focus", 0)))

        auto_exposure = bool(settings.get("auto_exposure", True))
        _apply_auto_exposure(cap, auto_exposure)
        if not auto_exposure:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(settings.get("exposure", -6)))

        cap.set(cv2.CAP_PROP_BRIGHTNESS, float(settings.get("brightness", 128)))
        cap.set(cv2.CAP_PROP_CONTRAST, float(settings.get("contrast", 128)))
    except Exception:
        pass


def list_video_device_indices(max_devices: int = 8) -> List[int]:
    if not sys.platform.startswith("win"):
        paths = sorted(glob.glob("/dev/video*"))
        indices: List[int] = []
        for p in paths:
            m = re.search(r"(\d+)$", p)
            if m:
                idx = int(m.group(1))
                if idx < max_devices and idx not in indices:
                    indices.append(idx)
        if indices:
            return indices
    return list(range(max_devices))


def probe_single_camera(idx: int, timeout_sec: float = 1.2) -> Optional[int]:
    result: List[Optional[int]] = [None]

    def _worker():
        cap = open_capture(idx, timeout_sec=timeout_sec)
        if cap is None:
            return
        try:
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                result[0] = idx
        except Exception:
            pass
        finally:
            _safe_release(cap)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec + 0.25)
    return result[0]


def scan_available_cameras(max_devices: int = 8) -> List[int]:
    candidates = list_video_device_indices(max_devices)
    available: List[int] = []
    for idx in candidates:
        found = probe_single_camera(idx, timeout_sec=1.2)
        if found is not None:
            available.append(found)
    return available if available else candidates[:2]


class SingleThreadStereoCapture:
    """Owns open/read/release exclusively inside _capture_loop and serializes scans."""

    def __init__(
        self,
        settings: dict,
        left_idx: int,
        right_idx: int,
        on_scan_complete: Optional[Callable[[List[int]], None]] = None,
        poll_interval_sec: float = 0.015,
    ):
        self._state_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._running = True
        self._poll_interval_sec = poll_interval_sec
        self._on_scan_complete = on_scan_complete

        self._settings = dict(settings)
        self._left_idx = int(left_idx)
        self._right_idx = int(right_idx)
        self._reopen_requested = True
        self._props_dirty = False
        self._scan_requested = False

        self._cap_left: Optional[cv2.VideoCapture] = None
        self._cap_right: Optional[cv2.VideoCapture] = None
        self._latest_left: Optional[np.ndarray] = None
        self._latest_right: Optional[np.ndarray] = None
        self._left_timestamp: Optional[float] = None
        self._right_timestamp: Optional[float] = None
        self._left_online = False
        self._right_online = False
        self._fps_left = 0.0
        self._fps_right = 0.0
        self._last_left_time = time.perf_counter()
        self._last_right_time = time.perf_counter()
        self._left_fail_count = 0
        self._right_fail_count = 0
        self._last_left_reconnect_try = 0.0
        self._last_right_reconnect_try = 0.0

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def set_indices(self, left_idx: int, right_idx: int) -> None:
        with self._state_lock:
            self._left_idx = int(left_idx)
            self._right_idx = int(right_idx)
            self._reopen_requested = True
            self._latest_left = None
            self._latest_right = None
            self._left_timestamp = None
            self._right_timestamp = None
            self._left_online = False
            self._right_online = False

    def update_settings(self, settings: dict) -> None:
        with self._state_lock:
            self._settings = dict(settings)
            self._props_dirty = True

    def request_scan(self) -> None:
        with self._state_lock:
            self._scan_requested = True

    def get_snapshot(self, copy_frames: bool = True) -> Dict[str, object]:
        with self._state_lock:
            return {
                "left_frame": (
                    None
                    if self._latest_left is None
                    else self._latest_left.copy()
                    if copy_frames
                    else self._latest_left
                ),
                "right_frame": (
                    None
                    if self._latest_right is None
                    else self._latest_right.copy()
                    if copy_frames
                    else self._latest_right
                ),
                "left_timestamp": self._left_timestamp,
                "right_timestamp": self._right_timestamp,
                "left_online": self._left_online,
                "right_online": self._right_online,
                "fps_left": self._fps_left,
                "fps_right": self._fps_right,
                "left_idx": self._left_idx,
                "right_idx": self._right_idx,
            }

    def get_synchronized_snapshot(
        self,
        max_skew_sec: float = MAX_STEREO_FRAME_SKEW_SEC,
        max_age_sec: float = MAX_FRAME_AGE_SEC,
    ) -> Dict[str, object]:
        snapshot = self.get_snapshot(copy_frames=False)
        left_timestamp = snapshot["left_timestamp"]
        right_timestamp = snapshot["right_timestamp"]
        now = time.perf_counter()
        synchronized = (
            snapshot["left_frame"] is not None
            and snapshot["right_frame"] is not None
            and isinstance(left_timestamp, float)
            and isinstance(right_timestamp, float)
            and abs(left_timestamp - right_timestamp) <= max_skew_sec
            and now - left_timestamp <= max_age_sec
            and now - right_timestamp <= max_age_sec
        )
        if not synchronized:
            snapshot["left_frame"] = None
            snapshot["right_frame"] = None
        return snapshot

    def close(self) -> None:
        with self._state_lock:
            self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            with self._state_lock:
                cap_l, cap_r = self._cap_left, self._cap_right
            _safe_release(cap_l)
            _safe_release(cap_r)
            self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            warnings.warn(
                "Camera capture did not stop after release; its daemon thread is still blocked.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        with self._io_lock:
            self._close_both_unlocked()

    def _close_both_unlocked(self) -> None:
        _safe_release(self._cap_left)
        _safe_release(self._cap_right)
        self._cap_left = None
        self._cap_right = None
        with self._state_lock:
            self._left_online = False
            self._right_online = False

    def _open_both_unlocked(self, left_idx: int, right_idx: int, settings: dict) -> None:
        self._close_both_unlocked()
        cap_l = open_capture(left_idx)
        if cap_l is not None:
            apply_camera_settings(cap_l, settings)

        if right_idx == left_idx:
            cap_r = None
        else:
            cap_r = open_capture(right_idx)
            if cap_r is not None:
                apply_camera_settings(cap_r, settings)

        now = time.perf_counter()
        with self._state_lock:
            self._cap_left = cap_l
            self._cap_right = cap_r
            self._left_online = cap_l is not None
            self._right_online = cap_r is not None
            self._left_fail_count = 0
            self._right_fail_count = 0
            self._last_left_time = now
            self._last_right_time = now
            self._latest_left = None
            self._latest_right = None
            self._left_timestamp = None
            self._right_timestamp = None

    def _reopen_side_unlocked(self, side: str, cam_idx: int, settings: dict, now: float) -> None:
        is_left = side == "left"
        with self._state_lock:
            old_cap = self._cap_left if is_left else self._cap_right
            if is_left:
                self._cap_left = None
                self._left_online = False
            else:
                self._cap_right = None
                self._right_online = False

        _safe_release(old_cap)
        new_cap = open_capture(cam_idx)
        if new_cap is not None:
            apply_camera_settings(new_cap, settings)

        with self._state_lock:
            if not self._running:
                _safe_release(new_cap)
                return
            if is_left:
                self._cap_left = new_cap
                self._left_online = new_cap is not None
                self._left_fail_count = 0
                self._latest_left = None
                self._left_timestamp = None
                if new_cap is not None:
                    self._last_left_time = now
            else:
                self._cap_right = new_cap
                self._right_online = new_cap is not None
                self._right_fail_count = 0
                self._latest_right = None
                self._right_timestamp = None
                if new_cap is not None:
                    self._last_right_time = now

    def _poll_side_unlocked(self, side: str, cam_idx: int, settings: dict, now: float) -> None:
        is_left = side == "left"
        with self._state_lock:
            cap = self._cap_left if is_left else self._cap_right
            last_try = self._last_left_reconnect_try if is_left else self._last_right_reconnect_try

        if cap is None:
            if now - last_try >= RECONNECT_BACKOFF_SEC:
                with self._state_lock:
                    if is_left:
                        self._last_left_reconnect_try = now
                    else:
                        self._last_right_reconnect_try = now
                self._reopen_side_unlocked(side, cam_idx, settings, now)
            return

        ret = False
        frame: Optional[np.ndarray] = None
        try:
            ret, frame = cap.read()
        except Exception:
            ret, frame = False, None

        if ret and frame is not None and frame.size > 0:
            timestamp = time.perf_counter()
            with self._state_lock:
                if is_left:
                    dt = timestamp - self._last_left_time
                    self._last_left_time = timestamp
                    self._latest_left = frame
                    self._left_timestamp = timestamp
                    self._left_online = True
                    self._left_fail_count = 0
                    if dt > 0:
                        self._fps_left = 0.85 * self._fps_left + 0.15 * (1.0 / dt)
                else:
                    dt = timestamp - self._last_right_time
                    self._last_right_time = timestamp
                    self._latest_right = frame
                    self._right_timestamp = timestamp
                    self._right_online = True
                    self._right_fail_count = 0
                    if dt > 0:
                        self._fps_right = 0.85 * self._fps_right + 0.15 * (1.0 / dt)
            return

        with self._state_lock:
            if is_left:
                self._left_fail_count += 1
                fails = self._left_fail_count
                self._fps_left = 0.0
                if fails >= MAX_CONSECUTIVE_FAILURES_BEFORE_REOPEN:
                    self._left_online = False
            else:
                self._right_fail_count += 1
                fails = self._right_fail_count
                self._fps_right = 0.0
                if fails >= MAX_CONSECUTIVE_FAILURES_BEFORE_REOPEN:
                    self._right_online = False

        if fails >= MAX_CONSECUTIVE_FAILURES_BEFORE_REOPEN and (now - last_try >= RECONNECT_BACKOFF_SEC):
            with self._state_lock:
                if is_left:
                    self._last_left_reconnect_try = now
                else:
                    self._last_right_reconnect_try = now
            self._reopen_side_unlocked(side, cam_idx, settings, now)

    def _poll_pair_unlocked(
        self, left_idx: int, right_idx: int, settings: dict, now: float
    ) -> None:
        with self._state_lock:
            cap_l = self._cap_left
            cap_r = self._cap_right

        if cap_l is None or cap_r is None:
            self._poll_side_unlocked("left", left_idx, settings, now)
            self._poll_side_unlocked("right", right_idx, settings, now)
            return

        try:
            grabbed_l = cap_l.grab()
            timestamp_l = time.perf_counter()
        except Exception:
            grabbed_l, timestamp_l = False, time.perf_counter()
        try:
            grabbed_r = cap_r.grab()
            timestamp_r = time.perf_counter()
        except Exception:
            grabbed_r, timestamp_r = False, time.perf_counter()

        frames: Dict[str, Tuple[bool, Optional[np.ndarray], float]] = {
            "left": (False, None, timestamp_l),
            "right": (False, None, timestamp_r),
        }
        for side, cap, grabbed, timestamp in (
            ("left", cap_l, grabbed_l, timestamp_l),
            ("right", cap_r, grabbed_r, timestamp_r),
        ):
            if grabbed:
                try:
                    ret, frame = cap.retrieve()
                except Exception:
                    ret, frame = False, None
                if ret and frame is not None and frame.size > 0:
                    frames[side] = (True, frame, timestamp)

        for side, cam_idx in (("left", left_idx), ("right", right_idx)):
            ok, frame, timestamp = frames[side]
            is_left = side == "left"
            if ok and frame is not None:
                with self._state_lock:
                    if is_left:
                        dt = timestamp - self._last_left_time
                        self._last_left_time = timestamp
                        self._latest_left = frame
                        self._left_timestamp = timestamp
                        self._left_online = True
                        self._left_fail_count = 0
                        if dt > 0:
                            self._fps_left = 0.85 * self._fps_left + 0.15 / dt
                    else:
                        dt = timestamp - self._last_right_time
                        self._last_right_time = timestamp
                        self._latest_right = frame
                        self._right_timestamp = timestamp
                        self._right_online = True
                        self._right_fail_count = 0
                        if dt > 0:
                            self._fps_right = 0.85 * self._fps_right + 0.15 / dt
                continue

            with self._state_lock:
                if is_left:
                    self._left_fail_count += 1
                    fails = self._left_fail_count
                    self._fps_left = 0.0
                    if fails >= MAX_CONSECUTIVE_FAILURES_BEFORE_REOPEN:
                        self._left_online = False
                else:
                    self._right_fail_count += 1
                    fails = self._right_fail_count
                    self._fps_right = 0.0
                    if fails >= MAX_CONSECUTIVE_FAILURES_BEFORE_REOPEN:
                        self._right_online = False

            if fails >= MAX_CONSECUTIVE_FAILURES_BEFORE_REOPEN:
                self._reopen_side_unlocked(side, cam_idx, settings, now)

    def _capture_loop(self) -> None:
        while True:
            with self._state_lock:
                if not self._running:
                    break
                want_scan = self._scan_requested
                want_reopen = self._reopen_requested
                props_dirty = self._props_dirty
                left_idx = self._left_idx
                right_idx = self._right_idx
                settings = dict(self._settings)
                if want_scan:
                    self._scan_requested = False
                if want_reopen:
                    self._reopen_requested = False
                if props_dirty:
                    self._props_dirty = False

            with self._io_lock:
                if want_scan:
                    self._close_both_unlocked()
                    cams = scan_available_cameras()
                    with self._state_lock:
                        still_running = self._running
                        self._reopen_requested = True
                    if still_running and self._on_scan_complete is not None:
                        self._on_scan_complete(cams)
                    continue

                if want_reopen:
                    self._open_both_unlocked(left_idx, right_idx, settings)
                elif props_dirty:
                    with self._state_lock:
                        cap_l, cap_r = self._cap_left, self._cap_right
                    apply_camera_settings(cap_l, settings)
                    if cap_r is not cap_l:
                        apply_camera_settings(cap_r, settings)

                now = time.perf_counter()
                if right_idx == left_idx:
                    self._poll_side_unlocked("left", left_idx, settings, now)
                else:
                    self._poll_pair_unlocked(left_idx, right_idx, settings, now)

            time.sleep(self._poll_interval_sec)