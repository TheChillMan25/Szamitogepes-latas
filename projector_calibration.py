import glob
import json
import os
import re
import threading
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk


SETTINGS_FILE = "./camera_settings.json"
CALIB_PARAMS_FILE = "./calibration_params.json"
PROJECTOR_CALIB_FILE = "./projector_calibration.json"
ARUCO_BOARD_IMAGE_FILE = "./projector_aruco_board.png"

DEFAULT_CAM_SETTINGS = {
    "left_camera_index": 0,
    "right_camera_index": 1,
    "resolution": "1280x720",
    "fps": 30,
    "focus": 0,
    "autofocus": False,
    "auto_exposure": True,
    "exposure": -6,
    "brightness": 128,
    "contrast": 128,
}

ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
}

MAX_CONSECUTIVE_FAILURES_BEFORE_REOPEN = 5
RECONNECT_BACKOFF_SEC = 0.6


def _open_capture(idx: int) -> cv2.VideoCapture:
    if sys.platform.startswith("win"):
        for backend_name in ("CAP_DSHOW", "CAP_MSMF"):
            if hasattr(cv2, backend_name):
                cap = cv2.VideoCapture(idx, getattr(cv2, backend_name))
                if cap is not None and cap.isOpened():
                    return cap
                if cap is not None:
                    cap.release()
        return cv2.VideoCapture(idx, getattr(cv2, "CAP_ANY", 0))

    if hasattr(cv2, "CAP_V4L2"):
        return cv2.VideoCapture(idx, cv2.CAP_V4L2)
    return cv2.VideoCapture(idx)


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


def _apply_auto_exposure(cap: cv2.VideoCapture, enabled: bool):
    if not hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
        return
    primary_val = 3 if enabled else 1
    fallback_val = 0.75 if enabled else 0.25
    if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, primary_val):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, fallback_val)


def probe_single_camera(idx: int, timeout_sec: float = 1.5) -> Optional[int]:
    result: List[Optional[int]] = [None]

    def _worker():
        cap = _open_capture(idx)
        try:
            if cap.isOpened():
                if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_sec * 1000))
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout_sec * 1000))
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    result[0] = idx
        finally:
            cap.release()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    return result[0]


def scan_available_cameras(max_devices: int = 8) -> List[int]:
    candidates = list_video_device_indices(max_devices)
    available = []
    for idx in candidates:
        found = probe_single_camera(idx, timeout_sec=1.5)
        if found is not None:
            available.append(found)
    return available if available else candidates[:2]


def get_aruco_detector(dict_name: str):
    dict_id = ARUCO_DICTS.get(dict_name, cv2.aruco.DICT_4X4_50)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "ArucoDetector") and hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
        return dictionary, cv2.aruco.ArucoDetector(dictionary, params)
    params = cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, "DetectorParameters_create") else None
    return dictionary, params


def detect_aruco_markers(
    gray: np.ndarray,
    dictionary,
    detector_or_params,
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    if hasattr(detector_or_params, "detectMarkers"):
        corners, ids, _rejected = detector_or_params.detectMarkers(gray)
    else:
        corners, ids, _rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_or_params)
    return list(corners) if corners is not None else [], ids


def generate_projector_aruco_image(
    proj_w: int = 1280,
    proj_h: int = 720,
    marker_px: int = 180,
    margin_px: int = 90,
    dict_name: str = "DICT_4X4_50",
    out_path: str = ARUCO_BOARD_IMAGE_FILE,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Generate a 2x2 ArUco marker board image (IDs 0, 1, 2, 3 at top-left, top-right,
    bottom-right, bottom-left) for display on the projector.
    Returns (image_bgr, marker_centers_px).
    """
    dictionary, _ = get_aruco_detector(dict_name)
    canvas = np.full((proj_h, proj_w, 3), 255, dtype=np.uint8)

    # Place 4 markers near the corners of the projected region
    positions = {
        0: (margin_px, margin_px),                                      # Top-Left
        1: (proj_w - margin_px - marker_px, margin_px),                 # Top-Right
        2: (proj_w - margin_px - marker_px, proj_h - margin_px - marker_px),  # Bottom-Right
        3: (margin_px, proj_h - margin_px - marker_px),                 # Bottom-Left
    }

    centers: Dict[int, np.ndarray] = {}
    for marker_id, (x0, y0) in positions.items():
        if hasattr(cv2.aruco, "generateImageMarker"):
            marker_gray = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
        else:
            marker_gray = cv2.aruco.drawMarker(dictionary, marker_id, marker_px)

        marker_bgr = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
        # Draw thin border + ID label outside marker box
        cv2.rectangle(canvas, (x0 - 4, y0 - 4), (x0 + marker_px + 4, y0 + marker_px + 4), (0, 0, 0), 2)
        canvas[y0 : y0 + marker_px, x0 : x0 + marker_px] = marker_bgr
        cv2.putText(
            canvas,
            f"ID {marker_id}",
            (x0, max(24, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        centers[marker_id] = np.array([x0 + marker_px / 2.0, y0 + marker_px / 2.0], dtype=np.float64)

    # Draw subtle outer frame boundary showing exact projector canvas edges
    cv2.rectangle(canvas, (2, 2), (proj_w - 3, proj_h - 3), (40, 40, 40), 2)
    cv2.imwrite(out_path, canvas)
    return canvas, centers


def triangulate_point_pair(
    pt_left_px: np.ndarray,
    pt_right_px: np.ndarray,
    mtx_l: np.ndarray,
    dist_l: np.ndarray,
    mtx_r: np.ndarray,
    dist_r: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """
    Undistort 2D pixel points to normalized camera coordinates and triangulate
    into the Left Camera coordinate frame (R1=I, t1=0; R2=R, t2=T).
    """
    u_l = cv2.undistortPoints(
        np.array([[[pt_left_px[0], pt_left_px[1]]]], dtype=np.float64), mtx_l, dist_l
    ).reshape(2, 1)
    u_r = cv2.undistortPoints(
        np.array([[[pt_right_px[0], pt_right_px[1]]]], dtype=np.float64), mtx_r, dist_r
    ).reshape(2, 1)

    P1 = np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    P2 = np.hstack([R.astype(np.float64), T.astype(np.float64).reshape(3, 1)])

    pts4d = cv2.triangulatePoints(P1, P2, u_l, u_r)
    w = pts4d[3, 0]
    if abs(w) < 1e-9:
        raise ValueError("Degenerate homogeneous coordinate during triangulation.")
    return (pts4d[:3, 0] / w).astype(np.float64)


class ProjectorCalibrationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Projector Plane & Coordinate Calibration")
        self.root.geometry("1380x860")

        self.cam_settings = self._load_cam_settings()
        self.stereo_params = self._load_stereo_params()
        self.proj_calib_result: Optional[dict] = self._load_existing_projector_calib()

        self.lock = threading.Lock()
        self.running = True
        self.cap_left: Optional[cv2.VideoCapture] = None
        self.cap_right: Optional[cv2.VideoCapture] = None
        self.latest_left: Optional[np.ndarray] = None
        self.latest_right: Optional[np.ndarray] = None
        self.left_online = False
        self.right_online = False
        self.fps_left = 0.0
        self.fps_right = 0.0
        self._last_left_time = time.perf_counter()
        self._last_right_time = time.perf_counter()
        self._left_fail_count = 0
        self._right_fail_count = 0
        self._last_left_reconnect_try = 0.0
        self._last_right_reconnect_try = 0.0

        self._build_ui()
        self._apply_cam_settings_to_widgets()
        if self.proj_calib_result:
            self._display_projector_calib(self.proj_calib_result)

        initial_cams = list_video_device_indices()
        self.left_cam_combo["values"] = initial_cams
        self.right_cam_combo["values"] = initial_cams

        if self.stereo_params is None:
            self.status_var.set(
                f"Warning: {CALIB_PARAMS_FILE} not found. Run stereo calibration first."
            )
        else:
            self.status_var.set(
                f"Loaded stereo calibration from {CALIB_PARAMS_FILE}. Generate ArUco image or calibrate projector."
            )

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(50, self._async_initial_camera_setup)
        self._update_ui_loop()

    # ------------------------------------------------------------------
    # File Loading & Saving
    # ------------------------------------------------------------------
    def _load_cam_settings(self) -> dict:
        data = dict(DEFAULT_CAM_SETTINGS)
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data.update(json.load(f))
            except Exception:
                pass
        return data

    def _load_stereo_params(self) -> Optional[dict]:
        if not os.path.exists(CALIB_PARAMS_FILE):
            return None
        try:
            with open(CALIB_PARAMS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"Failed to read {CALIB_PARAMS_FILE}: {exc}")
            return None

    def _load_existing_projector_calib(self) -> Optional[dict]:
        if not os.path.exists(PROJECTOR_CALIB_FILE):
            return None
        try:
            with open(PROJECTOR_CALIB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # UI Layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.status_var = tk.StringVar(value="Initializing...")

        top_bar = ttk.Frame(self.root, padding=8)
        top_bar.pack(fill=tk.X)

        ttk.Label(top_bar, text="Left Cam:").pack(side=tk.LEFT, padx=(0, 4))
        self.left_cam_var = tk.IntVar()
        self.left_cam_combo = ttk.Combobox(
            top_bar, textvariable=self.left_cam_var, values=[], width=5, state="readonly"
        )
        self.left_cam_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.left_cam_combo.bind("<<ComboboxSelected>>", lambda _e: self._open_selected_cameras())

        ttk.Label(top_bar, text="Right Cam:").pack(side=tk.LEFT, padx=(0, 4))
        self.right_cam_var = tk.IntVar()
        self.right_cam_combo = ttk.Combobox(
            top_bar, textvariable=self.right_cam_var, values=[], width=5, state="readonly"
        )
        self.right_cam_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.right_cam_combo.bind("<<ComboboxSelected>>", lambda _e: self._open_selected_cameras())

        ttk.Button(top_bar, text="Rescan Cams", command=self._rescan_cameras).pack(side=tk.LEFT, padx=(0, 16))

        ttk.Button(
            top_bar, text="Generate ArUco Image", command=self.generate_aruco_pattern_action
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            top_bar, text="Calibrate Projector & Save JSON", command=self.calibrate_projector_action
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            top_bar, text="Reload Stereo JSON", command=self.reload_stereo_action
        ).pack(side=tk.LEFT, padx=4)

        cfg_frame = ttk.LabelFrame(self.root, text="Projector & ArUco Pattern Settings", padding=8)
        cfg_frame.pack(fill=tk.X, padx=8, pady=(0, 6))

        self.dict_var = tk.StringVar(value="DICT_4X4_50")
        ttk.Label(cfg_frame, text="ArUco Dict:").grid(row=0, column=0, sticky=tk.W, padx=4)
        ttk.Combobox(
            cfg_frame, textvariable=self.dict_var, values=list(ARUCO_DICTS.keys()), width=14, state="readonly"
        ).grid(row=0, column=1, padx=4)

        self.proj_w_var = tk.IntVar(value=1280)
        self.proj_h_var = tk.IntVar(value=720)
        ttk.Label(cfg_frame, text="Projector Res (W×H):").grid(row=0, column=2, sticky=tk.W, padx=4)
        res_sub = ttk.Frame(cfg_frame)
        res_sub.grid(row=0, column=3, sticky=tk.W, padx=4)
        ttk.Entry(res_sub, textvariable=self.proj_w_var, width=6).pack(side=tk.LEFT)
        ttk.Label(res_sub, text="×").pack(side=tk.LEFT, padx=2)
        ttk.Entry(res_sub, textvariable=self.proj_h_var, width=6).pack(side=tk.LEFT)

        self.marker_px_var = tk.IntVar(value=180)
        self.margin_px_var = tk.IntVar(value=90)
        ttk.Label(cfg_frame, text="Marker Size (px):").grid(row=0, column=4, sticky=tk.W, padx=4)
        ttk.Entry(cfg_frame, textvariable=self.marker_px_var, width=6).grid(row=0, column=5, padx=4)

        ttk.Label(cfg_frame, text="Edge Margin (px):").grid(row=0, column=6, sticky=tk.W, padx=4)
        ttk.Entry(cfg_frame, textvariable=self.margin_px_var, width=6).grid(row=0, column=7, padx=4)

        # Optional known physical width/height of the projected rectangle (mm).
        # If left <= 0, physical size is derived directly from stereo triangulation metrics.
        self.known_w_mm_var = tk.DoubleVar(value=0.0)
        self.known_h_mm_var = tk.DoubleVar(value=0.0)
        ttk.Label(cfg_frame, text="Known Width (mm, 0=auto):").grid(row=1, column=0, sticky=tk.W, padx=4, pady=(6, 0))
        ttk.Entry(cfg_frame, textvariable=self.known_w_mm_var, width=8).grid(row=1, column=1, sticky=tk.W, padx=4, pady=(6, 0))

        ttk.Label(cfg_frame, text="Known Height (mm, 0=auto):").grid(row=1, column=2, sticky=tk.W, padx=4, pady=(6, 0))
        ttk.Entry(cfg_frame, textvariable=self.known_h_mm_var, width=8).grid(row=1, column=3, sticky=tk.W, padx=4, pady=(6, 0))

        self.aruco_preview_info_var = tk.StringVar(value=f"Target file: {ARUCO_BOARD_IMAGE_FILE}")
        ttk.Label(cfg_frame, textvariable=self.aruco_preview_info_var).grid(
            row=1, column=4, columnspan=4, sticky=tk.W, padx=8, pady=(6, 0)
        )

        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))

        cams_pane = ttk.Frame(main_pane)
        cams_pane.columnconfigure(0, weight=1, uniform="proj_cams")
        cams_pane.columnconfigure(1, weight=1, uniform="proj_cams")
        cams_pane.rowconfigure(0, weight=1)
        main_pane.add(cams_pane, weight=4)

        left_box = ttk.LabelFrame(cams_pane, text="Left Camera (ArUco Detection)", padding=4)
        left_box.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left_box.grid_propagate(False)
        left_box.columnconfigure(0, weight=1)
        left_box.rowconfigure(0, weight=1)
        self.left_label = tk.Label(left_box, bg="black", anchor="center")
        self.left_label.grid(row=0, column=0, sticky="nsew")

        right_box = ttk.LabelFrame(cams_pane, text="Right Camera (ArUco Detection)", padding=4)
        right_box.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right_box.grid_propagate(False)
        right_box.columnconfigure(0, weight=1)
        right_box.rowconfigure(0, weight=1)
        self.right_label = tk.Label(right_box, bg="black", anchor="center")
        self.right_label.grid(row=0, column=0, sticky="nsew")

        right_panel = ttk.Frame(main_pane)
        main_pane.add(right_panel, weight=2)

        aruco_box = ttk.LabelFrame(right_panel, text="Generated Projector ArUco Pattern", padding=4)
        aruco_box.pack(fill=tk.X, pady=(0, 6))
        self.aruco_thumb_label = tk.Label(aruco_box, bg="#222222", height=11)
        self.aruco_thumb_label.pack(fill=tk.X)

        out_box = ttk.LabelFrame(right_panel, text="Projector Calibration Output (JSON)", padding=6)
        out_box.pack(fill=tk.BOTH, expand=True)
        self.result_text = tk.Text(out_box, width=42, wrap=tk.WORD, font=("Consolas", 9))
        self.result_text.pack(fill=tk.BOTH, expand=True)

        status_bar = ttk.Frame(self.root, padding=(8, 4))
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(status_bar, textvariable=self.status_var).pack(side=tk.LEFT)

    def _apply_cam_settings_to_widgets(self):
        self.left_cam_var.set(self.cam_settings.get("left_camera_index", 0))
        self.right_cam_var.set(self.cam_settings.get("right_camera_index", 1))

    # ------------------------------------------------------------------
    # Camera Loop & Controls
    # ------------------------------------------------------------------
    def _async_initial_camera_setup(self):
        def _worker():
            cams = scan_available_cameras()
            if not self.running:
                return
            self.root.after(0, lambda c=cams: self._on_initial_scan_done(c))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_initial_scan_done(self, cams: List[int]):
        if not self.running:
            return
        self.left_cam_combo["values"] = cams
        self.right_cam_combo["values"] = cams
        if len(cams) >= 2:
            if int(self.left_cam_var.get()) not in cams:
                self.left_cam_var.set(cams[0])
            if int(self.right_cam_var.get()) not in cams:
                self.right_cam_var.set(cams[1])
        elif len(cams) == 1:
            self.left_cam_var.set(cams[0])
            self.right_cam_var.set(cams[0])
        self._open_selected_cameras()

    def _rescan_cameras(self):
        self.status_var.set("Rescanning cameras...")

        def _worker():
            cams = scan_available_cameras()
            if not self.running:
                return
            self.root.after(0, lambda c=cams: self._on_rescan_done(c))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_rescan_done(self, cams: List[int]):
        if not self.running:
            return
        self.left_cam_combo["values"] = cams
        self.right_cam_combo["values"] = cams
        self.status_var.set(f"Detected cameras: {cams}")

    def _release_single_cap(self, cap: Optional[cv2.VideoCapture]):
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _release_cameras(self):
        with self.lock:
            cap_l = self.cap_left
            cap_r = self.cap_right
            self.cap_left = None
            self.cap_right = None
            self.left_online = False
            self.right_online = False
        self._release_single_cap(cap_l)
        self._release_single_cap(cap_r)

    def _get_target_res_and_fps(self) -> Tuple[int, int, int]:
        res_str = str(self.cam_settings.get("resolution", "1280x720"))
        try:
            w, h = [int(x) for x in res_str.split("x")]
        except ValueError:
            w, h = 1280, 720
        return w, h, int(self.cam_settings.get("fps", 30))

    def _apply_single_cam_props(self, cap: Optional[cv2.VideoCapture], width: int, height: int, fps: int):
        if cap is None or not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        autofocus = bool(self.cam_settings.get("autofocus", False))
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        if not autofocus:
            cap.set(cv2.CAP_PROP_FOCUS, int(self.cam_settings.get("focus", 0)))

        auto_exp = bool(self.cam_settings.get("auto_exposure", True))
        _apply_auto_exposure(cap, auto_exp)
        if not auto_exp:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(self.cam_settings.get("exposure", -6)))

        cap.set(cv2.CAP_PROP_BRIGHTNESS, float(self.cam_settings.get("brightness", 128)))
        cap.set(cv2.CAP_PROP_CONTRAST, float(self.cam_settings.get("contrast", 128)))

    def _open_selected_cameras(self):
        self._release_cameras()
        left_idx = int(self.left_cam_var.get())
        right_idx = int(self.right_cam_var.get())
        w, h, fps = self._get_target_res_and_fps()

        cap_l = _open_capture(left_idx)
        cap_r = _open_capture(right_idx)
        if cap_l and cap_l.isOpened():
            self._apply_single_cam_props(cap_l, w, h, fps)
        if cap_r and cap_r.isOpened():
            self._apply_single_cam_props(cap_r, w, h, fps)

        with self.lock:
            self.cap_left = cap_l
            self.cap_right = cap_r
            self.left_online = bool(cap_l and cap_l.isOpened())
            self.right_online = bool(cap_r and cap_r.isOpened())
            self._left_fail_count = 0
            self._right_fail_count = 0

        self.status_var.set(f"Opened Left Cam #{left_idx} and Right Cam #{right_idx}")

    def _attempt_reconnect_side(self, side: str, now: float):
        is_left = side == "left"
        last_try = self._last_left_reconnect_try if is_left else self._last_right_reconnect_try
        if now - last_try < RECONNECT_BACKOFF_SEC:
            return

        if is_left:
            self._last_left_reconnect_try = now
            cam_idx = int(self.left_cam_var.get())
            old_cap = self.cap_left
        else:
            self._last_right_reconnect_try = now
            cam_idx = int(self.right_cam_var.get())
            old_cap = self.cap_right

        self._release_single_cap(old_cap)
        new_cap = _open_capture(cam_idx)
        w, h, fps = self._get_target_res_and_fps()
        opened = bool(new_cap and new_cap.isOpened())
        if opened:
            self._apply_single_cam_props(new_cap, w, h, fps)

        with self.lock:
            if not self.running:
                self._release_single_cap(new_cap)
                return
            if is_left:
                self.cap_left = new_cap
                self.left_online = opened
                self._left_fail_count = 0
                if opened:
                    self._last_left_time = now
            else:
                self.cap_right = new_cap
                self.right_online = opened
                self._right_fail_count = 0
                if opened:
                    self._last_right_time = now

    def _poll_one_camera(self, side: str, now: float):
        is_left = side == "left"
        with self.lock:
            cap = self.cap_left if is_left else self.cap_right

        if cap is None or not cap.isOpened():
            self._attempt_reconnect_side(side, now)
            return

        try:
            ret, frame = cap.read()
        except Exception:
            ret, frame = False, None

        if ret and frame is not None and frame.size > 0:
            with self.lock:
                if is_left:
                    dt = now - self._last_left_time
                    self._last_left_time = now
                    self.latest_left = frame
                    self.left_online = True
                    self._left_fail_count = 0
                    if dt > 0:
                        self.fps_left = 0.85 * self.fps_left + 0.15 * (1.0 / dt)
                else:
                    dt = now - self._last_right_time
                    self._last_right_time = now
                    self.latest_right = frame
                    self.right_online = True
                    self._right_fail_count = 0
                    if dt > 0:
                        self.fps_right = 0.85 * self.fps_right + 0.15 * (1.0 / dt)
            return

        with self.lock:
            if is_left:
                self._left_fail_count += 1
                fails = self._left_fail_count
                self.fps_left = 0.0
                if fails >= MAX_CONSECUTIVE_FAILURES_BEFORE_REOPEN:
                    self.left_online = False
            else:
                self._right_fail_count += 1
                fails = self._right_fail_count
                self.fps_right = 0.0
                if fails >= MAX_CONSECUTIVE_FAILURES_BEFORE_REOPEN:
                    self.right_online = False

        if fails >= MAX_CONSECUTIVE_FAILURES_BEFORE_REOPEN:
            self._attempt_reconnect_side(side, now)

    def _capture_loop(self):
        while self.running:
            now = time.perf_counter()
            self._poll_one_camera("left", now)
            self._poll_one_camera("right", now)
            time.sleep(0.005)

    # ------------------------------------------------------------------
    # ArUco Pattern Generation & Projector Plane Calibration
    # ------------------------------------------------------------------
    def generate_aruco_pattern_action(self):
        try:
            pw = int(self.proj_w_var.get())
            ph = int(self.proj_h_var.get())
            m_px = int(self.marker_px_var.get())
            margin = int(self.margin_px_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Projector resolution, marker size, and margin must be integers.")
            return

        if pw < 320 or ph < 240 or m_px < 40 or margin < 10:
            messagebox.showerror("Invalid Dimensions", "Please enter valid positive dimensions.")
            return

        img_bgr, _centers = generate_projector_aruco_image(
            proj_w=pw,
            proj_h=ph,
            marker_px=m_px,
            margin_px=margin,
            dict_name=self.dict_var.get(),
            out_path=ARUCO_BOARD_IMAGE_FILE,
        )

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((340, 180), Image.Resampling.BILINEAR)
        self._aruco_thumb_photo = ImageTk.PhotoImage(pil_img)
        self.aruco_thumb_label.configure(image=self._aruco_thumb_photo)
        self.aruco_preview_info_var.set(f"Saved: {ARUCO_BOARD_IMAGE_FILE} ({pw}×{ph})")
        self.status_var.set(
            f"Generated {ARUCO_BOARD_IMAGE_FILE} ({pw}×{ph}) with ArUco IDs 0..3. Display it on the projector and calibrate."
        )

    def reload_stereo_action(self):
        self.stereo_params = self._load_stereo_params()
        if self.stereo_params is None:
            messagebox.showwarning("Missing Calibration", f"Could not load {CALIB_PARAMS_FILE}.")
        else:
            self.status_var.set(f"Reloaded stereo calibration from {CALIB_PARAMS_FILE}.")

    def _extract_marker_centers_map(
        self, corners: List[np.ndarray], ids: Optional[np.ndarray]
    ) -> Dict[int, np.ndarray]:
        result: Dict[int, np.ndarray] = {}
        if ids is None or len(corners) == 0:
            return result
        for c, mid_arr in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            result[int(mid_arr)] = np.mean(pts, axis=0)
        return result

    def calibrate_projector_action(self):
        if self.stereo_params is None:
            self.stereo_params = self._load_stereo_params()
        if self.stereo_params is None:
            messagebox.showerror(
                "Stereo Calibration Missing",
                f"Please calibrate stereo cameras first so {CALIB_PARAMS_FILE} exists.",
            )
            return

        with self.lock:
            frame_l = None if self.latest_left is None else self.latest_left.copy()
            frame_r = None if self.latest_right is None else self.latest_right.copy()

        if frame_l is None or frame_r is None:
            messagebox.showwarning("No Frames", "Waiting for frames from both cameras.")
            return

        dictionary, detector = get_aruco_detector(self.dict_var.get())
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        corners_l, ids_l = detect_aruco_markers(gray_l, dictionary, detector)
        corners_r, ids_r = detect_aruco_markers(gray_r, dictionary, detector)

        centers_l = self._extract_marker_centers_map(corners_l, ids_l)
        centers_r = self._extract_marker_centers_map(corners_r, ids_r)

        common_ids = sorted(set(centers_l.keys()) & set(centers_r.keys()))
        if len(common_ids) < 3:
            messagebox.showinfo(
                "Not Enough Common Markers",
                f"Need at least 3 ArUco markers detected in BOTH cameras.\n"
                f"Left IDs: {sorted(centers_l.keys())}, Right IDs: {sorted(centers_r.keys())}",
            )
            return

        try:
            mtx_l = np.array(self.stereo_params["camera_matrix_left"], dtype=np.float64)
            dist_l = np.array(self.stereo_params["dist_coeffs_left"], dtype=np.float64)
            mtx_r = np.array(self.stereo_params["camera_matrix_right"], dtype=np.float64)
            dist_r = np.array(self.stereo_params["dist_coeffs_right"], dtype=np.float64)
            R = np.array(self.stereo_params["R"], dtype=np.float64)
            T = np.array(self.stereo_params["T"], dtype=np.float64)
        except Exception as exc:
            messagebox.showerror("Invalid Stereo JSON", f"Failed reading stereo matrices:\n{exc}")
            return

        pts_3d_by_id: Dict[int, np.ndarray] = {}
        for mid in common_ids:
            try:
                p3d = triangulate_point_pair(
                    centers_l[mid], centers_r[mid], mtx_l, dist_l, mtx_r, dist_r, R, T
                )
                pts_3d_by_id[mid] = p3d
            except Exception:
                continue

        if len(pts_3d_by_id) < 3:
            messagebox.showerror("Triangulation Failed", "Could not triangulate at least 3 common markers.")
            return

        all_pts = np.vstack(list(pts_3d_by_id.values()))  # (N, 3) in Left Camera frame

        # Fit plane n . (X - X_mean) = 0 via SVD on centered 3D marker points[cite: 2]
        centroid = np.mean(all_pts, axis=0)
        centered = all_pts - centroid
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[-1].astype(np.float64)
        normal /= np.linalg.norm(normal)

        # Orient normal toward the camera (-Z in typical camera coords, or toward camera origin)
        if np.dot(normal, -centroid) < 0:
            normal = -normal

        # Compute horizontal (u_axis) and vertical (v_axis) unit vectors on the projector plane
        # Prefer marker layout IDs: 0=TL, 1=TR, 2=BR, 3=BL when available
        if 0 in pts_3d_by_id and 1 in pts_3d_by_id:
            horiz_raw = pts_3d_by_id[1] - pts_3d_by_id[0]
        elif 3 in pts_3d_by_id and 2 in pts_3d_by_id:
            horiz_raw = pts_3d_by_id[2] - pts_3d_by_id[3]
        else:
            horiz_raw = centered[np.argmax(np.linalg.norm(centered, axis=1))]

        # Project horizontal direction onto plane orthogonal to normal
        horiz_proj = horiz_raw - np.dot(horiz_raw, normal) * normal
        if np.linalg.norm(horiz_proj) < 1e-9:
            horiz_proj = vt[0] - np.dot(vt[0], normal) * normal
        u_axis = horiz_proj / np.linalg.norm(horiz_proj)

        # Vertical unit vector pointing downward in projector/screen coordinates (u x n or orthogonalized 0->3)
        if 0 in pts_3d_by_id and 3 in pts_3d_by_id:
            vert_raw = pts_3d_by_id[3] - pts_3d_by_id[0]
            vert_proj = vert_raw - np.dot(vert_raw, normal) * normal
            # Remove any slight non-orthogonality with u_axis
            vert_proj = vert_proj - np.dot(vert_proj, u_axis) * u_axis
            if np.linalg.norm(vert_proj) > 1e-9:
                v_axis = vert_proj / np.linalg.norm(vert_proj)
            else:
                v_axis = np.cross(normal, u_axis)
        else:
            v_axis = np.cross(normal, u_axis)
        v_axis = v_axis / np.linalg.norm(v_axis)

        # Recompute orthonormal plane normal n = u_axis x v_axis (or right-handed basis)
        plane_normal = np.cross(u_axis, v_axis)
        plane_normal /= np.linalg.norm(plane_normal)
        if np.dot(plane_normal, normal) < 0:
            v_axis = -v_axis
            plane_normal = -plane_normal

        # Project all 3D points onto (u_axis, v_axis) to find span & top-left origin P_origin
        proj_u = centered @ u_axis
        proj_v = centered @ v_axis

        min_u, max_u = float(np.min(proj_u)), float(np.max(proj_u))
        min_v, max_v = float(np.min(proj_v)), float(np.max(proj_v))

        # Adjust span for marker center offset if all 4 corner markers (0,1,2,3) were used
        proj_w = int(self.proj_w_var.get())
        proj_h = int(self.proj_h_var.get())
        m_px = float(self.marker_px_var.get())
        margin_px = float(self.margin_px_var.get())
        center_span_x_px = max(1.0, (proj_w - margin_px - m_px / 2.0) - (margin_px + m_px / 2.0))
        center_span_y_px = max(1.0, (proj_h - margin_px - m_px / 2.0) - (margin_px + m_px / 2.0))

        measured_span_u = max(1e-6, max_u - min_u)
        measured_span_v = max(1e-6, max_v - min_v)

        # Full projected width/height scale factor from marker-center span to full projector image
        scale_u = proj_w / center_span_x_px if len(common_ids) >= 2 else 1.0
        scale_v = proj_h / center_span_y_px if len(common_ids) >= 2 else 1.0

        width_units = measured_span_u * scale_u
        height_units = measured_span_v * scale_v

        known_w = float(self.known_w_mm_var.get())
        known_h = float(self.known_h_mm_var.get())
        if known_w > 0.0:
            width_units = known_w
        if known_h > 0.0:
            height_units = known_h

        # Top-left origin of the calibrated projector plane in Left Camera coordinates
        p_origin = centroid + min_u * u_axis + min_v * v_axis
        if known_w > 0.0 and 0 in pts_3d_by_id:
            p_origin = pts_3d_by_id[0]
        elif known_w > 0.0 and known_h <= 0.0:
            height_units = width_units * (proj_h / max(1, proj_w))
        elif known_h > 0.0 and known_w <= 0.0:
            width_units = height_units * (proj_w / max(1, proj_h))

        # Plane equation: a*x + b*y + c*z + d = 0
        a, b, c = [float(x) for x in plane_normal]
        d = float(-np.dot(plane_normal, p_origin))

        # Plane fit RMS residual in 3D units
        residuals = np.abs(centered @ plane_normal)
        plane_rms = float(np.sqrt(np.mean(residuals ** 2)))

        output_data = {
            "coordinate_frame": "left_camera",
            "unit_note": "Matches units of stereo calibration T vector (typically mm if calibrated with square_size_mm)",
            "detected_marker_ids": common_ids,
            "projector_resolution": {"width_px": proj_w, "height_px": proj_h},
            "projector_size": {
                "width": float(width_units),
                "height": float(height_units),
            },
            "P_origin": [float(x) for x in p_origin],
            "u_axis": [float(x) for x in u_axis],
            "v_axis": [float(x) for x in v_axis],
            "normal": [a, b, c],
            "plane_coeffs": {"a": a, "b": b, "c": c, "d": d},
            "plane_fit_rms": plane_rms,
            "triangulated_markers_3d": {
                str(k): [float(v) for v in vec] for k, vec in pts_3d_by_id.items()
            },
        }

        with open(PROJECTOR_CALIB_FILE, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        self.proj_calib_result = output_data
        self._display_projector_calib(output_data)
        self.status_var.set(
            f"Saved projector calibration to {PROJECTOR_CALIB_FILE} "
            f"(Size: {width_units:.1f} × {height_units:.1f}, Plane RMS: {plane_rms:.4f})"
        )

    def _display_projector_calib(self, data: dict):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Preview Rendering
    # ------------------------------------------------------------------
    def _compute_preview_bounds(self, widget: tk.Widget) -> Tuple[int, int]:
        parent = widget.master
        pw = parent.winfo_width() if parent is not None else 0
        ph = parent.winfo_height() if parent is not None else 0
        ww = widget.winfo_width()
        wh = widget.winfo_height()

        max_w = pw - 12 if pw > 24 else ww
        max_h = ph - 12 if ph > 24 else wh
        if max_w <= 20:
            max_w = 480
        if max_h <= 20:
            max_h = 360
        return max(120, min(max_w, 1280)), max(90, min(max_h, 960))

    def _annotate_and_to_photo(
        self,
        frame: np.ndarray,
        dictionary,
        detector,
        target_w: int,
        target_h: int,
        header: str,
    ) -> ImageTk.PhotoImage:
        vis = frame.copy()
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_aruco_markers(gray, dictionary, detector)
        if corners and ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        cv2.putText(vis, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((target_w, target_h), Image.Resampling.BILINEAR)
        return ImageTk.PhotoImage(pil_img)

    def _update_ui_loop(self):
        if not self.running:
            return

        with self.lock:
            frame_l = None if self.latest_left is None else self.latest_left.copy()
            frame_r = None if self.latest_right is None else self.latest_right.copy()
            fps_l = self.fps_left
            fps_r = self.fps_right
            online_l = self.left_online
            online_r = self.right_online

        dictionary, detector = get_aruco_detector(self.dict_var.get())

        if frame_l is not None:
            tw_l, th_l = self._compute_preview_bounds(self.left_label)
            st_l = f"{fps_l:.1f} FPS" if online_l else "RECONNECTING..."
            photo_l = self._annotate_and_to_photo(frame_l, dictionary, detector, tw_l, th_l, f"LEFT | {st_l}")
            self.left_label.configure(image=photo_l)
            self.left_label.image = photo_l

        if frame_r is not None:
            tw_r, th_r = self._compute_preview_bounds(self.right_label)
            st_r = f"{fps_r:.1f} FPS" if online_r else "RECONNECTING..."
            photo_r = self._annotate_and_to_photo(frame_r, dictionary, detector, tw_r, th_r, f"RIGHT | {st_r}")
            self.right_label.configure(image=photo_r)
            self.right_label.image = photo_r

        self.root.after(25, self._update_ui_loop)

    def on_close(self):
        self.running = False
        if hasattr(self, "capture_thread") and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.5)
        self._release_cameras()
        self.root.destroy()


def main():
    root = tk.Tk()
    ProjectorCalibrationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()