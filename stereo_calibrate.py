import glob
import json
import os
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk


SETTINGS_FILE = "./camera_settings.json"
CALIB_IMAGES_DIR = "./stereo_calibration_images"
CALIB_PARAMS_FILE = "./calibration_params.json"

DEFAULT_SETTINGS = {
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
    "checkerboard_cols": 8,   # inner corners horizontally (9 squares -> 8 inner corners)
    "checkerboard_rows": 6,   # inner corners vertically (7 squares -> 6 inner corners)
    "square_size_mm": 25.0,
}

RESOLUTIONS = ["640x480", "800x600", "1280x720", "1920x1080"]
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
    """Set V4L2/OpenCV auto exposure mode with fallback values across drivers."""
    if not hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
        return
    primary_val = 3 if enabled else 1
    fallback_val = 0.75 if enabled else 0.25
    if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, primary_val):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, fallback_val)


def _orient_corners_top_left(corners: np.ndarray, pattern_size: Tuple[int, int]) -> np.ndarray:
    """Reorder detected checkerboard corners so index 0 is always at the top-left of the grid."""
    cols, rows = pattern_size
    pts = corners.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] != cols * rows:
        return pts.reshape(-1, 1, 2)

    grid = pts.reshape(rows, cols, 2)
    candidates = [grid]

    # 180-degree rotation (most common cause of bottom-right origin)
    candidates.append(grid[::-1, ::-1])

    # Horizontal or vertical flip fallback if grid is inverted along one axis
    candidates.append(grid[::-1, :])
    candidates.append(grid[:, ::-1])

    # If square inner-corner count, also allow 90/270-degree transposed orientations
    if cols == rows:
        t_grid = np.transpose(grid, (1, 0, 2))
        candidates.extend([
            t_grid,
            t_grid[::-1, ::-1],
            t_grid[::-1, :],
            t_grid[:, ::-1],
        ])

    def _top_left_score(g: np.ndarray) -> float:
        tl = g[0, 0]
        # Prefer smallest (x + y), breaking ties by smaller y then x
        return float(tl[0] + tl[1]) + float(tl[1]) * 1e-4

    best_grid = min(candidates, key=_top_left_score)
    return best_grid.reshape(-1, 1, 2).astype(np.float32)


def _detect_checkerboard_corners(gray: np.ndarray, pattern_size: Tuple[int, int]) -> Tuple[bool, Optional[np.ndarray]]:
    """Detect inner checkerboard corners using SB detector when available, with classic fallback."""
    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0) + getattr(cv2, "CALIB_CB_ACCURACY", 0)
        found_sb, corners_sb = cv2.findChessboardCornersSB(gray, pattern_size, sb_flags)
        if found_sb and corners_sb is not None:
            oriented = _orient_corners_top_left(corners_sb, pattern_size)
            return True, oriented

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FILTER_QUADS
    )
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found or corners is None:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    oriented = _orient_corners_top_left(refined, pattern_size)
    return True, oriented


def probe_single_camera(idx: int, timeout_sec: float = 1.5) -> Optional[int]:
    """Probe a single camera index with a timeout so a hung V4L2 driver cannot freeze startup."""
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
    """Probe candidate video indices with per-device timeouts."""
    candidates = list_video_device_indices(max_devices)
    available = []
    for idx in candidates:
        found = probe_single_camera(idx, timeout_sec=1.5)
        if found is not None:
            available.append(found)
    return available if available else candidates[:2]


def _draw_pose_axes_fallback(
    img: np.ndarray,
    mtx: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    length: float,
):
    """Draw X (red), Y (green), Z (blue) axes with labels onto an image."""
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, length, 3)
    else:
        axis_pts = np.float32(
            [[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, -length]]
        )
        projected, _ = cv2.projectPoints(axis_pts, rvec, tvec, mtx, dist)
        pts = projected.reshape(-1, 2).astype(int)
        origin = tuple(pts[0])
        cv2.line(img, origin, tuple(pts[1]), (0, 0, 255), 3, cv2.LINE_AA)   # X: Red
        cv2.line(img, origin, tuple(pts[2]), (0, 255, 0), 3, cv2.LINE_AA)   # Y: Green
        cv2.line(img, origin, tuple(pts[3]), (255, 0, 0), 3, cv2.LINE_AA)   # Z: Blue

    label_pts = np.float32(
        [[length * 1.15, 0, 0], [0, length * 1.15, 0], [0, 0, -length * 1.15]]
    )
    proj_lbl, _ = cv2.projectPoints(label_pts, rvec, tvec, mtx, dist)
    lbl_xy = proj_lbl.reshape(-1, 2).astype(int)
    cv2.putText(img, "X", tuple(lbl_xy[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Y", tuple(lbl_xy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Z", tuple(lbl_xy[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)


class CalibrationReviewWindow:
    """Review window displaying all calibration pairs with corners, axes, and deletion."""

    def __init__(self, app: "StereoCameraApp"):
        self.app = app
        self.win = tk.Toplevel(app.root)
        self.win.title("Calibration Pair Review — Corners & Coordinate Axes")
        self.win.geometry("1220x740")
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self.selected_idx = 0
        self._photo_left: Optional[ImageTk.PhotoImage] = None
        self._photo_right: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self.refresh_list(keep_index=0)

    def _build_ui(self):
        top = ttk.Frame(self.win, padding=8)
        top.pack(fill=tk.X)

        self.info_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.info_var, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)

        ttk.Button(
            top,
            text="Remove Selected Pair & Recalibrate",
            command=self.remove_selected_pair,
        ).pack(side=tk.RIGHT, padx=4)

        ttk.Button(top, text="Close Review", command=self.close).pack(side=tk.RIGHT, padx=4)

        legend = ttk.Label(
            top,
            text="Axes: Red=X, Green=Y, Blue=Z | Cyan ring = Top-Left Origin (idx 0)",
        )
        legend.pack(side=tk.RIGHT, padx=16)

        body = ttk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        list_frame = ttk.LabelFrame(body, text="Captured Pairs", padding=6)
        body.add(list_frame, weight=1)

        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.pair_listbox = tk.Listbox(
            list_frame,
            width=36,
            exportselection=False,
            yscrollcommand=list_scroll.set,
            font=("Consolas", 9),
        )
        list_scroll.config(command=self.pair_listbox.yview)
        self.pair_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.pair_listbox.bind("<<ListboxSelect>>", self._on_select)

        preview_frame = ttk.Frame(body)
        preview_frame.columnconfigure(0, weight=1, uniform="review_cams")
        preview_frame.columnconfigure(1, weight=1, uniform="review_cams")
        preview_frame.rowconfigure(0, weight=1)
        body.add(preview_frame, weight=4)

        left_box = ttk.LabelFrame(preview_frame, text="Left Camera (Corners + Axes)", padding=4)
        left_box.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left_box.grid_propagate(False)
        left_box.columnconfigure(0, weight=1)
        left_box.rowconfigure(0, weight=1)
        self.left_img_label = tk.Label(left_box, bg="black", anchor="center")
        self.left_img_label.grid(row=0, column=0, sticky="nsew")

        right_box = ttk.LabelFrame(preview_frame, text="Right Camera (Corners + Axes)", padding=4)
        right_box.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right_box.grid_propagate(False)
        right_box.columnconfigure(0, weight=1)
        right_box.rowconfigure(0, weight=1)
        self.right_img_label = tk.Label(right_box, bg="black", anchor="center")
        self.right_img_label.grid(row=0, column=0, sticky="nsew")

        self.win.bind("<Configure>", lambda _e: self._render_selected_pair())

    def refresh_list(self, keep_index: int = 0):
        self.pair_listbox.delete(0, tk.END)
        records = self.app.calib_records
        for i, rec in enumerate(records):
            err_str = f"{rec['pair_rms_px']:.3f} px" if rec.get("pair_rms_px") is not None else "--"
            name = os.path.basename(rec["left_path"])
            self.pair_listbox.insert(tk.END, f"#{i + 1:02d} | {name} | err: {err_str}")

        if not records:
            self.info_var.set("No calibration pairs remaining.")
            self.left_img_label.configure(image="")
            self.right_img_label.configure(image="")
            return

        self.selected_idx = max(0, min(keep_index, len(records) - 1))
        self.pair_listbox.selection_set(self.selected_idx)
        self.pair_listbox.see(self.selected_idx)
        self._render_selected_pair()

    def _on_select(self, _event=None):
        sel = self.pair_listbox.curselection()
        if not sel:
            return
        self.selected_idx = int(sel[0])
        self._render_selected_pair()

    def _annotate_view(
        self,
        img_bgr: np.ndarray,
        objp: np.ndarray,
        corners: np.ndarray,
        pattern_size: Tuple[int, int],
        sq_size_mm: float,
        mtx: Optional[np.ndarray],
        dist: Optional[np.ndarray],
        title: str,
    ) -> np.ndarray:
        vis = img_bgr.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners, True)

        c0 = tuple(corners[0].reshape(2).astype(int))
        cv2.circle(vis, c0, 9, (255, 255, 0), 2, cv2.LINE_AA)

        if mtx is not None and dist is not None:
            ok, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
            if ok:
                axis_len = sq_size_mm * 3.0
                _draw_pose_axes_fallback(vis, mtx, dist, rvec, tvec, axis_len)

        cv2.putText(vis, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        return vis

    def _to_photo(self, bgr: np.ndarray, container: tk.Widget) -> ImageTk.PhotoImage:
        pw = container.master.winfo_width() if container.master else 520
        ph = container.master.winfo_height() if container.master else 460
        tw = max(200, pw - 14)
        th = max(160, ph - 14)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((tw, th), Image.Resampling.BILINEAR)
        return ImageTk.PhotoImage(pil_img)

    def _render_selected_pair(self):
        records = self.app.calib_records
        if not records or self.selected_idx >= len(records):
            return

        rec = records[self.selected_idx]
        params = self.app.calib_params
        mtx_l = np.array(params["camera_matrix_left"], dtype=np.float64) if params else None
        dist_l = np.array(params["dist_coeffs_left"], dtype=np.float64) if params else None
        mtx_r = np.array(params["camera_matrix_right"], dtype=np.float64) if params else None
        dist_r = np.array(params["dist_coeffs_right"], dtype=np.float64) if params else None

        img_l = rec.get("frame_left")
        if img_l is None and os.path.exists(rec["left_path"]):
            img_l = cv2.imread(rec["left_path"])
        img_r = rec.get("frame_right")
        if img_r is None and os.path.exists(rec["right_path"]):
            img_r = cv2.imread(rec["right_path"])

        if img_l is None or img_r is None:
            return

        pattern_size = rec["pattern_size"]
        sq_mm = rec["square_size_mm"]
        err_txt = f" | Pair RMS: {rec['pair_rms_px']:.4f} px" if rec.get("pair_rms_px") is not None else ""
        self.info_var.set(
            f"Pair {self.selected_idx + 1} of {len(records)} ({os.path.basename(rec['left_path'])}){err_txt}"
        )

        vis_l = self._annotate_view(
            img_l, rec["objp"], rec["corners_left"], pattern_size, sq_mm, mtx_l, dist_l, "LEFT"
        )
        vis_r = self._annotate_view(
            img_r, rec["objp"], rec["corners_right"], pattern_size, sq_mm, mtx_r, dist_r, "RIGHT"
        )

        self._photo_left = self._to_photo(vis_l, self.left_img_label)
        self._photo_right = self._to_photo(vis_r, self.right_img_label)
        self.left_img_label.configure(image=self._photo_left)
        self.right_img_label.configure(image=self._photo_right)

    def remove_selected_pair(self):
        records = self.app.calib_records
        if not records:
            return
        if len(records) <= 4:
            messagebox.showwarning(
                "Minimum Pairs Required",
                "At least 4 calibration pairs are required to compute stereo calibration.",
                parent=self.win,
            )
            return

        idx = self.selected_idx
        self.app.remove_calibration_pair_and_recompute(idx)
        self.refresh_list(keep_index=idx)

    def close(self):
        self.app.review_window = None
        self.win.destroy()


class StereoCameraApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Stereo Camera & Calibration Studio")
        self.root.geometry("1360x860")

        self.settings = self.load_settings()

        # Camera & thread state
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

        # Calibration & rectification state
        self.calibration_mode = False
        self._loading_images = False
        self.calib_records: List[dict] = []
        self.image_size: Optional[Tuple[int, int]] = None  # (width, height)
        self.captured_seq = 0
        self.review_window: Optional[CalibrationReviewWindow] = None

        self.rectification_ready = False
        self.map_l1 = self.map_l2 = None
        self.map_r1 = self.map_r2 = None
        self.calib_params: Optional[dict] = self.load_existing_calibration()

        self._build_ui()
        self._apply_settings_to_widgets()

        initial_candidates = list_video_device_indices()
        self.left_cam_combo["values"] = initial_candidates
        self.right_cam_combo["values"] = initial_candidates
        self.status_var.set("Scanning cameras & loading saved calibration pairs...")

        # Start capture loop & UI refresh loop without blocking initial render
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(50, self._async_initial_camera_setup)
        self.root.after(80, lambda: self._start_loading_existing_images(open_review_when_done=False))
        self._update_ui_loop()

    # ------------------------------------------------------------------
    # Settings & Calibration Persistence
    # ------------------------------------------------------------------
    def load_settings(self) -> dict:
        data = dict(DEFAULT_SETTINGS)
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                data.update(loaded)
            except Exception as exc:
                print(f"Warning: failed to load {SETTINGS_FILE}: {exc}")
        return data

    def save_settings(self):
        self._sync_widgets_to_settings()
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
            self.status_var.set(f"Settings saved to {SETTINGS_FILE}")
        except Exception as exc:
            messagebox.showerror("Save Error", f"Could not save settings:\n{exc}")

    def load_existing_calibration(self) -> Optional[dict]:
        if os.path.exists(CALIB_PARAMS_FILE):
            try:
                with open(CALIB_PARAMS_FILE, "r", encoding="utf-8") as f:
                    params = json.load(f)
                self._prepare_rectification_maps(params)
                return params
            except Exception:
                return None
        return None

    def _start_loading_existing_images(self, open_review_when_done: bool = False, on_done=None):
        """Load saved stereo calibration image pairs in a background thread so the UI never freezes."""
        if self._loading_images:
            return
        if not os.path.isdir(CALIB_IMAGES_DIR):
            if on_done:
                on_done()
            return

        left_files = sorted(glob.glob(os.path.join(CALIB_IMAGES_DIR, "left_*.png")))
        if not left_files:
            if on_done:
                on_done()
            return

        self._loading_images = True
        cols = int(self.cols_var.get())
        rows = int(self.rows_var.get())
        sq_mm = float(self.sq_size_var.get())
        pattern_size = (cols, rows)
        objp_template = np.zeros((rows * cols, 3), np.float32)
        objp_template[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq_mm

        def _worker():
            loaded_records: List[dict] = []
            detected_size: Optional[Tuple[int, int]] = None
            max_seq = 0

            for idx_num, lp in enumerate(left_files, start=1):
                if not self.running:
                    break
                base = os.path.basename(lp)
                rp = os.path.join(CALIB_IMAGES_DIR, "right_" + base[len("left_"):])
                if not os.path.exists(rp):
                    continue

                self.root.after(
                    0,
                    lambda c=idx_num, n=len(left_files), name=base: self.status_var.set(
                        f"Detecting corners on saved pair {c}/{n} ({name})..."
                    ),
                )

                frame_l = cv2.imread(lp)
                frame_r = cv2.imread(rp)
                if frame_l is None or frame_r is None:
                    continue
                if frame_l.shape[:2] != frame_r.shape[:2]:
                    continue

                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

                found_l, corners_l = _detect_checkerboard_corners(gray_l, pattern_size)
                found_r, corners_r = _detect_checkerboard_corners(gray_r, pattern_size)
                if not (found_l and found_r):
                    continue

                assert corners_l is not None and corners_r is not None
                detected_size = (gray_l.shape[1], gray_l.shape[0])

                m_num = re.search(r"(\d+)", base)
                if m_num:
                    max_seq = max(max_seq, int(m_num.group(1)))
                else:
                    max_seq += 1

                loaded_records.append(
                    {
                        "left_path": lp,
                        "right_path": rp,
                        "frame_left": frame_l,
                        "frame_right": frame_r,
                        "objp": objp_template.copy(),
                        "corners_left": corners_l,
                        "corners_right": corners_r,
                        "pattern_size": pattern_size,
                        "square_size_mm": sq_mm,
                        "pair_rms_px": None,
                    }
                )

            if not self.running:
                return
            self.root.after(
                0,
                lambda recs=loaded_records, sz=detected_size, seq=max_seq: self._on_existing_images_loaded(
                    recs, sz, seq, open_review_when_done, on_done
                ),
            )

        threading.Thread(target=_worker, daemon=True).start()

    def _on_existing_images_loaded(
        self,
        records: List[dict],
        detected_size: Optional[Tuple[int, int]],
        max_seq: int,
        open_review_when_done: bool,
        on_done,
    ):
        self._loading_images = False
        if not self.running:
            return

        self.calib_records = records
        self.captured_seq = max_seq
        if detected_size is not None:
            self.image_size = detected_size

        total = len(self.calib_records)
        if total > 0:
            self.review_pairs_btn.config(state=tk.NORMAL)
            if self.calib_params is not None and not self.calibration_mode:
                self._compute_per_pair_errors_from_params(self.calib_params)
                self._update_certainty_indicator(
                    score=float(self.calib_params.get("certainty_percent", 0.0)),
                    note=f"Loaded {total} existing pairs (RMS={self.calib_params.get('rms_error_px', 0.0):.4f} px)",
                )
            else:
                sample_certainty = min(100.0, (total / 10.0) * 80.0)
                self._update_certainty_indicator(
                    score=sample_certainty,
                    note=f"Loaded {total} existing pairs from {CALIB_IMAGES_DIR}",
                )
            self.status_var.set(f"Loaded {total} calibration image pair(s) from {CALIB_IMAGES_DIR}")
        else:
            self.status_var.set("Ready.")

        if open_review_when_done and total > 0:
            self._open_review_window_direct()

        if on_done:
            on_done()

    def _compute_per_pair_errors_from_params(self, params: dict):
        """Estimate per-pair RMS using solvePnP and existing intrinsics without re-solving stereo."""
        try:
            mtx_l = np.array(params["camera_matrix_left"], dtype=np.float64)
            dist_l = np.array(params["dist_coeffs_left"], dtype=np.float64)
            mtx_r = np.array(params["camera_matrix_right"], dtype=np.float64)
            dist_r = np.array(params["dist_coeffs_right"], dtype=np.float64)
        except Exception:
            return

        for rec in self.calib_records:
            objp = rec["objp"]
            cl = rec["corners_left"]
            cr = rec["corners_right"]
            ok_l, rv_l, tv_l = cv2.solvePnP(objp, cl, mtx_l, dist_l)
            ok_r, rv_r, tv_r = cv2.solvePnP(objp, cr, mtx_r, dist_r)
            if ok_l and ok_r:
                proj_l, _ = cv2.projectPoints(objp, rv_l, tv_l, mtx_l, dist_l)
                proj_r, _ = cv2.projectPoints(objp, rv_r, tv_r, mtx_r, dist_r)
                err_l = cv2.norm(cl, proj_l, cv2.NORM_L2) / np.sqrt(len(objp))
                err_r = cv2.norm(cr, proj_r, cv2.NORM_L2) / np.sqrt(len(objp))
                rec["pair_rms_px"] = float(np.sqrt(0.5 * (err_l ** 2 + err_r ** 2)))

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.status_var = tk.StringVar(value="Initializing...")

        top_bar = ttk.Frame(self.root, padding=8)
        top_bar.pack(fill=tk.X)

        ttk.Label(top_bar, text="Left (Main) Cam:").pack(side=tk.LEFT, padx=(0, 4))
        self.left_cam_var = tk.IntVar()
        self.left_cam_combo = ttk.Combobox(
            top_bar, textvariable=self.left_cam_var, values=[], width=5, state="readonly"
        )
        self.left_cam_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.left_cam_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_camera_selection_changed())

        ttk.Label(top_bar, text="Right Cam:").pack(side=tk.LEFT, padx=(0, 4))
        self.right_cam_var = tk.IntVar()
        self.right_cam_combo = ttk.Combobox(
            top_bar, textvariable=self.right_cam_var, values=[], width=5, state="readonly"
        )
        self.right_cam_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.right_cam_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_camera_selection_changed())

        ttk.Button(top_bar, text="Rescan Cams", command=self._rescan_cameras).pack(side=tk.LEFT, padx=(0, 18))

        self.calib_btn = ttk.Button(
            top_bar, text="Stereo Calibrate", command=self.toggle_calibration_mode
        )
        self.calib_btn.pack(side=tk.LEFT, padx=4)

        self.take_img_btn = ttk.Button(
            top_bar, text="Take Image", command=self.capture_calibration_pair, state=tk.DISABLED
        )
        self.take_img_btn.pack(side=tk.LEFT, padx=4)

        self.finish_calib_btn = ttk.Button(
            top_bar, text="Finish Calibration", command=self.finish_calibration, state=tk.DISABLED
        )
        self.finish_calib_btn.pack(side=tk.LEFT, padx=4)

        self.review_pairs_btn = ttk.Button(
            top_bar, text="Review Pairs", command=self.open_review_window, state=tk.DISABLED
        )
        self.review_pairs_btn.pack(side=tk.LEFT, padx=4)

        self.show_rectified_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top_bar, text="Show Rectified Preview", variable=self.show_rectified_var
        ).pack(side=tk.RIGHT, padx=8)

        settings_frame = ttk.LabelFrame(self.root, text="Camera & Checkerboard Settings", padding=8)
        settings_frame.pack(fill=tk.X, padx=8, pady=(0, 6))

        self.res_var = tk.StringVar()
        ttk.Label(settings_frame, text="Resolution:").grid(row=0, column=0, sticky=tk.W, padx=4)
        res_combo = ttk.Combobox(settings_frame, textvariable=self.res_var, values=RESOLUTIONS, width=10, state="readonly")
        res_combo.grid(row=0, column=1, padx=4)
        res_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_camera_properties())

        self.fps_var = tk.IntVar()
        ttk.Label(settings_frame, text="FPS:").grid(row=0, column=2, sticky=tk.W, padx=4)
        fps_spin = ttk.Spinbox(settings_frame, from_=5, to=120, textvariable=self.fps_var, width=6, command=self._apply_camera_properties)
        fps_spin.grid(row=0, column=3, padx=4)

        self.autofocus_var = tk.BooleanVar()
        ttk.Checkbutton(
            settings_frame, text="Autofocus", variable=self.autofocus_var, command=self._apply_camera_properties
        ).grid(row=0, column=4, padx=6)

        self.focus_var = tk.IntVar()
        ttk.Label(settings_frame, text="Focus:").grid(row=0, column=5, sticky=tk.W, padx=4)
        ttk.Scale(
            settings_frame, from_=0, to=255, variable=self.focus_var, orient=tk.HORIZONTAL, length=100,
            command=lambda _v: self._apply_camera_properties()
        ).grid(row=0, column=6, padx=4)

        self.auto_exposure_var = tk.BooleanVar()
        ttk.Checkbutton(
            settings_frame, text="Auto Exposure", variable=self.auto_exposure_var, command=self._apply_camera_properties
        ).grid(row=0, column=7, padx=6)

        self.exposure_var = tk.IntVar()
        exp_sub = ttk.Frame(settings_frame)
        exp_sub.grid(row=0, column=8, sticky=tk.W, padx=4)
        ttk.Label(exp_sub, text="Exp:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Scale(
            exp_sub, from_=-13, to=-1, variable=self.exposure_var, orient=tk.HORIZONTAL, length=90,
            command=lambda _v: self._apply_camera_properties()
        ).pack(side=tk.LEFT)

        self.brightness_var = tk.IntVar()
        ttk.Label(settings_frame, text="Brightness:").grid(row=1, column=0, sticky=tk.W, padx=4, pady=(6, 0))
        ttk.Scale(
            settings_frame, from_=0, to=255, variable=self.brightness_var, orient=tk.HORIZONTAL, length=100,
            command=lambda _v: self._apply_camera_properties()
        ).grid(row=1, column=1, padx=4, pady=(6, 0))

        self.contrast_var = tk.IntVar()
        ttk.Label(settings_frame, text="Contrast:").grid(row=1, column=2, sticky=tk.W, padx=4, pady=(6, 0))
        ttk.Scale(
            settings_frame, from_=0, to=255, variable=self.contrast_var, orient=tk.HORIZONTAL, length=100,
            command=lambda _v: self._apply_camera_properties()
        ).grid(row=1, column=3, padx=4, pady=(6, 0))

        self.cols_var = tk.IntVar()
        self.rows_var = tk.IntVar()
        self.sq_size_var = tk.DoubleVar()
        ttk.Label(settings_frame, text="Inner Corners (Cols×Rows):").grid(row=1, column=4, sticky=tk.W, padx=4, pady=(6, 0))
        board_sub = ttk.Frame(settings_frame)
        board_sub.grid(row=1, column=5, columnspan=2, sticky=tk.W, pady=(6, 0))
        ttk.Spinbox(board_sub, from_=3, to=20, textvariable=self.cols_var, width=4).pack(side=tk.LEFT)
        ttk.Label(board_sub, text="×").pack(side=tk.LEFT, padx=2)
        ttk.Spinbox(board_sub, from_=3, to=20, textvariable=self.rows_var, width=4).pack(side=tk.LEFT)
        ttk.Label(board_sub, text="  Square(mm):").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Entry(board_sub, textvariable=self.sq_size_var, width=6).pack(side=tk.LEFT)

        ttk.Button(settings_frame, text="Apply Props", command=self._apply_camera_properties).grid(
            row=1, column=7, padx=4, pady=(6, 0)
        )
        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).grid(
            row=1, column=8, padx=4, pady=(6, 0)
        )

        certainty_frame = ttk.LabelFrame(self.root, text="Calibration Certainty & Status", padding=6)
        certainty_frame.pack(fill=tk.X, padx=8, pady=(0, 6))

        self.certainty_bar = ttk.Progressbar(certainty_frame, orient=tk.HORIZONTAL, length=240, maximum=100)
        self.certainty_bar.pack(side=tk.LEFT, padx=(0, 10))

        self.certainty_label_var = tk.StringVar(value="Certainty: -- (0 pairs captured)")
        ttk.Label(certainty_frame, textvariable=self.certainty_label_var).pack(side=tk.LEFT)

        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))

        video_container = ttk.Frame(main_pane)
        video_container.columnconfigure(0, weight=1, uniform="cams")
        video_container.columnconfigure(1, weight=1, uniform="cams")
        video_container.rowconfigure(0, weight=1)
        main_pane.add(video_container, weight=4)

        left_box = ttk.LabelFrame(video_container, text="Left (Main) Camera", padding=4)
        left_box.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left_box.grid_propagate(False)
        left_box.columnconfigure(0, weight=1)
        left_box.rowconfigure(0, weight=1)
        self.left_canvas = tk.Label(left_box, bg="black", anchor="center")
        self.left_canvas.grid(row=0, column=0, sticky="nsew")

        right_box = ttk.LabelFrame(video_container, text="Right Camera", padding=4)
        right_box.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right_box.grid_propagate(False)
        right_box.columnconfigure(0, weight=1)
        right_box.rowconfigure(0, weight=1)
        self.right_canvas = tk.Label(right_box, bg="black", anchor="center")
        self.right_canvas.grid(row=0, column=0, sticky="nsew")

        params_box = ttk.LabelFrame(main_pane, text="Stereo Calibration Parameters", padding=6)
        main_pane.add(params_box, weight=1)

        self.params_text = tk.Text(params_box, width=36, wrap=tk.WORD, font=("Consolas", 9))
        self.params_text.pack(fill=tk.BOTH, expand=True)
        if self.calib_params:
            self._display_calibration_params(self.calib_params)

        status_bar = ttk.Frame(self.root, padding=(8, 4))
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(status_bar, textvariable=self.status_var).pack(side=tk.LEFT)

    def _apply_settings_to_widgets(self):
        self.left_cam_var.set(self.settings["left_camera_index"])
        self.right_cam_var.set(self.settings["right_camera_index"])
        self.res_var.set(self.settings["resolution"])
        self.fps_var.set(self.settings["fps"])
        self.autofocus_var.set(self.settings["autofocus"])
        self.focus_var.set(self.settings["focus"])
        self.auto_exposure_var.set(self.settings["auto_exposure"])
        self.exposure_var.set(self.settings["exposure"])
        self.brightness_var.set(self.settings["brightness"])
        self.contrast_var.set(self.settings["contrast"])
        self.cols_var.set(self.settings["checkerboard_cols"])
        self.rows_var.set(self.settings["checkerboard_rows"])
        self.sq_size_var.set(self.settings["square_size_mm"])

    def _sync_widgets_to_settings(self):
        self.settings["left_camera_index"] = int(self.left_cam_var.get())
        self.settings["right_camera_index"] = int(self.right_cam_var.get())
        self.settings["resolution"] = self.res_var.get()
        self.settings["fps"] = int(self.fps_var.get())
        self.settings["autofocus"] = bool(self.autofocus_var.get())
        self.settings["focus"] = int(self.focus_var.get())
        self.settings["auto_exposure"] = bool(self.auto_exposure_var.get())
        self.settings["exposure"] = int(self.exposure_var.get())
        self.settings["brightness"] = int(self.brightness_var.get())
        self.settings["contrast"] = int(self.contrast_var.get())
        self.settings["checkerboard_cols"] = int(self.cols_var.get())
        self.settings["checkerboard_rows"] = int(self.rows_var.get())
        self.settings["square_size_mm"] = float(self.sq_size_var.get())

    # ------------------------------------------------------------------
    # Camera Management & Auto-Reconnect
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

    def _on_camera_selection_changed(self):
        left_idx = int(self.left_cam_var.get())
        right_idx = int(self.right_cam_var.get())
        if left_idx == right_idx:
            messagebox.showwarning("Same Camera", "Left and Right cameras should use different indices.")
        self._open_selected_cameras()

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

    def _open_selected_cameras(self):
        self._release_cameras()
        left_idx = int(self.left_cam_var.get())
        right_idx = int(self.right_cam_var.get())

        cap_l = _open_capture(left_idx)
        cap_r = _open_capture(right_idx)

        with self.lock:
            self.cap_left = cap_l
            self.cap_right = cap_r
            self.left_online = bool(cap_l and cap_l.isOpened())
            self.right_online = bool(cap_r and cap_r.isOpened())
            self._left_fail_count = 0
            self._right_fail_count = 0

        self._apply_camera_properties()
        self.status_var.set(f"Opened Left Cam #{left_idx} and Right Cam #{right_idx}")

    def _get_target_res_and_fps(self) -> Tuple[int, int, int]:
        res_str = self.res_var.get()
        try:
            w, h = [int(x) for x in res_str.split("x")]
        except ValueError:
            w, h = 1280, 720
        return w, h, int(self.fps_var.get())

    def _apply_single_cam_props(self, cap: Optional[cv2.VideoCapture], width: int, height: int, fps: int):
        if cap is None or not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.autofocus_var.get() else 0)
        if not self.autofocus_var.get():
            cap.set(cv2.CAP_PROP_FOCUS, int(self.focus_var.get()))

        auto_exp = bool(self.auto_exposure_var.get())
        _apply_auto_exposure(cap, auto_exp)
        if not auto_exp:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure_var.get()))

        cap.set(cv2.CAP_PROP_BRIGHTNESS, float(self.brightness_var.get()))
        cap.set(cv2.CAP_PROP_CONTRAST, float(self.contrast_var.get()))

    def _apply_camera_properties(self):
        w, h, fps = self._get_target_res_and_fps()
        with self.lock:
            self._apply_single_cam_props(self.cap_left, w, h, fps)
            self._apply_single_cam_props(self.cap_right, w, h, fps)

        exp_desc = "AUTO" if self.auto_exposure_var.get() else f"MANUAL ({self.exposure_var.get()})"
        self.status_var.set(
            f"Applied {w}x{h} @ {fps} FPS | Focus={self.focus_var.get()} (AF={self.autofocus_var.get()}) | Exp={exp_desc}"
        )

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

    # ------------------------------------------------------------------
    # Stereo Calibration Mode, Capture & Pair Review
    # ------------------------------------------------------------------
    def toggle_calibration_mode(self):
        if self._loading_images:
            self.status_var.set("Still loading saved calibration images, please wait...")
            return

        if not self.calibration_mode:
            os.makedirs(CALIB_IMAGES_DIR, exist_ok=True)
            self.calibration_mode = True
            self.calib_btn.config(text="Cancel Calibration")
            self.take_img_btn.config(state=tk.NORMAL)
            self.finish_calib_btn.config(state=tk.NORMAL)
            self._update_certainty_indicator(score=0.0, note="Loading existing pairs & waiting for captures")
            self.status_var.set(f"Stereo calibration mode ON — saving pairs to {CALIB_IMAGES_DIR}")

            def _after_load():
                if not self.calibration_mode:
                    return
                count = len(self.calib_records)
                if count > 0:
                    sample_certainty = min(100.0, (count / 10.0) * 80.0)
                    self._update_certainty_indicator(
                        score=sample_certainty,
                        note=f"{count} existing pairs ready — capture more or click Finish Calibration",
                    )
                else:
                    self._update_certainty_indicator(score=0.0, note="Move checkerboard in front of both cameras")

            self._start_loading_existing_images(open_review_when_done=False, on_done=_after_load)
        else:
            self.calibration_mode = False
            self.calib_btn.config(text="Stereo Calibrate")
            self.take_img_btn.config(state=tk.DISABLED)
            self.finish_calib_btn.config(state=tk.DISABLED)
            self.status_var.set("Stereo calibration mode cancelled.")

    def _checkerboard_object_points(self) -> np.ndarray:
        cols = int(self.cols_var.get())
        rows = int(self.rows_var.get())
        sq_mm = float(self.sq_size_var.get())
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq_mm
        return objp

    def capture_calibration_pair(self):
        with self.lock:
            frame_l = None if self.latest_left is None else self.latest_left.copy()
            frame_r = None if self.latest_right is None else self.latest_right.copy()

        if frame_l is None or frame_r is None:
            messagebox.showwarning("No Frame", "Waiting for frames from both cameras.")
            return

        cols = int(self.cols_var.get())
        rows = int(self.rows_var.get())
        sq_mm = float(self.sq_size_var.get())
        pattern_size = (cols, rows)

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = _detect_checkerboard_corners(gray_l, pattern_size)
        found_r, corners_r = _detect_checkerboard_corners(gray_r, pattern_size)

        if not (found_l and found_r):
            messagebox.showinfo(
                "Checkerboard Not Detected",
                f"Both cameras must see all {cols}×{rows} inner corners clearly "
                f"(e.g., a 9×7 square grid has 8×6 inner corners).\n"
                f"Detected — Left: {found_l}, Right: {found_r}",
            )
            return

        assert corners_l is not None and corners_r is not None
        self.image_size = (gray_l.shape[1], gray_l.shape[0])
        self.captured_seq += 1

        idx_str = f"{self.captured_seq:03d}"
        left_path = os.path.join(CALIB_IMAGES_DIR, f"left_{idx_str}.png")
        right_path = os.path.join(CALIB_IMAGES_DIR, f"right_{idx_str}.png")
        cv2.imwrite(left_path, frame_l)
        cv2.imwrite(right_path, frame_r)

        self.calib_records.append(
            {
                "left_path": left_path,
                "right_path": right_path,
                "frame_left": frame_l,
                "frame_right": frame_r,
                "objp": self._checkerboard_object_points(),
                "corners_left": corners_l,
                "corners_right": corners_r,
                "pattern_size": pattern_size,
                "square_size_mm": sq_mm,
                "pair_rms_px": None,
            }
        )

        count = len(self.calib_records)
        self.review_pairs_btn.config(state=tk.NORMAL)
        sample_certainty = min(100.0, (count / 10.0) * 80.0)
        self._update_certainty_indicator(
            score=sample_certainty,
            note=f"{count} valid pairs saved (recommend >= 10 varied angles)",
        )
        self.status_var.set(f"Saved calibration pair #{self.captured_seq} ({count} active) to {CALIB_IMAGES_DIR}")

    def _update_certainty_indicator(self, score: float, note: str):
        clamped = max(0.0, min(100.0, score))
        self.certainty_bar["value"] = clamped
        self.certainty_label_var.set(f"Certainty: {clamped:.1f}% — {note}")

    def _run_stereo_calibration_solve(self) -> bool:
        if len(self.calib_records) < 4:
            messagebox.showwarning(
                "Not Enough Samples",
                f"Capture at least 4 checkerboard pairs first (currently {len(self.calib_records)}).",
            )
            return False
        if self.image_size is None:
            messagebox.showerror("Error", "Missing image dimensions.")
            return False

        self.status_var.set("Computing stereo calibration...")
        self.root.update_idletasks()

        objpoints = [r["objp"] for r in self.calib_records]
        imgpoints_left = [r["corners_left"] for r in self.calib_records]
        imgpoints_right = [r["corners_right"] for r in self.calib_records]

        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        _ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
            objpoints, imgpoints_left, self.image_size, None, None
        )
        _ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
            objpoints, imgpoints_right, self.image_size, None, None
        )

        for i, rec in enumerate(self.calib_records):
            proj_l, _ = cv2.projectPoints(objpoints[i], rvecs_l[i], tvecs_l[i], mtx_l, dist_l)
            proj_r, _ = cv2.projectPoints(objpoints[i], rvecs_r[i], tvecs_r[i], mtx_r, dist_r)
            err_l = cv2.norm(imgpoints_left[i], proj_l, cv2.NORM_L2) / np.sqrt(len(objpoints[i]))
            err_r = cv2.norm(imgpoints_right[i], proj_r, cv2.NORM_L2) / np.sqrt(len(objpoints[i]))
            rec["pair_rms_px"] = float(np.sqrt(0.5 * (err_l ** 2 + err_r ** 2)))

        rms, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            mtx_l,
            dist_l,
            mtx_r,
            dist_r,
            self.image_size,
            criteria=criteria,
            flags=flags,
        )

        sample_factor = min(1.0, len(self.calib_records) / 10.0)
        error_factor = max(0.0, 1.0 - (rms / 1.5))
        certainty_score = 100.0 * (0.35 * sample_factor + 0.65 * error_factor)
        self._update_certainty_indicator(
            score=certainty_score,
            note=f"RMS reprojection error: {rms:.4f} px across {len(self.calib_records)} pairs",
        )

        baseline_mm = float(np.linalg.norm(T))
        params_dict = {
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
            "num_pairs": len(self.calib_records),
            "rms_error_px": float(rms),
            "certainty_percent": float(certainty_score),
            "baseline_mm": baseline_mm,
            "camera_matrix_left": mtx_l.tolist(),
            "dist_coeffs_left": dist_l.tolist(),
            "camera_matrix_right": mtx_r.tolist(),
            "dist_coeffs_right": dist_r.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
            "E": E.tolist(),
            "F": F.tolist(),
        }

        with open(CALIB_PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(params_dict, f, indent=2)

        self.calib_params = params_dict
        self._prepare_rectification_maps(params_dict)
        self._display_calibration_params(params_dict)
        self.status_var.set(
            f"Calibration complete! Saved to {CALIB_PARAMS_FILE} (RMS={rms:.4f} px, {len(self.calib_records)} pairs)"
        )
        return True

    def finish_calibration(self):
        if self._loading_images:
            messagebox.showinfo("Loading Images", "Please wait for saved calibration images to finish loading.")
            return
        if not self._run_stereo_calibration_solve():
            return

        self.calibration_mode = False
        self.calib_btn.config(text="Stereo Calibrate")
        self.take_img_btn.config(state=tk.DISABLED)
        self.finish_calib_btn.config(state=tk.DISABLED)
        self.review_pairs_btn.config(state=tk.NORMAL)
        self.show_rectified_var.set(True)
        self._open_review_window_direct()

    def _open_review_window_direct(self):
        if not self.calib_records:
            return
        if self.review_window is not None and self.review_window.win.winfo_exists():
            self.review_window.win.lift()
            self.review_window.refresh_list(keep_index=self.review_window.selected_idx)
            return
        self.review_window = CalibrationReviewWindow(self)

    def open_review_window(self):
        if self._loading_images:
            self.status_var.set("Still loading saved calibration images...")
            return
        if not self.calib_records:
            self._start_loading_existing_images(
                open_review_when_done=True,
                on_done=lambda: (
                    messagebox.showinfo(
                        "No Calibration Pairs",
                        f"No valid calibration image pairs found in {CALIB_IMAGES_DIR}.",
                    )
                    if not self.calib_records
                    else None
                ),
            )
            return
        self._open_review_window_direct()

    def remove_calibration_pair_and_recompute(self, idx: int):
        if idx < 0 or idx >= len(self.calib_records):
            return
        removed = self.calib_records.pop(idx)
        for path_key in ("left_path", "right_path"):
            p = removed.get(path_key)
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

        self._run_stereo_calibration_solve()

    def _prepare_rectification_maps(self, params: dict):
        w = int(params["image_width"])
        h = int(params["image_height"])
        mtx_l = np.array(params["camera_matrix_left"], dtype=np.float64)
        dist_l = np.array(params["dist_coeffs_left"], dtype=np.float64)
        mtx_r = np.array(params["camera_matrix_right"], dtype=np.float64)
        dist_r = np.array(params["dist_coeffs_right"], dtype=np.float64)
        R = np.array(params["R"], dtype=np.float64)
        T = np.array(params["T"], dtype=np.float64)

        R1, R2, P1, P2, _Q, _, _ = cv2.stereoRectify(
            mtx_l, dist_l, mtx_r, dist_r, (w, h), R, T, alpha=0
        )
        self.map_l1, self.map_l2 = cv2.initUndistortRectifyMap(
            mtx_l, dist_l, R1, P1, (w, h), cv2.CV_16SC2
        )
        self.map_r1, self.map_r2 = cv2.initUndistortRectifyMap(
            mtx_r, dist_r, R2, P2, (w, h), cv2.CV_16SC2
        )
        self.rectification_ready = True

    def _display_calibration_params(self, params: dict):
        summary = {
            "image_size": f"{params['image_width']}x{params['image_height']}",
            "num_pairs": params["num_pairs"],
            "rms_error_px": round(params["rms_error_px"], 5),
            "certainty_percent": round(params["certainty_percent"], 2),
            "baseline_mm": round(params["baseline_mm"], 3),
            "camera_matrix_left": params["camera_matrix_left"],
            "dist_coeffs_left": params["dist_coeffs_left"],
            "camera_matrix_right": params["camera_matrix_right"],
            "dist_coeffs_right": params["dist_coeffs_right"],
            "R": params["R"],
            "T": params["T"],
        }
        self.params_text.delete("1.0", tk.END)
        self.params_text.insert(tk.END, json.dumps(summary, indent=2))

    # ------------------------------------------------------------------
    # Video Capture & Live GUI Rendering
    # ------------------------------------------------------------------
    def _capture_loop(self):
        while self.running:
            now = time.perf_counter()
            self._poll_one_camera("left", now)
            self._poll_one_camera("right", now)
            time.sleep(0.005)

    def _compute_preview_bounds(self, widget: tk.Widget) -> Tuple[int, int]:
        """Clamp preview target bounds to the widget's allocated container size."""
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

    def _frame_to_photo(self, frame: np.ndarray, target_w: int, target_h: int, label_text: str) -> ImageTk.PhotoImage:
        annotated = frame.copy()
        cv2.putText(
            annotated,
            label_text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if self.rectification_ready and self.show_rectified_var.get():
            h_img, w_img = annotated.shape[:2]
            for y in range(40, h_img, 40):
                cv2.line(annotated, (0, y), (w_img, y), (0, 255, 255), 1)

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
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

        use_rect = self.rectification_ready and self.show_rectified_var.get()
        if use_rect:
            if (
                frame_l is not None
                and self.map_l1 is not None
                and frame_l.shape[1::-1] == (self.map_l1.shape[1], self.map_l1.shape[0])
            ):
                frame_l = cv2.remap(frame_l, self.map_l1, self.map_l2, cv2.INTER_LINEAR)
            if (
                frame_r is not None
                and self.map_r1 is not None
                and frame_r.shape[1::-1] == (self.map_r1.shape[1], self.map_r1.shape[0])
            ):
                frame_r = cv2.remap(frame_r, self.map_r1, self.map_r2, cv2.INTER_LINEAR)

        mode_tag = "RECTIFIED" if use_rect else "RAW"

        if frame_l is not None:
            tw_l, th_l = self._compute_preview_bounds(self.left_canvas)
            status_l = f"{fps_l:.1f} FPS" if online_l else "RECONNECTING..."
            photo_l = self._frame_to_photo(
                frame_l,
                tw_l,
                th_l,
                f"LEFT [{mode_tag}] | {status_l}",
            )
            self.left_canvas.configure(image=photo_l)
            self.left_canvas.image = photo_l

        if frame_r is not None:
            tw_r, th_r = self._compute_preview_bounds(self.right_canvas)
            status_r = f"{fps_r:.1f} FPS" if online_r else "RECONNECTING..."
            photo_r = self._frame_to_photo(
                frame_r,
                tw_r,
                th_r,
                f"RIGHT [{mode_tag}] | {status_r}",
            )
            self.right_canvas.configure(image=photo_r)
            self.right_canvas.image = photo_r

        self.root.after(20, self._update_ui_loop)

    def on_close(self):
        self.running = False
        if hasattr(self, "capture_thread") and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.5)
        self._release_cameras()
        self.root.destroy()


def main():
    root = tk.Tk()
    StereoCameraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()