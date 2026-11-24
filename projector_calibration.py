import json
import os
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from camera_manager import (
    SingleThreadStereoCapture,
    list_video_device_indices,
)

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
    if proj_w <= 0 or proj_h <= 0 or marker_px <= 0 or margin_px < 4:
        raise ValueError("Projector dimensions, marker size, and margin must be positive.")
    if proj_w <= 2 * (margin_px + marker_px) or proj_h <= 2 * (margin_px + marker_px):
        raise ValueError("The four markers do not fit without overlapping.")

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


def scale_camera_matrix(
    camera_matrix: np.ndarray,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> np.ndarray:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Camera image dimensions must be positive.")
    scale = np.diag(
        [
            target_width / source_width,
            target_height / source_height,
            1.0,
        ]
    )
    return scale @ camera_matrix


def compute_projector_geometry(
    marker_points_3d: Dict[int, np.ndarray],
    projector_width: int,
    projector_height: int,
    marker_size_px: float,
    margin_px: float,
    known_width: float = 0.0,
    known_height: float = 0.0,
) -> dict:
    if not {0, 1, 2, 3}.issubset(marker_points_3d):
        raise ValueError("All four projector corner markers are required.")
    if projector_width <= 0 or projector_height <= 0 or marker_size_px <= 0 or margin_px < 0:
        raise ValueError("Projector dimensions, marker size, and margin must be valid.")

    marker_points = {
        marker_id: np.asarray(marker_points_3d[marker_id], dtype=np.float64).reshape(3)
        for marker_id in range(4)
    }
    if not np.isfinite(np.vstack(list(marker_points.values()))).all():
        raise ValueError("Triangulated projector markers must be finite.")

    all_points = np.vstack([marker_points[marker_id] for marker_id in range(4)])
    centroid = np.mean(all_points, axis=0)
    centered = all_points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    fitted_normal = vt[-1].astype(np.float64)
    fitted_normal /= np.linalg.norm(fitted_normal)

    u_raw = marker_points[1] - marker_points[0]
    u_projected = u_raw - np.dot(u_raw, fitted_normal) * fitted_normal
    u_length = float(np.linalg.norm(u_projected))
    if u_length < 1e-8:
        raise ValueError("Horizontal marker axis is degenerate.")
    u_axis = u_projected / u_length

    v_raw = marker_points[3] - marker_points[0]
    v_projected = v_raw - np.dot(v_raw, fitted_normal) * fitted_normal
    v_projected -= np.dot(v_projected, u_axis) * u_axis
    v_length = float(np.linalg.norm(v_projected))
    if v_length < 1e-8:
        raise ValueError("Vertical marker axis is degenerate.")
    v_axis = v_projected / v_length

    plane_normal = np.cross(u_axis, v_axis)
    normal_length = float(np.linalg.norm(plane_normal))
    if normal_length < 1e-8:
        raise ValueError("Projector plane axes are degenerate.")
    plane_normal /= normal_length

    center_x0 = margin_px + marker_size_px / 2.0
    center_x1 = projector_width - margin_px - marker_size_px / 2.0
    center_y0 = margin_px + marker_size_px / 2.0
    center_y1 = projector_height - margin_px - marker_size_px / 2.0
    if (
        projector_width <= 2 * (margin_px + marker_size_px)
        or projector_height <= 2 * (margin_px + marker_size_px)
        or center_x1 <= center_x0
        or center_y1 <= center_y0
    ):
        raise ValueError("ArUco markers overlap or do not fit inside the projector image.")

    projector_marker_centers = np.array(
        [
            [center_x0, center_y0],
            [center_x1, center_y0],
            [center_x1, center_y1],
            [center_x0, center_y1],
        ],
        dtype=np.float64,
    )
    marker_plane_coords = np.array(
        [
            [
                np.dot(marker_points[mid] - marker_points[0], u_axis),
                np.dot(marker_points[mid] - marker_points[0], v_axis),
            ]
            for mid in range(4)
        ],
        dtype=np.float64,
    )
    image_to_plane, _ = cv2.findHomography(
        projector_marker_centers, marker_plane_coords, method=0
    )
    if image_to_plane is None or not np.isfinite(image_to_plane).all():
        raise ValueError("Could not fit projector coordinate mapping.")

    image_corners = np.array(
        [
            [
                [0.0, 0.0],
                [projector_width - 1.0, 0.0],
                [projector_width - 1.0, projector_height - 1.0],
                [0.0, projector_height - 1.0],
            ]
        ],
        dtype=np.float64,
    )
    plane_corners_from_marker = cv2.perspectiveTransform(
        image_corners, image_to_plane
    )[0]
    origin_marker = marker_points[0]
    p_origin = (
        origin_marker
        + plane_corners_from_marker[0, 0] * u_axis
        + plane_corners_from_marker[0, 1] * v_axis
    )
    plane_corners = plane_corners_from_marker - plane_corners_from_marker[0]
    plane_to_pixel, _ = cv2.findHomography(
        plane_corners.astype(np.float64), image_corners[0], method=0
    )
    if plane_to_pixel is None or not np.isfinite(plane_to_pixel).all():
        raise ValueError("Could not fit inverse projector mapping.")

    projector_corners_3d = np.array(
        [
            p_origin + point[0] * u_axis + point[1] * v_axis
            for point in plane_corners
        ],
        dtype=np.float64,
    )
    width = 0.5 * (
        np.linalg.norm(projector_corners_3d[1] - projector_corners_3d[0])
        + np.linalg.norm(projector_corners_3d[2] - projector_corners_3d[3])
    )
    height = 0.5 * (
        np.linalg.norm(projector_corners_3d[3] - projector_corners_3d[0])
        + np.linalg.norm(projector_corners_3d[2] - projector_corners_3d[1])
    )
    if known_width > 0.0 and known_height <= 0.0:
        height *= known_width / max(width, 1e-8)
        width = known_width
    elif known_height > 0.0 and known_width <= 0.0:
        width *= known_height / max(height, 1e-8)
        height = known_height
    elif known_width > 0.0 and known_height > 0.0:
        width, height = known_width, known_height

    plane_coeffs = (
        float(plane_normal[0]),
        float(plane_normal[1]),
        float(plane_normal[2]),
        float(-np.dot(plane_normal, p_origin)),
    )
    plane_rms = float(np.sqrt(np.mean((centered @ plane_normal) ** 2)))
    return {
        "P_origin": p_origin,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "plane_normal": plane_normal,
        "plane_coeffs": plane_coeffs,
        "plane_to_pixel_homography": plane_to_pixel,
        "projector_corners_3d": projector_corners_3d,
        "width": float(width),
        "height": float(height),
        "plane_rms": plane_rms,
    }


class ProjectorCalibrationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Projector Plane & Coordinate Calibration")
        self.root.geometry("1380x860")

        self.cam_settings = self._load_cam_settings()
        self.stereo_params = self._load_stereo_params()
        self.proj_calib_result: Optional[dict] = self._load_existing_projector_calib()

        self.running = True
        self._build_ui()
        self._apply_cam_settings_to_widgets()
        if self.proj_calib_result:
            self._display_projector_calib(self.proj_calib_result)

        initial_cams = list_video_device_indices()
        self.left_cam_combo["values"] = initial_cams
        self.right_cam_combo["values"] = initial_cams
        self.capture = SingleThreadStereoCapture(
            settings=self.cam_settings,
            left_idx=int(self.left_cam_var.get()),
            right_idx=int(self.right_cam_var.get()),
            on_scan_complete=self._on_camera_scan_complete,
        )

        if self.stereo_params is None:
            self.status_var.set(
                f"Warning: {CALIB_PARAMS_FILE} not found. Run stereo calibration first."
            )
        else:
            self.status_var.set(
                f"Loaded stereo calibration from {CALIB_PARAMS_FILE}. Generate ArUco image or calibrate projector."
            )

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
        self.capture.request_scan()

    def _on_camera_scan_complete(self, cams: List[int]):
        if self.running:
            self.root.after(0, lambda c=cams: self._on_initial_scan_done(c))

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
        self.capture.request_scan()

    def _open_selected_cameras(self):
        left_idx = int(self.left_cam_var.get())
        right_idx = int(self.right_cam_var.get())
        self.capture.set_indices(left_idx, right_idx)
        self.status_var.set(f"Opened Left Cam #{left_idx} and Right Cam #{right_idx}")

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
        if pw <= 2 * (margin + m_px) or ph <= 2 * (margin + m_px):
            messagebox.showerror(
                "Invalid Dimensions",
                "The four markers must fit without overlapping inside the projector image.",
            )
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

        snapshot = self.capture.get_synchronized_snapshot()
        frame_l = snapshot["left_frame"]
        frame_r = snapshot["right_frame"]

        if frame_l is None or frame_r is None:
            messagebox.showwarning(
                "No Synchronized Frames",
                "Waiting for recent, time-aligned frames from both cameras.",
            )
            return

        dictionary, detector = get_aruco_detector(self.dict_var.get())
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        corners_l, ids_l = detect_aruco_markers(gray_l, dictionary, detector)
        corners_r, ids_r = detect_aruco_markers(gray_r, dictionary, detector)

        centers_l = self._extract_marker_centers_map(corners_l, ids_l)
        centers_r = self._extract_marker_centers_map(corners_r, ids_r)

        common_ids = sorted(set(centers_l.keys()) & set(centers_r.keys()))
        required_ids = {0, 1, 2, 3}
        if not required_ids.issubset(common_ids):
            messagebox.showinfo(
                "Missing Projector Corner Markers",
                f"All four projector corner markers {sorted(required_ids)} must be detected in BOTH cameras.\n"
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
            source_size = (
                int(self.stereo_params["image_width"]),
                int(self.stereo_params["image_height"]),
            )
            mtx_l = scale_camera_matrix(
                mtx_l,
                source_size,
                (frame_l.shape[1], frame_l.shape[0]),
            )
            mtx_r = scale_camera_matrix(
                mtx_r,
                source_size,
                (frame_r.shape[1], frame_r.shape[0]),
            )
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
            except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                messagebox.showerror(
                    "Triangulation Failed",
                    f"Could not triangulate marker {mid}: {exc}",
                )
                return

        if not required_ids.issubset(pts_3d_by_id):
            messagebox.showerror(
                "Triangulation Failed",
                "All four projector corner markers must triangulate successfully.",
            )
            return

        proj_w = int(self.proj_w_var.get())
        proj_h = int(self.proj_h_var.get())
        m_px = float(self.marker_px_var.get())
        margin_px = float(self.margin_px_var.get())
        known_w = float(self.known_w_mm_var.get())
        known_h = float(self.known_h_mm_var.get())
        try:
            geometry = compute_projector_geometry(
                pts_3d_by_id,
                proj_w,
                proj_h,
                m_px,
                margin_px,
                known_width=known_w,
                known_height=known_h,
            )
        except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
            messagebox.showerror("Invalid Projector Geometry", str(exc))
            return

        p_origin = geometry["P_origin"]
        u_axis = geometry["u_axis"]
        v_axis = geometry["v_axis"]
        plane_normal = geometry["plane_normal"]
        a, b, c, d = geometry["plane_coeffs"]
        plane_to_pixel = geometry["plane_to_pixel_homography"]
        projector_corners_3d = geometry["projector_corners_3d"]
        width_units = geometry["width"]
        height_units = geometry["height"]
        plane_rms = geometry["plane_rms"]

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
            "plane_to_pixel_homography": plane_to_pixel.tolist(),
            "projector_corners_3d": projector_corners_3d.tolist(),
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

        snapshot = self.capture.get_snapshot(copy_frames=False)
        frame_l = snapshot["left_frame"]
        frame_r = snapshot["right_frame"]
        fps_l = float(snapshot["fps_left"])
        fps_r = float(snapshot["fps_right"])
        online_l = bool(snapshot["left_online"])
        online_r = bool(snapshot["right_online"])

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
        self.capture.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    ProjectorCalibrationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()