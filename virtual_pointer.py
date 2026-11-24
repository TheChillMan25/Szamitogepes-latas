import argparse
import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QRadialGradient
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from camera_manager import (
    SingleThreadStereoCapture,
    list_video_device_indices,
)

SETTINGS_FILE = Path("./camera_settings.json")
CALIB_PARAMS_FILE = Path("./calibration_params.json")
PROJECTOR_CALIB_FILE = Path("./projector_calibration.json")
HAND_LANDMARKER_MODEL_FILE = Path("./hand_landmarker.task")
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

DEFAULT_CAMERA_SETTINGS = {
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

PREVIEW_FRAME_SIZE = (640, 360)
MAX_TRACKING_WIDTH = 640


@dataclass(frozen=True)
class StereoCalibration:
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    image_size: Tuple[int, int]
    P1: np.ndarray
    P2: np.ndarray


@dataclass(frozen=True)
class ProjectorPlane:
    p_origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    plane_coeffs: Tuple[float, float, float, float]
    dimensions: Tuple[float, float]  # (L_w, L_h)
    plane_to_pixel_homography: Optional[np.ndarray] = None
    corners: Optional[np.ndarray] = None
    projector_resolution: Optional[Tuple[int, int]] = None


@dataclass
class CameraOverlay3D:
    plane_polygon_px: Optional[np.ndarray] = None
    axes_px: Optional[Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]] = None
    ray_line_px: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    intersection_px: Optional[Tuple[int, int]] = None


def _matrix(data: dict, name: str, shape: Tuple[int, int]) -> np.ndarray:
    value = np.asarray(data[name], dtype=np.float64)
    if value.size != shape[0] * shape[1]:
        raise ValueError(f"{name} must contain {shape[0] * shape[1]} values.")
    value = value.reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values.")
    return value


def load_camera_settings(settings_path: Path) -> dict:
    settings = dict(DEFAULT_CAMERA_SETTINGS)
    if settings_path.exists():
        try:
            with settings_path.open("r", encoding="utf-8") as f:
                settings.update(json.load(f))
        except Exception:
            pass
    return settings


def load_stereo_calibration(config_path: Path) -> StereoCalibration:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    K1 = _matrix(data, "camera_matrix_left", (3, 3))
    K2 = _matrix(data, "camera_matrix_right", (3, 3))
    R = _matrix(data, "R", (3, 3))
    T = _matrix(data, "T", (3, 1))
    D1 = np.asarray(data["dist_coeffs_left"], dtype=np.float64).reshape(-1, 1)
    D2 = np.asarray(data["dist_coeffs_right"], dtype=np.float64).reshape(-1, 1)

    width = int(data["image_width"])
    height = int(data["image_height"])
    if width <= 0 or height <= 0:
        raise ValueError("Calibration image dimensions must be positive.")
    if not np.isfinite(D1).all() or not np.isfinite(D2).all():
        raise ValueError("Distortion coefficients must be finite.")
    if K1[0, 0] <= 0 or K1[1, 1] <= 0 or K2[0, 0] <= 0 or K2[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive.")

    P1 = K1 @ np.hstack((np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)))
    P2 = K2 @ np.hstack((R, T))
    return StereoCalibration(
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        R=R,
        T=T,
        image_size=(width, height),
        P1=P1,
        P2=P2,
    )


def load_projector_calibration(projector_path: Path) -> ProjectorPlane:
    with projector_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    p_origin = np.asarray(data["P_origin"], dtype=np.float64).reshape(3)
    u_axis = np.asarray(data["u_axis"], dtype=np.float64).reshape(3)
    v_axis = np.asarray(data["v_axis"], dtype=np.float64).reshape(3)

    u_norm = float(np.linalg.norm(u_axis))
    v_norm = float(np.linalg.norm(v_axis))
    if u_norm < 1e-8 or v_norm < 1e-8:
        raise ValueError("Projector u_axis and v_axis must have non-zero length.")
    u_axis = u_axis / u_norm
    v_axis = v_axis / v_norm

    coeffs = data["plane_coeffs"]
    plane_coeffs = (
        float(coeffs["a"]),
        float(coeffs["b"]),
        float(coeffs["c"]),
        float(coeffs["d"]),
    )

    size_data = data["projector_size"]
    lw = float(size_data["width"])
    lh = float(size_data["height"])
    if lw <= 0.0 or lh <= 0.0:
        raise ValueError("Projector width and height must be positive.")

    if not (
        np.isfinite(p_origin).all()
        and np.isfinite(u_axis).all()
        and np.isfinite(v_axis).all()
        and np.isfinite(plane_coeffs).all()
    ):
        raise ValueError("Projector calibration contains non-finite values.")

    homography_data = data.get("plane_to_pixel_homography")
    if homography_data is None:
        raise ValueError(
            "Projector calibration predates full-frame mapping; recalibrate the projector."
        )
    plane_to_pixel_homography = _matrix(
        {"homography": homography_data},
        "homography",
        (3, 3),
    )
    corners_data = data.get("projector_corners_3d")
    if corners_data is None:
        raise ValueError("Projector calibration is missing its calibrated screen corners.")
    projector_corners = np.asarray(corners_data, dtype=np.float64).reshape(4, 3)
    if not np.isfinite(projector_corners).all():
        raise ValueError("Projector corners contain non-finite values.")
    projector_resolution_data = data["projector_resolution"]
    projector_resolution = (
        int(projector_resolution_data["width_px"]),
        int(projector_resolution_data["height_px"]),
    )
    if min(projector_resolution) <= 0:
        raise ValueError("Projector resolution must be positive.")

    return ProjectorPlane(
        p_origin=p_origin,
        u_axis=u_axis,
        v_axis=v_axis,
        plane_coeffs=plane_coeffs,
        dimensions=(lw, lh),
        plane_to_pixel_homography=plane_to_pixel_homography,
        corners=projector_corners,
        projector_resolution=projector_resolution,
    )


def calibration_for_frame_sizes(
    calibration: StereoCalibration,
    size1: Tuple[int, int],
    size2: Tuple[int, int],
) -> StereoCalibration:
    if size1 == calibration.image_size and size2 == calibration.image_size:
        return calibration

    def scale_intrinsics(
        matrix: np.ndarray, source_size: Tuple[int, int], target_size: Tuple[int, int]
    ) -> np.ndarray:
        w0, h0 = source_size
        w, h = target_size
        return np.diag([w / w0, h / h0, 1.0]) @ matrix

    K1 = scale_intrinsics(calibration.K1, calibration.image_size, size1)
    K2 = scale_intrinsics(calibration.K2, calibration.image_size, size2)
    P1 = K1 @ np.hstack((np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)))
    P2 = K2 @ np.hstack((calibration.R, calibration.T))
    return StereoCalibration(
        K1=K1,
        D1=calibration.D1,
        K2=K2,
        D2=calibration.D2,
        R=calibration.R,
        T=calibration.T,
        image_size=calibration.image_size,
        P1=P1,
        P2=P2,
    )


def project_3d_points_to_camera(
    points_3d: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R_cam: np.ndarray,
    t_cam: np.ndarray,
) -> Optional[np.ndarray]:
    """Project N×3 points in left-camera reference frame into a camera's pixel coords."""
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0 or not np.isfinite(pts).all():
        return None
    rvec, _ = cv2.Rodrigues(np.asarray(R_cam, dtype=np.float64))
    tvec = np.asarray(t_cam, dtype=np.float64).reshape(3, 1)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, K, D)
    projected = proj.reshape(-1, 2)
    if not np.isfinite(projected).all():
        return None
    return projected


def _to_int_pt(pt_2d: np.ndarray) -> Tuple[int, int]:
    return int(round(float(pt_2d[0]))), int(round(float(pt_2d[1])))


def build_camera_overlay_3d(
    projector_plane: Optional[ProjectorPlane],
    frame_calib: Optional[StereoCalibration],
    is_right_camera: bool,
    p_tip: Optional[np.ndarray] = None,
    v_hat: Optional[np.ndarray] = None,
    p_intersect: Optional[np.ndarray] = None,
) -> Optional[CameraOverlay3D]:
    if projector_plane is None or frame_calib is None:
        return None

    if is_right_camera:
        K = frame_calib.K2
        D = frame_calib.D2
        R_cam = frame_calib.R
        t_cam = frame_calib.T
    else:
        K = frame_calib.K1
        D = frame_calib.D1
        R_cam = np.eye(3, dtype=np.float64)
        t_cam = np.zeros((3, 1), dtype=np.float64)

    p0 = projector_plane.p_origin
    u = projector_plane.u_axis
    v = projector_plane.v_axis
    lw, lh = projector_plane.dimensions

    # 4 corners of the projector rectangle in 3D space
    corners_3d = projector_plane.corners
    if corners_3d is None:
        corners_3d = np.vstack(
            [
                p0,
                p0 + lw * u,
                p0 + lw * u + lh * v,
                p0 + lh * v,
            ]
        )
    corners_2d = project_3d_points_to_camera(corners_3d, K, D, R_cam, t_cam)
    plane_poly: Optional[np.ndarray] = None
    if corners_2d is not None and len(corners_2d) == 4:
        plane_poly = np.rint(corners_2d).astype(np.int32).reshape(-1, 1, 2)

    # Unit vectors u, v, and normal n = normalize(u x v) scaled for clear visibility
    axis_len = max(10.0, 0.15 * min(lw, lh))
    n_axis = np.cross(u, v)
    n_norm = float(np.linalg.norm(n_axis))
    if n_norm > 1e-8:
        n_axis = n_axis / n_norm
    else:
        n_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    axes_3d = np.vstack(
        [
            p0,
            p0 + axis_len * u,
            p0 + axis_len * v,
            p0 + axis_len * n_axis,
        ]
    )
    axes_2d = project_3d_points_to_camera(axes_3d, K, D, R_cam, t_cam)
    axes_px: Optional[Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]] = None
    if axes_2d is not None and len(axes_2d) == 4:
        origin_pt = _to_int_pt(axes_2d[0])
        axes_px = {
            "u": (origin_pt, _to_int_pt(axes_2d[1])),
            "v": (origin_pt, _to_int_pt(axes_2d[2])),
            "n": (origin_pt, _to_int_pt(axes_2d[3])),
        }

    # Pointing ray line & plane intersection point
    ray_line_px: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    intersection_px: Optional[Tuple[int, int]] = None

    if p_tip is not None and v_hat is not None:
        ray_start = p_tip - 0.05 * v_hat
        if p_intersect is not None:
            ray_end = p_intersect + 0.05 * v_hat
        else:
            ray_end = p_tip + max(lw, lh) * 0.8 * v_hat

        ray_2d = project_3d_points_to_camera(
            np.vstack([ray_start, ray_end]), K, D, R_cam, t_cam
        )
        if ray_2d is not None and len(ray_2d) == 2:
            ray_line_px = (_to_int_pt(ray_2d[0]), _to_int_pt(ray_2d[1]))

    if p_intersect is not None:
        inter_2d = project_3d_points_to_camera(
            p_intersect.reshape(1, 3), K, D, R_cam, t_cam
        )
        if inter_2d is not None and len(inter_2d) == 1:
            intersection_px = _to_int_pt(inter_2d[0])

    return CameraOverlay3D(
        plane_polygon_px=plane_poly,
        axes_px=axes_px,
        ray_line_px=ray_line_px,
        intersection_px=intersection_px,
    )


class HandTrackerBackend:
    """Wrapper supporting MediaPipe Tasks HandLandmarker and legacy mp.solutions.hands."""

    def __init__(self, stream_count: int = 1):
        if stream_count < 1:
            raise ValueError("stream_count must be positive.")
        self._stream_count = stream_count
        self._mode = None
        self._task_landmarkers = []
        self._legacy_hands = []
        self._mp = None
        self._last_timestamp_ms = [0] * stream_count
        self._init_tracker()

    def _ensure_model_file(self) -> Path:
        if HAND_LANDMARKER_MODEL_FILE.exists() and HAND_LANDMARKER_MODEL_FILE.stat().st_size > 0:
            return HAND_LANDMARKER_MODEL_FILE

        for candidate in (
            Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task",
            Path(__file__).with_name("hand_landmarker.task"),
        ):
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate

        print(f"Downloading {HAND_LANDMARKER_MODEL_FILE.name}...", flush=True)
        try:
            urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, str(HAND_LANDMARKER_MODEL_FILE))
            if HAND_LANDMARKER_MODEL_FILE.exists() and HAND_LANDMARKER_MODEL_FILE.stat().st_size > 0:
                return HAND_LANDMARKER_MODEL_FILE
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {HAND_LANDMARKER_MODEL_FILE.name} automatically ({exc}). "
                f"Place a valid hand_landmarker.task file next to the script."
            ) from exc

        raise RuntimeError(f"Missing {HAND_LANDMARKER_MODEL_FILE.name}.")

    def _init_tracker(self):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(f"MediaPipe is not installed: {exc}") from exc

        self._mp = mp

        tasks_vision = getattr(getattr(mp, "tasks", None), "vision", None)
        base_options_cls = getattr(getattr(mp, "tasks", None), "BaseOptions", None)
        if (
            tasks_vision is not None
            and base_options_cls is not None
            and hasattr(tasks_vision, "HandLandmarker")
        ):
            model_path = self._ensure_model_file()
            running_mode_cls = tasks_vision.RunningMode
            options = tasks_vision.HandLandmarkerOptions(
                base_options=base_options_cls(model_asset_path=str(model_path)),
                running_mode=running_mode_cls.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.65,
                min_hand_presence_confidence=0.65,
                min_tracking_confidence=0.6,
            )
            for _ in range(self._stream_count):
                try:
                    self._task_landmarkers.append(
                        tasks_vision.HandLandmarker.create_from_options(options)
                    )
                except Exception:
                    self.close()
                    raise
            self._mode = "tasks"
            return

        hands_api = getattr(getattr(mp, "solutions", None), "hands", None)
        if hands_api is not None:
            try:
                for _ in range(self._stream_count):
                    self._legacy_hands.append(
                        hands_api.Hands(
                            static_image_mode=False,
                            max_num_hands=1,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.6,
                        )
                    )
            except Exception:
                self.close()
                raise
            self._mode = "solutions"
            return

        raise RuntimeError(
            "MediaPipe Hands API is unavailable: neither mediapipe.tasks.vision.HandLandmarker "
            "nor mediapipe.solutions.hands was found in the installed mediapipe package."
        )

    def detect_index_landmarks(
        self, frame_bgr: np.ndarray, stream_index: int = 0
    ) -> Optional[Dict[str, np.ndarray]]:
        if not 0 <= stream_index < self._stream_count:
            raise ValueError(f"stream_index must be in [0, {self._stream_count - 1}].")

        orig_h, orig_w = frame_bgr.shape[:2]
        if orig_w > MAX_TRACKING_WIDTH:
            scale = MAX_TRACKING_WIDTH / float(orig_w)
            proc_w = MAX_TRACKING_WIDTH
            proc_h = max(1, int(round(orig_h * scale)))
            proc_bgr = cv2.resize(frame_bgr, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        else:
            proc_bgr = frame_bgr

        rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
        names = (
            "INDEX_FINGER_MCP",
            "INDEX_FINGER_PIP",
            "INDEX_FINGER_DIP",
            "INDEX_FINGER_TIP",
        )
        indices = (5, 6, 7, 8)

        if self._mode == "tasks" and self._task_landmarkers:
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(time.perf_counter() * 1000)
            if ts_ms <= self._last_timestamp_ms[stream_index]:
                ts_ms = self._last_timestamp_ms[stream_index] + 1
            self._last_timestamp_ms[stream_index] = ts_ms

            result = self._task_landmarkers[stream_index].detect_for_video(mp_image, ts_ms)
            if not result.hand_landmarks:
                return None
            hand_lms = result.hand_landmarks[0]
            return {
                name: np.array(
                    [hand_lms[idx].x * orig_w, hand_lms[idx].y * orig_h],
                    dtype=np.float32,
                )
                for name, idx in zip(names, indices)
            }

        if self._mode == "solutions" and self._legacy_hands:
            result = self._legacy_hands[stream_index].process(rgb)
            if not result.multi_hand_landmarks:
                return None
            hand = result.multi_hand_landmarks[0]
            return {
                name: np.array(
                    [hand.landmark[idx].x * orig_w, hand.landmark[idx].y * orig_h],
                    dtype=np.float32,
                )
                for name, idx in zip(names, indices)
            }

        return None

    def close(self):
        for landmarker in self._task_landmarkers:
            try:
                landmarker.close()
            except Exception:
                pass
        self._task_landmarkers.clear()
        for hands in self._legacy_hands:
            try:
                hands.close()
            except Exception:
                pass
        self._legacy_hands.clear()


def _undistort_landmarks(
    landmarks: Dict[str, np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> Dict[str, np.ndarray]:
    corrected: Dict[str, np.ndarray] = {}
    for name, pixel in landmarks.items():
        point = np.asarray(pixel, dtype=np.float32).reshape(1, 1, 2)
        corrected[name] = cv2.undistortPoints(
            point, camera_matrix, distortion, P=camera_matrix
        )[0, 0]
    return corrected


def triangulate_landmarks(
    landmarks_cam1: Dict[str, np.ndarray],
    landmarks_cam2: Dict[str, np.ndarray],
    P1: np.ndarray,
    P2: np.ndarray,
) -> Dict[str, np.ndarray]:
    if landmarks_cam1.keys() != landmarks_cam2.keys():
        raise ValueError("The two cameras must provide matching landmarks.")

    points_3d: Dict[str, np.ndarray] = {}
    for name, point1 in landmarks_cam1.items():
        p1 = np.asarray(point1, dtype=np.float64).reshape(2, 1)
        p2 = np.asarray(landmarks_cam2[name], dtype=np.float64).reshape(2, 1)
        points_4d = cv2.triangulatePoints(P1, P2, p1, p2)
        w = float(points_4d[3, 0])
        if abs(w) < 1e-8:
            raise ValueError(f"Could not triangulate landmark {name}.")
        point_3d = points_4d[:3, 0] / w
        if not np.isfinite(point_3d).all():
            raise ValueError(f"Triangulated landmark {name} is not finite.")
        points_3d[name] = point_3d
    return points_3d


def calculate_pointing_direction(
    points_3d: Dict[str, np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if "INDEX_FINGER_DIP" not in points_3d or "INDEX_FINGER_TIP" not in points_3d:
        return None, None
    p_dip = points_3d["INDEX_FINGER_DIP"]
    p_tip = points_3d["INDEX_FINGER_TIP"]
    direction = p_tip - p_dip
    length = float(np.linalg.norm(direction))
    if length < 1e-5:
        return None, None
    return p_tip, direction / length


def calculate_intersection(
    p_tip: np.ndarray,
    v_hat: np.ndarray,
    plane_coeffs: Tuple[float, float, float, float],
) -> Optional[np.ndarray]:
    a, b, c, d = plane_coeffs
    x0, y0, z0 = float(p_tip[0]), float(p_tip[1]), float(p_tip[2])
    vx, vy, vz = float(v_hat[0]), float(v_hat[1]), float(v_hat[2])

    divider = a * vx + b * vy + c * vz
    if abs(divider) < 1e-6:
        return None

    lam = -(a * x0 + b * y0 + c * z0 + d) / divider
    if lam < 0.0:
        return None

    return p_tip + lam * v_hat


def to_screen_pixels(
    p_intersect: np.ndarray,
    p_origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    dimensions: Tuple[float, float],
    resolution: Tuple[int, int],
    plane_to_pixel_homography: Optional[np.ndarray] = None,
    projector_resolution: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    lw, lh = dimensions
    sw, sh = resolution
    w = p_intersect - p_origin
    x_2d = float(np.dot(w, u_axis))
    y_2d = float(np.dot(w, v_axis))
    if plane_to_pixel_homography is not None:
        projected = plane_to_pixel_homography @ np.array(
            [x_2d, y_2d, 1.0], dtype=np.float64
        )
        if abs(float(projected[2])) < 1e-9:
            raise ValueError("Projector mapping produced an invalid homogeneous point.")
        x_px = float(projected[0] / projected[2])
        y_px = float(projected[1] / projected[2])
        if projector_resolution is None:
            raise ValueError("Projector resolution is required with a homography mapping.")
        projector_w, projector_h = projector_resolution
        x_px *= (sw - 1) / max(projector_w - 1, 1)
        y_px *= (sh - 1) / max(projector_h - 1, 1)
        return (
            round(float(np.clip(x_px, 0.0, sw - 1))),
            round(float(np.clip(y_px, 0.0, sh - 1))),
        )
    x_norm = float(np.clip(x_2d / lw, 0.0, 1.0))
    y_norm = float(np.clip(y_2d / lh, 0.0, 1.0))
    return round(x_norm * (sw - 1)), round(y_norm * (sh - 1))


class KalmanFilter2D:
    def __init__(self):
        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.filter.transitionMatrix = np.eye(4, dtype=np.float32)
        self.filter.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.initialized = False
        self._last_update_time: Optional[float] = None

    def reset(self):
        zero_state = np.zeros((4, 1), dtype=np.float32)
        initial_covariance = np.eye(4, dtype=np.float32)
        self.filter.statePre = zero_state.copy()
        self.filter.statePost = zero_state.copy()
        self.filter.errorCovPre = initial_covariance.copy()
        self.filter.errorCovPost = initial_covariance.copy()
        self.initialized = False
        self._last_update_time = None

    def update(self, x: float, y: float) -> Tuple[int, int]:
        measurement = np.array([[x], [y]], dtype=np.float32)
        if not self.initialized:
            state = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
            self.filter.statePre = state.copy()
            self.filter.statePost = state.copy()
            self.initialized = True
            self._last_update_time = time.perf_counter()
            return round(x), round(y)

        now = time.perf_counter()
        dt = min(0.25, max(0.001, now - (self._last_update_time or now)))
        self.filter.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self._last_update_time = now
        self.filter.predict()
        corrected = self.filter.correct(measurement)
        return round(float(corrected[0, 0])), round(float(corrected[1, 0]))


class CameraPanel(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        heading = QLabel(title)
        heading.setStyleSheet("font-weight: 600; color: #d7e0ea;")
        self.image = QLabel()
        self.image.setFixedSize(*PREVIEW_FRAME_SIZE)
        self.image.setAlignment(Qt.AlignCenter)
        layout.addWidget(heading)
        layout.addWidget(self.image)

    def set_frame(
        self,
        frame: np.ndarray,
        landmarks: Optional[Dict[str, np.ndarray]] = None,
        header_text: str = "",
        overlay_3d: Optional[CameraOverlay3D] = None,
    ) -> None:
        vis = frame.copy()

        if overlay_3d is not None:
            # 1. Projector rectangle in 3D space
            if overlay_3d.plane_polygon_px is not None:
                fill = vis.copy()
                cv2.fillPoly(fill, [overlay_3d.plane_polygon_px], (255, 160, 40))
                cv2.addWeighted(fill, 0.16, vis, 0.84, 0, dst=vis)
                cv2.polylines(
                    vis,
                    [overlay_3d.plane_polygon_px],
                    isClosed=True,
                    color=(255, 180, 60),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

            # 2. Unit vectors of plane (u = cyan/reddish, v = green, n = amber/blue)
            if overlay_3d.axes_px is not None:
                axis_styles = {
                    "u": ((0, 120, 255), "u"),
                    "v": ((60, 230, 80), "v"),
                    "n": ((255, 210, 40), "n"),
                }
                for key, (start_pt, end_pt) in overlay_3d.axes_px.items():
                    color, label = axis_styles.get(key, ((255, 255, 255), key))
                    cv2.arrowedLine(
                        vis,
                        start_pt,
                        end_pt,
                        color=color,
                        thickness=2,
                        line_type=cv2.LINE_AA,
                        tipLength=0.22,
                    )
                    cv2.putText(
                        vis,
                        label,
                        (end_pt[0] + 5, end_pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            # 3. Pointing ray line
            if overlay_3d.ray_line_px is not None:
                p_a, p_b = overlay_3d.ray_line_px
                cv2.line(vis, p_a, p_b, (0, 255, 255), 2, cv2.LINE_AA)

            # 4. Intersection point on projector plane
            if overlay_3d.intersection_px is not None:
                ix, iy = overlay_3d.intersection_px
                cv2.circle(vis, (ix, iy), 11, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(vis, (ix, iy), 5, (0, 255, 255), -1, cv2.LINE_AA)

        if landmarks:
            ordered = [
                landmarks[k]
                for k in (
                    "INDEX_FINGER_MCP",
                    "INDEX_FINGER_PIP",
                    "INDEX_FINGER_DIP",
                    "INDEX_FINGER_TIP",
                )
                if k in landmarks
            ]
            if len(ordered) >= 2:
                pts = np.rint(ordered).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], False, (80, 190, 255), 3, cv2.LINE_AA)
            for name, pt in landmarks.items():
                color = (0, 0, 255) if name == "INDEX_FINGER_TIP" else (70, 220, 255)
                cv2.circle(vis, tuple(np.rint(pt).astype(int)), 6, color, -1, cv2.LINE_AA)

        if header_text:
            cv2.putText(
                vis,
                header_text,
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        height, width, _ = vis.shape
        image = QImage(
            vis.data, width, height, vis.strides[0], QImage.Format_BGR888
        ).copy()
        self.image.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )


class LaserOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.position: Optional[Tuple[int, int]] = None

    def set_position(self, position: Optional[Tuple[int, int]]) -> None:
        self.position = position
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.position is None:
            return

        x, y = self.position
        glow = QRadialGradient(x, y, 18)
        glow.setColorAt(0.0, QColor(255, 25, 25, 255))
        glow.setColorAt(0.18, QColor(255, 0, 0, 235))
        glow.setColorAt(0.42, QColor(255, 0, 0, 90))
        glow.setColorAt(1.0, QColor(255, 0, 0, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(x - 18, y - 18, 36, 36)
        painter.setBrush(QColor(255, 20, 20, 245))
        painter.drawEllipse(x - 4, y - 4, 8, 8)
        painter.setBrush(QColor(255, 255, 255, 235))
        painter.drawEllipse(x - 1, y - 1, 2, 2)


class _ScanSignalBridge(QObject):
    cameras_scanned = Signal(list)


class VirtualPointerWindow(QMainWindow):
    def __init__(
        self,
        cam_settings: dict,
        calibration: Optional[StereoCalibration],
        projector_plane: Optional[ProjectorPlane],
        screen,
        laser_overlay: LaserOverlay,
        tracker: Optional[HandTrackerBackend],
        tracker_error: Optional[str],
    ):
        super().__init__()
        self.cam_settings = cam_settings
        self.calibration = calibration
        self.projector_plane = projector_plane
        self.tracker = tracker
        self.tracker_error = tracker_error
        self.kalman = KalmanFilter2D()
        self._last_processed_pair: Optional[Tuple[float, float]] = None
        self._last_processed_at: Optional[float] = None
        self._processing_fps = 0.0
        self.screen_geometry = screen.geometry()
        self.laser_overlay = laser_overlay

        self._scan_bridge = _ScanSignalBridge()
        self._scan_bridge.cameras_scanned.connect(self._on_cameras_scanned)

        self.setWindowTitle("Virtual Pointer")
        self.setStyleSheet(
            "QMainWindow, QWidget { background: #171d26; color: #eef2f6; }"
            "QFrame { background: #202936; border: 1px solid #354354; border-radius: 6px; }"
            "QPushButton { background: #2c394b; border: 1px solid #475b75; padding: 5px 10px; border-radius: 4px; }"
            "QPushButton:hover { background: #38485f; }"
            "QComboBox { background: #253040; border: 1px solid #475b75; padding: 4px 8px; border-radius: 4px; }"
        )

        central = QWidget()
        layout = QVBoxLayout(central)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Left (Main) Cam:"))
        self.left_cam_combo = QComboBox()
        self.left_cam_combo.setMinimumWidth(75)
        top_bar.addWidget(self.left_cam_combo)

        top_bar.addSpacing(12)
        top_bar.addWidget(QLabel("Right Cam:"))
        self.right_cam_combo = QComboBox()
        self.right_cam_combo.setMinimumWidth(75)
        top_bar.addWidget(self.right_cam_combo)

        top_bar.addSpacing(8)
        self.rescan_btn = QPushButton("Rescan Cams")
        self.rescan_btn.clicked.connect(self._request_camera_scan)
        top_bar.addWidget(self.rescan_btn)

        top_bar.addStretch()

        if self.projector_plane is not None:
            lw, lh = self.projector_plane.dimensions
            top_bar.addWidget(
                QLabel(
                    f"Plane: {lw:.1f}×{lh:.1f} | Screen: {self.screen_geometry.width()}×{self.screen_geometry.height()}"
                )
            )
        else:
            top_bar.addWidget(QLabel("Plane: MISSING"))

        layout.addLayout(top_bar)

        panels = QHBoxLayout()
        self.camera1 = CameraPanel("Left (Main) Camera")
        self.camera2 = CameraPanel("Right Camera")
        panels.addWidget(self.camera1)
        panels.addWidget(self.camera2)
        layout.addLayout(panels)

        self.status = QLabel("Initializing cameras...")
        self.status.setStyleSheet("color: #a9bacb; padding: 6px;")
        layout.addWidget(self.status)
        self.setCentralWidget(central)
        self.setMinimumSize(920, 480)

        left_init = int(self.cam_settings.get("left_camera_index", 0))
        right_init = int(self.cam_settings.get("right_camera_index", 1))
        initial_cams = list_video_device_indices()
        self._populate_camera_combos(initial_cams, left_init, right_init)

        self.left_cam_combo.currentIndexChanged.connect(self._on_cam_combo_changed)
        self.right_cam_combo.currentIndexChanged.connect(self._on_cam_combo_changed)

        self.capture = SingleThreadStereoCapture(
            settings=self.cam_settings,
            left_idx=self._selected_left_idx(),
            right_idx=self._selected_right_idx(),
            on_scan_complete=self._scan_bridge.cameras_scanned.emit,
            poll_interval_sec=0.015,
        )

        QTimer.singleShot(250, self._request_camera_scan)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_pointer)
        self.timer.start(30)

    def _populate_camera_combos(self, cams: List[int], left_idx: int, right_idx: int):
        all_items = sorted(set(cams) | {left_idx, right_idx})
        self.left_cam_combo.blockSignals(True)
        self.right_cam_combo.blockSignals(True)
        self.left_cam_combo.clear()
        self.right_cam_combo.clear()
        for idx in all_items:
            self.left_cam_combo.addItem(str(idx), idx)
            self.right_cam_combo.addItem(str(idx), idx)

        l_pos = self.left_cam_combo.findData(left_idx)
        r_pos = self.right_cam_combo.findData(right_idx)
        self.left_cam_combo.setCurrentIndex(l_pos if l_pos >= 0 else 0)
        self.right_cam_combo.setCurrentIndex(
            r_pos if r_pos >= 0 else min(1, self.right_cam_combo.count() - 1)
        )
        self.left_cam_combo.blockSignals(False)
        self.right_cam_combo.blockSignals(False)

    def _selected_left_idx(self) -> int:
        data = self.left_cam_combo.currentData()
        return int(data) if data is not None else int(self.left_cam_combo.currentText() or 0)

    def _selected_right_idx(self) -> int:
        data = self.right_cam_combo.currentData()
        return int(data) if data is not None else int(self.right_cam_combo.currentText() or 1)

    def _request_camera_scan(self):
        self.status.setText("Scanning available cameras...")
        self.capture.request_scan()

    def _on_cameras_scanned(self, cams: List[int]):
        cur_l = self._selected_left_idx()
        cur_r = self._selected_right_idx()
        if len(cams) >= 2:
            if cur_l not in cams:
                cur_l = cams[0]
            if cur_r not in cams or cur_r == cur_l:
                cur_r = next((c for c in cams if c != cur_l), cams[1])
        elif len(cams) == 1:
            cur_l = cams[0]
            cur_r = cams[0]

        self._populate_camera_combos(cams, cur_l, cur_r)
        self.kalman.reset()
        self._last_processed_pair = None
        self.capture.set_indices(self._selected_left_idx(), self._selected_right_idx())
        self.status.setText(f"Detected cameras: {cams}")

    def _on_cam_combo_changed(self):
        self.kalman.reset()
        self._last_processed_pair = None
        self.capture.set_indices(self._selected_left_idx(), self._selected_right_idx())

    def update_pointer(self) -> None:
        display_snap = self.capture.get_snapshot(copy_frames=False)
        snap = self.capture.get_synchronized_snapshot()
        frame1: Optional[np.ndarray] = display_snap["left_frame"]  # type: ignore[assignment]
        frame2: Optional[np.ndarray] = display_snap["right_frame"]  # type: ignore[assignment]
        process_frame1: Optional[np.ndarray] = snap["left_frame"]  # type: ignore[assignment]
        process_frame2: Optional[np.ndarray] = snap["right_frame"]  # type: ignore[assignment]
        fps_l = float(snap["fps_left"])
        fps_r = float(snap["fps_right"])
        online_l = bool(display_snap["left_online"])
        online_r = bool(display_snap["right_online"])
        left_idx = int(display_snap["left_idx"])
        right_idx = int(display_snap["right_idx"])

        head_l = (
            f"LEFT #{left_idx} | CAPTURE {fps_l:.1f} FPS"
            if online_l
            else f"LEFT #{left_idx} | RECONNECTING"
        )
        head_r = (
            f"RIGHT #{right_idx} | CAPTURE {fps_r:.1f} FPS"
            if online_r
            else f"RIGHT #{right_idx} | RECONNECTING"
        )

        frame_calib: Optional[StereoCalibration] = None
        if self.calibration is not None and frame1 is not None and frame2 is not None:
            frame_calib = calibration_for_frame_sizes(
                self.calibration,
                (frame1.shape[1], frame1.shape[0]),
                (frame2.shape[1], frame2.shape[0]),
            )

        base_overlay1 = build_camera_overlay_3d(
            self.projector_plane, frame_calib, is_right_camera=False
        )
        base_overlay2 = build_camera_overlay_3d(
            self.projector_plane, frame_calib, is_right_camera=True
        )

        if frame1 is not None:
            self.camera1.set_frame(frame1, None, head_l, base_overlay1)
        if frame2 is not None:
            self.camera2.set_frame(frame2, None, head_r, base_overlay2)

        if frame1 is None or frame2 is None:
            self.laser_overlay.set_position(None)
            self.kalman.reset()
            self._last_processed_pair = None
            self.status.setText("Waiting for frames from both cameras...")
            return

        if self.tracker is None:
            self.laser_overlay.set_position(None)
            self.status.setText(f"Hand tracker unavailable: {self.tracker_error}")
            return

        if self.calibration is None or self.projector_plane is None or frame_calib is None:
            self.laser_overlay.set_position(None)
            self.status.setText(
                "Missing calibration_params.json or projector_calibration.json — run calibration first."
            )
            return

        if process_frame1 is None or process_frame2 is None:
            self.laser_overlay.set_position(None)
            self.kalman.reset()
            self._last_processed_pair = None
            self.status.setText("Waiting for a synchronized stereo frame pair...")
            return

        pair_timestamp = (
            float(snap["left_timestamp"]),
            float(snap["right_timestamp"]),
        )
        if pair_timestamp == self._last_processed_pair:
            return
        self._last_processed_pair = pair_timestamp
        processed_at = time.perf_counter()
        if self._last_processed_at is not None:
            dt = processed_at - self._last_processed_at
            if dt > 0.0:
                self._processing_fps = 0.85 * self._processing_fps + 0.15 / dt
        self._last_processed_at = processed_at

        raw1 = self.tracker.detect_index_landmarks(process_frame1, stream_index=0)
        raw2 = self.tracker.detect_index_landmarks(process_frame2, stream_index=1)

        if raw1 is None or raw2 is None:
            self.camera1.set_frame(frame1, raw1, head_l, base_overlay1)
            self.camera2.set_frame(frame2, raw2, head_r, base_overlay2)
            self.laser_overlay.set_position(None)
            self.kalman.reset()
            self.status.setText("Show index finger to both cameras to point.")
            return

        try:
            undist1 = _undistort_landmarks(raw1, frame_calib.K1, frame_calib.D1)
            undist2 = _undistort_landmarks(raw2, frame_calib.K2, frame_calib.D2)

            points_3d = triangulate_landmarks(
                undist1, undist2, frame_calib.P1, frame_calib.P2
            )
            p_tip, v_hat = calculate_pointing_direction(points_3d)
            if p_tip is None or v_hat is None:
                self.camera1.set_frame(frame1, raw1, head_l, base_overlay1)
                self.camera2.set_frame(frame2, raw2, head_r, base_overlay2)
                self.laser_overlay.set_position(None)
                self.kalman.reset()
                self.status.setText("Could not compute valid 3D index finger direction.")
                return

            p_intersect = calculate_intersection(
                p_tip, v_hat, self.projector_plane.plane_coeffs
            )

            overlay1 = build_camera_overlay_3d(
                self.projector_plane,
                frame_calib,
                is_right_camera=False,
                p_tip=p_tip,
                v_hat=v_hat,
                p_intersect=p_intersect,
            )
            overlay2 = build_camera_overlay_3d(
                self.projector_plane,
                frame_calib,
                is_right_camera=True,
                p_tip=p_tip,
                v_hat=v_hat,
                p_intersect=p_intersect,
            )
            self.camera1.set_frame(frame1, raw1, head_l, overlay1)
            self.camera2.set_frame(frame2, raw2, head_r, overlay2)

            if p_intersect is None:
                self.laser_overlay.set_position(None)
                self.kalman.reset()
                self.status.setText("Pointing ray does not intersect the calibrated projector plane.")
                return

            resolution = (self.screen_geometry.width(), self.screen_geometry.height())
            raw_px = to_screen_pixels(
                p_intersect,
                self.projector_plane.p_origin,
                self.projector_plane.u_axis,
                self.projector_plane.v_axis,
                self.projector_plane.dimensions,
                resolution,
                self.projector_plane.plane_to_pixel_homography,
                self.projector_plane.projector_resolution,
            )
            filtered_px = self.kalman.update(*raw_px)

            overlay_geom = self.laser_overlay.geometry()
            self.laser_overlay.set_position(
                (
                    filtered_px[0] + self.screen_geometry.x() - overlay_geom.x(),
                    filtered_px[1] + self.screen_geometry.y() - overlay_geom.y(),
                )
            )
            self.status.setText(
                f"Pointer: ({filtered_px[0]}, {filtered_px[1]}) px | "
                f"3D Tip: ({p_tip[0]:.1f}, {p_tip[1]:.1f}, {p_tip[2]:.1f}) | "
                f"Processing: {self._processing_fps:.1f} FPS"
            )
        except (ValueError, cv2.error, np.linalg.LinAlgError) as err:
            self.camera1.set_frame(frame1, raw1, head_l, base_overlay1)
            self.camera2.set_frame(frame2, raw2, head_r, base_overlay2)
            self.laser_overlay.set_position(None)
            self.kalman.reset()
            self.status.setText(f"Tracking/projection error: {err}")

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.capture.close()
        if self.tracker is not None:
            self.tracker.close()
        self.laser_overlay.close()
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stereo Virtual Pointer")
    parser.add_argument(
        "--settings",
        type=Path,
        default=SETTINGS_FILE,
        help="Camera settings JSON (default: ./camera_settings.json).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=CALIB_PARAMS_FILE,
        help="Stereo calibration JSON (default: ./calibration_params.json).",
    )
    parser.add_argument(
        "--projector",
        type=Path,
        default=PROJECTOR_CALIB_FILE,
        help="Projector calibration JSON (default: ./projector_calibration.json).",
    )
    parser.add_argument(
        "--screen",
        type=int,
        default=None,
        help="Target display index for the projector overlay (default: secondary display if present, else 0).",
    )
    args = parser.parse_args()

    print("Starting Qt application...", flush=True)
    application = QApplication([sys.argv[0]])
    screens = application.screens()
    if not screens:
        raise RuntimeError("No display is available for the virtual pointer overlay.")

    if args.screen is None:
        target_screen_idx = 1 if len(screens) > 1 else 0
    else:
        if not 0 <= args.screen < len(screens):
            parser.error(f"Screen index must be between 0 and {len(screens) - 1}.")
        target_screen_idx = args.screen

    screen = screens[target_screen_idx]
    desktop_geometry = screens[0].geometry()
    for other_screen in screens[1:]:
        desktop_geometry = desktop_geometry.united(other_screen.geometry())

    cam_settings = load_camera_settings(args.settings)

    calibration: Optional[StereoCalibration] = None
    projector_plane: Optional[ProjectorPlane] = None
    try:
        if args.calibration.exists():
            calibration = load_stereo_calibration(args.calibration)
    except Exception as exc:
        print(f"Warning loading {args.calibration}: {exc}", flush=True)

    try:
        if args.projector.exists():
            projector_plane = load_projector_calibration(args.projector)
    except Exception as exc:
        print(f"Warning loading {args.projector}: {exc}", flush=True)

    tracker: Optional[HandTrackerBackend] = None
    tracker_error: Optional[str] = None
    try:
        print("Initializing hand tracker...", flush=True)
        tracker = HandTrackerBackend(stream_count=2)
    except Exception as exc:
        tracker_error = str(exc)
        print(f"Hand tracker error: {tracker_error}", flush=True)

    laser_overlay = LaserOverlay()
    laser_overlay.setGeometry(desktop_geometry)
    laser_overlay.show()

    window = VirtualPointerWindow(
        cam_settings=cam_settings,
        calibration=calibration,
        projector_plane=projector_plane,
        screen=screen,
        laser_overlay=laser_overlay,
        tracker=tracker,
        tracker_error=tracker_error,
    )
    window.show()

    exit_code = application.exec()
    if tracker is not None:
        tracker.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())