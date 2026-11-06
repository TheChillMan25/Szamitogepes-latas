import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QRadialGradient
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

IMAGE_SIZE = (1280, 720)
MOCK_BOARD_SIZE = (600.0, 400.0)
MOCK_BOARD_DEPTH = 1000.0
CAMERA_FRAME_SIZE = (640, 360)
BOARD_MARKER_MARGIN_MM = 20.0


@dataclass(frozen=True)
class StereoCalibration:
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    image_size: tuple
    image_size1: tuple
    image_size2: tuple
    P1: np.ndarray
    P2: np.ndarray


def _matrix(data, name, shape):
    value = np.asarray(data[name], dtype=np.float64)
    if value.size != shape[0] * shape[1]:
        raise ValueError(f"{name} must contain {shape[0] * shape[1]} values.")
    value = value.reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values.")
    return value


def load_stereo_calibration(config_path):
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as calibration_file:
        data = json.load(calibration_file)

    K1 = _matrix(data, "K1", (3, 3))
    K2 = _matrix(data, "K2", (3, 3))
    R = _matrix(data, "R", (3, 3))
    T = _matrix(data, "T", (3, 1))
    D1 = np.asarray(data["D1"], dtype=np.float64).reshape(-1, 1)
    D2 = np.asarray(data["D2"], dtype=np.float64).reshape(-1, 1)
    image_size = tuple(int(value) for value in data.get("image_size", IMAGE_SIZE))
    image_size1 = tuple(int(value) for value in data.get("image_size1", image_size))
    image_size2 = tuple(int(value) for value in data.get("image_size2", image_size))
    for name, size in (
        ("image_size", image_size),
        ("image_size1", image_size1),
        ("image_size2", image_size2),
    ):
        if len(size) != 2 or min(size) <= 0:
            raise ValueError(f"{name} must contain positive width and height.")
    if not np.isfinite(D1).all() or not np.isfinite(D2).all():
        raise ValueError("Distortion coefficients must be finite.")
    if K1[0, 0] <= 0 or K1[1, 1] <= 0 or K2[0, 0] <= 0 or K2[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive.")

    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T))
    return StereoCalibration(
        K1, D1, K2, D2, R, T, image_size, image_size1, image_size2, P1, P2
    )


def mock_stereo_calibration():
    width, height = IMAGE_SIZE
    K1 = np.array(
        [[1000.0, 0.0, width / 2], [0.0, 1000.0, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    K2 = K1.copy()
    T = np.array([[-150.0], [0.0], [0.0]], dtype=np.float64)
    zero_distortion = np.zeros((5, 1), dtype=np.float64)
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((np.eye(3), T))
    return StereoCalibration(
        K1,
        zero_distortion,
        K2,
        zero_distortion.copy(),
        np.eye(3),
        T,
        IMAGE_SIZE,
        IMAGE_SIZE,
        IMAGE_SIZE,
        P1,
        P2,
    )


def calibration_for_frame_sizes(calibration, size1, size2):
    """Scale each camera's intrinsics from calibration pixels to current frame pixels."""
    def scale_intrinsics(matrix, source_size, size):
        width0, height0 = source_size
        width, height = size
        scale = np.diag([width / width0, height / height0, 1.0])
        return scale @ matrix

    K1 = scale_intrinsics(calibration.K1, calibration.image_size1, size1)
    K2 = scale_intrinsics(calibration.K2, calibration.image_size2, size2)
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((calibration.R, calibration.T))
    return StereoCalibration(
        K1,
        calibration.D1,
        K2,
        calibration.D2,
        calibration.R,
        calibration.T,
        calibration.image_size,
        calibration.image_size1,
        calibration.image_size2,
        P1,
        P2,
    )


class ArucoBoardPoseEstimator:
    marker_ids = (0, 1, 2, 3)

    def __init__(self, board_size, marker_size):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "ArUco support is missing; install opencv-contrib-python."
            )
        if marker_size <= 0 or marker_size >= min(board_size):
            raise ValueError("Marker size must be positive and smaller than the board.")

        self.board_width, self.board_height = board_size
        self.marker_size = marker_size
        aruco = cv2.aruco
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        if hasattr(aruco, "ArucoDetector"):
            self.detector = aruco.ArucoDetector(
                self.dictionary, aruco.DetectorParameters()
            )
        else:
            self.detector = None
            self.parameters = aruco.DetectorParameters_create()

        margin = BOARD_MARKER_MARGIN_MM
        if marker_size + 2 * margin >= min(board_size):
            raise ValueError("Board is too small for the four corner markers.")
        marker_origins = (
            (margin, margin),
            (self.board_width - margin - marker_size, margin),
            (
                self.board_width - margin - marker_size,
                self.board_height - margin - marker_size,
            ),
            (margin, self.board_height - margin - marker_size),
        )
        marker_corners = []
        for marker_x, marker_y in marker_origins:
            marker_corners.extend(
                [
                    [marker_x, marker_y, 0.0],
                    [marker_x + marker_size, marker_y, 0.0],
                    [marker_x + marker_size, marker_y + marker_size, 0.0],
                    [marker_x, marker_y + marker_size, 0.0],
                ]
            )
        self.marker_corners_board = np.asarray(marker_corners, dtype=np.float64)

    def estimate(self, frame, camera_matrix, distortion):
        aruco = cv2.aruco
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, marker_ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, marker_ids, _ = aruco.detectMarkers(
                gray, self.dictionary, parameters=self.parameters
            )

        if marker_ids is None:
            return None
        image_points = []
        object_points = []
        for detected_corners, detected_id in zip(corners, marker_ids.ravel()):
            if detected_id not in self.marker_ids:
                continue
            start = self.marker_ids.index(int(detected_id)) * 4
            object_points.extend(self.marker_corners_board[start : start + 4])
            image_points.extend(np.asarray(detected_corners).reshape(4, 2))
        if len(object_points) < 4:
            return None

        try:
            success, rotation_vector, translation = cv2.solvePnP(
                np.asarray(object_points, dtype=np.float64),
                np.asarray(image_points, dtype=np.float64),
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None
        if not success:
            return None

        return board_geometry_from_pose(
            rotation_vector, translation, self.board_width, self.board_height
        )


def write_board_marker(path, board_size, marker_size):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("ArUco support is missing; install opencv-contrib-python.")
    if marker_size <= 0 or marker_size + 2 * BOARD_MARKER_MARGIN_MM >= min(board_size):
        raise ValueError("Board is too small for four markers with 20 mm margins.")
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    pixels_per_mm = 3
    width = round(board_size[0] * pixels_per_mm)
    height = round(board_size[1] * pixels_per_mm)
    marker_pixels = round(marker_size * pixels_per_mm)
    image = np.full((height, width), 255, dtype=np.uint8)
    margin_pixels = round(BOARD_MARKER_MARGIN_MM * pixels_per_mm)
    origins = (
        (margin_pixels, margin_pixels),
        (width - margin_pixels - marker_pixels, margin_pixels),
        (width - margin_pixels - marker_pixels, height - margin_pixels - marker_pixels),
        (margin_pixels, height - margin_pixels - marker_pixels),
    )
    for marker_id, (x, y) in enumerate(origins):
        if hasattr(aruco, "generateImageMarker"):
            marker = aruco.generateImageMarker(dictionary, marker_id, 600)
        else:
            marker = aruco.drawMarker(dictionary, marker_id, 600)
        marker = cv2.resize(
            marker, (marker_pixels, marker_pixels), interpolation=cv2.INTER_NEAREST
        )
        image[y : y + marker_pixels, x : x + marker_pixels] = marker
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write marker image to {path}.")


class StereoCameras:
    def __init__(self, source1, source2):
        self.captures = []
        for source in (source1, source2):
            capture = cv2.VideoCapture(source, cv2.CAP_ANY)
            if not capture.isOpened():
                capture.release()
                self.release()
                raise OSError(f"Could not open camera source {source!r}.")
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.captures.append(capture)

    def read(self):
        frames = []
        for index, capture in enumerate(self.captures, start=1):
            success, frame = capture.read()
            if not success or frame is None:
                return None, None, f"Camera {index} did not provide a frame."
            frames.append(frame)
        return frames[0], frames[1], None

    def release(self):
        for capture in self.captures:
            capture.release()
        self.captures.clear()


def create_hands_detector():
    try:
        import mediapipe as mp
    except ImportError as error:
        return None, f"MediaPipe unavailable: {error}"

    hands_api = getattr(getattr(mp, "solutions", None), "hands", None)
    if hands_api is None:
        return (
            None,
            "MediaPipe Hands API is unavailable; live cameras will show without tracking.",
        )
    try:
        detector = hands_api.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
    except (AttributeError, RuntimeError, ValueError) as error:
        return None, f"Could not initialize MediaPipe Hands: {error}"
    return detector, None


def detect_finger_landmarks(frame1, frame2, calibration, detectors):
    if detectors[0] is None or detectors[1] is None:
        return None, None

    def detect(frame, detector):
        height, width = frame.shape[:2]
        result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return None
        hand = result.multi_hand_landmarks[0]
        names = (
            "INDEX_FINGER_MCP",
            "INDEX_FINGER_PIP",
            "INDEX_FINGER_DIP",
            "INDEX_FINGER_TIP",
        )
        indices = (5, 6, 7, 8)
        pixels = {
            name: np.array(
                [hand.landmark[index].x * width, hand.landmark[index].y * height],
                dtype=np.float32,
            )
            for name, index in zip(names, indices)
        }
        return pixels

    landmarks1 = detect(frame1, detectors[0])
    landmarks2 = detect(frame2, detectors[1])
    if landmarks1 is None or landmarks2 is None:
        return None, None

    return (
        _undistort_landmarks(landmarks1, calibration.K1, calibration.D1),
        _undistort_landmarks(landmarks2, calibration.K2, calibration.D2),
    )


def compute_board_geometry(corners_2d, width, height, K1, D1):
    corners_3d = np.array(
        [[0.0, 0.0, 0.0], [width, 0.0, 0.0], [width, height, 0.0], [0.0, height, 0.0]],
        dtype=np.float64,
    )
    corners_2d = np.asarray(corners_2d, dtype=np.float64).reshape(4, 2)
    undistorted = cv2.undistortPoints(
        corners_2d.reshape(-1, 1, 2), K1, D1
    ).reshape(-1, 2)
    success, rvec, tvec = cv2.solvePnP(
        corners_3d,
        undistorted,
        np.eye(3),
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise ValueError("Could not estimate the mock board pose.")

    return board_geometry_from_pose(rvec, tvec, width, height)


def board_geometry_from_pose(rotation_vector, translation, width, height):
    board_rotation, _ = cv2.Rodrigues(rotation_vector)
    corners_3d = np.array(
        [[0.0, 0.0, 0.0], [width, 0.0, 0.0], [width, height, 0.0], [0.0, height, 0.0]],
        dtype=np.float64,
    )
    board_corners_cam = (board_rotation @ corners_3d.T + translation).T
    origin = board_corners_cam[0]
    u_axis = board_corners_cam[1] - origin
    v_axis = board_corners_cam[3] - origin
    u_axis /= np.linalg.norm(u_axis)
    v_axis /= np.linalg.norm(v_axis)
    normal = np.cross(u_axis, v_axis)
    normal /= np.linalg.norm(normal)
    plane = (*normal, -float(np.dot(normal, origin)))
    return origin, u_axis, v_axis, plane


def triangulate_landmarks(landmarks_cam1, landmarks_cam2, P1, P2):
    if landmarks_cam1.keys() != landmarks_cam2.keys():
        raise ValueError("The two cameras must provide matching landmarks.")

    points_3d = {}
    for name, point1 in landmarks_cam1.items():
        point1 = np.asarray(point1, dtype=np.float64).reshape(2)
        point2 = np.asarray(landmarks_cam2[name], dtype=np.float64).reshape(2)
        points_4d = cv2.triangulatePoints(
            P1,
            P2,
            point1.astype(np.float32).reshape(2, 1),
            point2.astype(np.float32).reshape(2, 1),
        )
        if abs(float(points_4d[3, 0])) < 1e-8:
            raise ValueError(f"Could not triangulate landmark {name}.")
        point_3d = points_4d[:3, 0] / points_4d[3, 0]
        if not np.isfinite(point_3d).all():
            raise ValueError(f"Triangulated landmark {name} is not finite.")
        points_3d[name] = point_3d
    return points_3d


def calculate_pointing_direction(points_3d):
    if "INDEX_FINGER_DIP" not in points_3d or "INDEX_FINGER_TIP" not in points_3d:
        return None, None
    point_dip = points_3d["INDEX_FINGER_DIP"]
    point_tip = points_3d["INDEX_FINGER_TIP"]
    direction = point_tip - point_dip
    length = np.linalg.norm(direction)
    if length < 1e-5:
        return None, None
    return point_tip, direction / length


def calculate_intersection(point, direction, plane):
    normal = np.asarray(plane[:3], dtype=np.float64)
    denominator = float(np.dot(normal, direction))
    if abs(denominator) < 1e-6:
        return None
    distance = -(
        float(np.dot(normal, point)) + float(plane[3])
    ) / denominator
    if distance < 0:
        return None
    return point + distance * direction


def to_screen_pixels(point, origin, u_axis, v_axis, dimensions, resolution):
    board_width, board_height = dimensions
    screen_width, screen_height = resolution
    offset = point - origin
    x = float(np.dot(offset, u_axis)) / board_width
    y = float(np.dot(offset, v_axis)) / board_height
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    return round(x * (screen_width - 1)), round(y * (screen_height - 1))


def _undistort_landmarks(landmarks, camera_matrix, distortion):
    corrected = {}
    for name, pixel in landmarks.items():
        point = np.asarray(pixel, dtype=np.float32).reshape(1, 1, 2)
        corrected[name] = cv2.undistortPoints(
            point, camera_matrix, distortion, P=camera_matrix
        )[0, 0]
    return corrected


def _project_points(points, rotation, translation, camera_matrix, distortion):
    projected, _ = cv2.projectPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 3),
        rotation,
        translation,
        camera_matrix,
        distortion,
    )
    return projected.reshape(-1, 2)


class MockStereoScene:
    """Generate synchronized camera images and matching projected finger landmarks."""

    landmark_names = (
        "INDEX_FINGER_MCP",
        "INDEX_FINGER_PIP",
        "INDEX_FINGER_DIP",
        "INDEX_FINGER_TIP",
    )

    def __init__(self, calibration, board_size=MOCK_BOARD_SIZE):
        self.calibration = calibration
        self.board_size = board_size
        self.frame_count = 0
        half_width = self.board_size[0] / 2
        half_height = self.board_size[1] / 2
        self.board_corners = np.array(
            [
                [-half_width, -half_height, MOCK_BOARD_DEPTH],
                [half_width, -half_height, MOCK_BOARD_DEPTH],
                [half_width, half_height, MOCK_BOARD_DEPTH],
                [-half_width, half_height, MOCK_BOARD_DEPTH],
            ],
            dtype=np.float64,
        )
        self.board_image_corners = _project_points(
            self.board_corners,
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            calibration.K1,
            calibration.D1,
        )
        self.board_geometry = compute_board_geometry(
            self.board_image_corners,
            *self.board_size,
            calibration.K1,
            calibration.D1,
        )

    def _finger_points(self):
        phase = self.frame_count * 0.035
        half_width, half_height = self.board_size[0] / 2, self.board_size[1] / 2
        target_x = np.sin(phase) * half_width * 0.82
        target_y = np.cos(phase * 0.73) * half_height * 0.78
        return np.array(
            [
                [target_x, target_y, MOCK_BOARD_DEPTH - 260.0],
                [target_x, target_y, MOCK_BOARD_DEPTH - 200.0],
                [target_x, target_y, MOCK_BOARD_DEPTH - 150.0],
                [target_x, target_y, MOCK_BOARD_DEPTH - 100.0],
            ],
            dtype=np.float64,
        )

    def _render_camera(self, landmarks, camera_label, board_corners):
        width, height = self.calibration.image_size
        frame = np.full((height, width, 3), (30, 34, 40), dtype=np.uint8)
        board_pixels = np.rint(board_corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [board_pixels], True, (75, 85, 100), 3)
        for pixel in landmarks.values():
            cv2.circle(frame, tuple(np.rint(pixel).astype(int)), 10, (70, 200, 255), -1)
        points = np.rint(list(landmarks.values())).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [points], False, (80, 190, 255), 5)
        cv2.putText(
            frame,
            f"{camera_label}  |  MOCK INPUT",
            (32, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (235, 240, 245),
            2,
            cv2.LINE_AA,
        )
        preview_width, preview_height = CAMERA_FRAME_SIZE
        return cv2.resize(
            frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA
        )

    def next_frames(self):
        self.frame_count += 1
        points_cam1 = self._finger_points()
        points_cam2 = (self.calibration.R @ points_cam1.T + self.calibration.T).T

        image_points1 = _project_points(
            points_cam1,
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            self.calibration.K1,
            self.calibration.D1,
        )
        image_points2 = _project_points(
            points_cam2,
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            self.calibration.K2,
            self.calibration.D2,
        )
        board_points2 = _project_points(
            (self.calibration.R @ self.board_corners.T + self.calibration.T).T,
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            self.calibration.K2,
            self.calibration.D2,
        )
        landmarks1 = dict(zip(self.landmark_names, image_points1))
        landmarks2 = dict(zip(self.landmark_names, image_points2))
        frame1 = self._render_camera(landmarks1, "CAMERA 1", self.board_image_corners)
        frame2 = self._render_camera(landmarks2, "CAMERA 2", board_points2)
        return frame1, frame2, landmarks1, landmarks2


class KalmanFilter2D:
    def __init__(self):
        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.filter.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.filter.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.initialized = False

    def update(self, x, y):
        measurement = np.array([[x], [y]], dtype=np.float32)
        if not self.initialized:
            self.filter.statePre[:2, 0] = measurement[:, 0]
            self.filter.statePost[:2, 0] = measurement[:, 0]
            self.initialized = True
        self.filter.predict()
        corrected = self.filter.correct(measurement)
        return round(float(corrected[0, 0])), round(float(corrected[1, 0]))


class CameraPanel(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        heading = QLabel(title)
        heading.setStyleSheet("font-weight: 600; color: #d7e0ea;")
        self.image = QLabel()
        self.image.setFixedSize(*CAMERA_FRAME_SIZE)
        self.image.setAlignment(Qt.AlignCenter)
        layout.addWidget(heading)
        layout.addWidget(self.image)

    def set_frame(self, frame):
        height, width, _ = frame.shape
        image = QImage(
            frame.data, width, height, frame.strides[0], QImage.Format_BGR888
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
        self.position = None

    def set_position(self, position):
        self.position = position
        self.update()

    def paintEvent(self, event):
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


class VirtualPointerWindow(QMainWindow):
    def __init__(
        self,
        calibration,
        calibration_label,
        screen,
        laser_overlay,
        cameras,
        hands_detectors,
        input_status,
        board_size,
        marker_size,
        board_pose_estimator,
    ):
        super().__init__()
        self.calibration = calibration
        self.board_size = board_size
        self.marker_size = marker_size
        self.scene = MockStereoScene(calibration, board_size)
        self.cameras = cameras
        self.hands_detectors = hands_detectors
        self.board_pose_estimator = board_pose_estimator
        self.input_status = input_status
        self.kalman = KalmanFilter2D()
        self.board_geometry = self.scene.board_geometry
        self.screen_geometry = screen.geometry()
        self.laser_overlay = laser_overlay
        self.setWindowTitle("Virtual Pointer | Stereo Demo")
        self.setStyleSheet(
            "QMainWindow, QWidget { background: #171d26; color: #eef2f6; }"
            "QFrame { background: #202936; border: 1px solid #354354; border-radius: 6px; }"
        )

        central = QWidget()
        layout = QVBoxLayout(central)
        panels = QHBoxLayout()
        self.camera1 = CameraPanel("Camera 1")
        self.camera2 = CameraPanel("Camera 2")
        panels.addWidget(self.camera1)
        panels.addWidget(self.camera2)
        layout.addLayout(panels)

        self.status = QLabel(
            f"{input_status}  |  {calibration_label}  |  "
            f"Board: {board_size[0]:.0f} x {board_size[1]:.0f} mm  |  "
            f"Corner DICT_4X4_50 IDs 0-3, {marker_size:g} mm"
        )
        self.status.setStyleSheet("color: #a9bacb; padding: 6px;")
        layout.addWidget(self.status)
        self.setCentralWidget(central)
        self.setMinimumSize(900, 440)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_demo)
        self.timer.start(33)

    def update_demo(self):
        if self.cameras is None:
            frame1, frame2, raw1, raw2 = self.scene.next_frames()
            frame_calibration = self.calibration
        else:
            frame1, frame2, error = self.cameras.read()
            if error:
                self.laser_overlay.set_position(None)
                self.status.setText(f"LIVE CAMERA ERROR  |  {error}")
                return
            size1 = (frame1.shape[1], frame1.shape[0])
            size2 = (frame2.shape[1], frame2.shape[0])
            frame_calibration = calibration_for_frame_sizes(
                self.calibration, size1, size2
            )
            detected_board = self.board_pose_estimator.estimate(
                frame1, frame_calibration.K1, frame_calibration.D1
            )
            if detected_board is None:
                self.camera1.set_frame(frame1)
                self.camera2.set_frame(frame2)
                self.laser_overlay.set_position(None)
                self.status.setText(
                    f"LIVE CAMERAS  |  {self.input_status}  |  "
                    f"{size1[0]}x{size1[1]} + {size2[0]}x{size2[1]}  |  "
                    "No board markers (IDs 0-3) visible in Camera 1."
                )
                return
            self.board_geometry = detected_board
            raw1, raw2 = detect_finger_landmarks(
                frame1, frame2, frame_calibration, self.hands_detectors
            )
            if raw1 is None or raw2 is None:
                self.camera1.set_frame(frame1)
                self.camera2.set_frame(frame2)
                self.laser_overlay.set_position(None)
                if self.hands_detectors[0] is None or self.hands_detectors[1] is None:
                    self.status.setText(
                        f"LIVE CAMERAS  |  {self.input_status}  |  "
                        "Frames received; hand tracking unavailable."
                    )
                else:
                    self.status.setText(
                        f"LIVE CAMERAS  |  {self.input_status}  |  "
                        f"{size1[0]}x{size1[1]} + {size2[0]}x{size2[1]}  |  "
                        "Show one hand to both cameras."
                    )
                return

        self.camera1.set_frame(frame1)
        self.camera2.set_frame(frame2)
        try:
            if self.cameras is None:
                landmarks1 = _undistort_landmarks(
                    raw1, frame_calibration.K1, frame_calibration.D1
                )
                landmarks2 = _undistort_landmarks(
                    raw2, frame_calibration.K2, frame_calibration.D2
                )
            else:
                landmarks1, landmarks2 = raw1, raw2
            points_3d = triangulate_landmarks(
                landmarks1, landmarks2, frame_calibration.P1, frame_calibration.P2
            )
            point_tip, direction = calculate_pointing_direction(points_3d)
            if point_tip is None or direction is None:
                self.laser_overlay.set_position(None)
                self.status.setText(
                    f"{self.input_status}  |  No valid index-finger direction."
                )
                return
            intersection = calculate_intersection(
                point_tip, direction, self.board_geometry[3]
            )
            if intersection is None:
                self.laser_overlay.set_position(None)
                self.status.setText(
                    f"{self.input_status}  |  Finger ray does not meet the board."
                )
                return
            resolution = (self.screen_geometry.width(), self.screen_geometry.height())
            raw_position = to_screen_pixels(
                intersection,
                self.board_geometry[0],
                self.board_geometry[1],
                self.board_geometry[2],
                self.board_size,
                resolution,
            )
            position = self.kalman.update(*raw_position)
            self.laser_overlay.set_position(position)
            self.status.setText(
                f"{self.input_status}  |  {position[0]}, {position[1]} px  |  "
                f"Laser overlay on {self.screen_geometry.width()} x "
                f"{self.screen_geometry.height()} desktop  |  "
                f"{frame1.shape[1]}x{frame1.shape[0]} + "
                f"{frame2.shape[1]}x{frame2.shape[0]} cameras  |  "
                f"{'Mock board pose' if self.cameras is None else 'ArUco board pose from Camera 1'}"
            )
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            self.laser_overlay.set_position(None)
            self.status.setText(f"Stereo processing error: {error}")


def main():
    parser = argparse.ArgumentParser(description="Stereo virtual pointer")
    parser.add_argument(
        "--camera1",
        default="0",
        help="First OpenCV camera index or stream URL (default: 0).",
    )
    parser.add_argument(
        "--camera2",
        default="1",
        help="Second OpenCV camera index or stream URL (default: 1).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic camera feeds instead of live cameras.",
    )
    parser.add_argument(
        "--board-width-mm",
        type=float,
        default=600.0,
        help="Physical board width in mm (default: 600).",
    )
    parser.add_argument(
        "--board-height-mm",
        type=float,
        default=400.0,
        help="Physical board height in mm (default: 400).",
    )
    parser.add_argument(
        "--marker-size-mm",
        type=float,
        default=80.0,
        help="Printed black ArUco marker size, excluding white margin (default: 80).",
    )
    parser.add_argument(
        "--generate-board",
        "--generate-board-marker",
        dest="generate_board",
        metavar="PNG",
        help="Write the printable four-marker DICT_4X4_50 board image and exit.",
    )
    args = parser.parse_args()
    board_size = (args.board_width_mm, args.board_height_mm)
    if min(*board_size, args.marker_size_mm) <= 0:
        parser.error("Board and marker dimensions must be positive.")
    if args.marker_size_mm + 2 * BOARD_MARKER_MARGIN_MM >= min(board_size):
        parser.error("Four markers need at least 20 mm of board margin on each side.")
    if args.generate_board:
        try:
            write_board_marker(
                args.generate_board, board_size, args.marker_size_mm
            )
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        return 0

    def camera_source(value):
        try:
            return int(value)
        except ValueError:
            return value

    application = QApplication([sys.argv[0]])
    screen = application.primaryScreen()
    if screen is None:
        raise RuntimeError("No display is available for the laser-pointer overlay.")
    calibration_path = Path(__file__).with_name("calibration_params.json")
    try:
        calibration = load_stereo_calibration(calibration_path)
        calibration_label = f"Calibration: {calibration_path.name}"
    except FileNotFoundError:
        calibration = mock_stereo_calibration()
        calibration_label = "Built-in mock calibration (file not found)"

    cameras = None
    hands_detectors = (None, None)
    board_pose_estimator = None
    if args.mock:
        input_status = "MOCK CAMERAS (requested)"
    else:
        try:
            cameras = StereoCameras(
                camera_source(args.camera1), camera_source(args.camera2)
            )
            board_pose_estimator = ArucoBoardPoseEstimator(
                board_size, args.marker_size_mm
            )
            detector1, detector_error1 = create_hands_detector()
            detector2, detector_error2 = create_hands_detector()
            hands_detectors = (detector1, detector2)
            input_status = f"LIVE CAMERAS ({args.camera1}, {args.camera2})"
            detector_errors = [
                message for message in (detector_error1, detector_error2) if message
            ]
            if detector_errors:
                input_status += "  |  " + "; ".join(dict.fromkeys(detector_errors))
        except (OSError, RuntimeError, ValueError) as error:
            if cameras is not None:
                cameras.release()
                cameras = None
            input_status = f"MOCK CAMERAS (live open failed: {error})"

    laser_overlay = LaserOverlay()
    laser_overlay.setGeometry(screen.geometry())
    laser_overlay.show()
    window = VirtualPointerWindow(
        calibration,
        calibration_label,
        screen,
        laser_overlay,
        cameras,
        hands_detectors,
        input_status,
        board_size,
        args.marker_size_mm,
        board_pose_estimator,
    )
    if cameras is not None:
        application.aboutToQuit.connect(cameras.release)
    for detector in hands_detectors:
        if detector is not None:
            application.aboutToQuit.connect(detector.close)
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
