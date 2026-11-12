"""Capture checkerboard pairs and calibrate two fixed cameras in millimetres."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def camera_source(value):
    try:
        return int(value)
    except ValueError:
        return value


def checkerboard_corners(frame, pattern):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern)
    else:
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
    return corners.astype(np.float32) if found else None


def capture_pairs(args):
    left_dir, right_dir = args.images / "left", args.images / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    if list(left_dir.glob("*.png")) or list(right_dir.glob("*.png")):
        raise ValueError("Output image folders must be empty to avoid mixing capture sessions.")
    cameras = [cv2.VideoCapture(camera_source(v)) for v in (args.camera1, args.camera2)]
    try:
        if not all(camera.isOpened() for camera in cameras):
            raise OSError("Could not open both cameras.")
        saved = 0
        while saved < args.count:
            if not all(camera.grab() for camera in cameras):
                raise OSError("A camera stopped supplying frames.")
            results = [camera.retrieve() for camera in cameras]
            if not all(success and frame is not None for success, frame in results):
                raise OSError("Could not decode both frames.")
            frames = [frame for _, frame in results]
            corners = [checkerboard_corners(frame, args.pattern) for frame in frames]
            previews = []
            for frame, points in zip(frames, corners):
                preview = frame.copy()
                if points is not None:
                    cv2.drawChessboardCorners(preview, args.pattern, points, True)
                previews.append(preview)
            height = min(preview.shape[0] for preview in previews)
            previews = [cv2.resize(p, (round(p.shape[1] * height / p.shape[0]), height)) for p in previews]
            combined = cv2.hconcat(previews)
            cv2.putText(combined, f"SPACE: save pair ({saved}/{args.count})  Q: quit",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Stereo checkerboard capture", combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32 and all(points is not None for points in corners):
                name = f"{saved + 1:04d}.png"
                paths = (left_dir / name, right_dir / name)
                if not all(cv2.imwrite(str(path), frame) for path, frame in zip(paths, frames)):
                    raise OSError("Could not save the image pair.")
                saved += 1
                print(f"Saved pair {saved}/{args.count}")
        print(f"Captured {saved} pairs in {args.images}")
    finally:
        for camera in cameras:
            camera.release()
        cv2.destroyAllWindows()


def calibrate(args):
    left = {path.stem: path for path in (args.images / "left").glob("*.png")}
    right = {path.stem: path for path in (args.images / "right").glob("*.png")}
    if left.keys() != right.keys():
        raise ValueError("Left and right image filenames do not match.")
    if len(left) < 10:
        raise ValueError("At least 10 paired checkerboard images are required.")

    cols, rows = args.pattern
    board_points = np.zeros((cols * rows, 3), np.float32)
    board_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square_mm
    object_points, image_points1, image_points2 = [], [], []
    image_size = None
    rejected = []
    for name in sorted(left):
        frames = [cv2.imread(str(path)) for path in (left[name], right[name])]
        if any(frame is None for frame in frames):
            raise ValueError(f"Could not read image pair {name}.")
        sizes = [(frame.shape[1], frame.shape[0]) for frame in frames]
        if sizes[0] != sizes[1]:
            raise ValueError(f"Pair {name} has different camera image sizes.")
        if image_size is None:
            image_size = sizes[0]
        if sizes[0] != image_size:
            raise ValueError(f"Pair {name} differs in resolution from the other pairs.")
        corners = [checkerboard_corners(frame, args.pattern) for frame in frames]
        if any(points is None for points in corners):
            rejected.append(name)
            continue
        object_points.append(board_points.copy())
        image_points1.append(corners[0])
        image_points2.append(corners[1])
    if len(object_points) < 10:
        raise ValueError(f"Only {len(object_points)} usable pairs; at least 10 are required.")

    rms1, K1, D1, _, _ = cv2.calibrateCamera(object_points, image_points1, image_size, None, None)
    rms2, K2, D2, _, _ = cv2.calibrateCamera(object_points, image_points2, image_size, None, None)
    stereo_rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        object_points, image_points1, image_points2, K1, D1, K2, D2, image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    values = [rms1, rms2, stereo_rms, *K1.flat, *K2.flat, *D1.flat, *D2.flat, *R.flat, *T.flat]
    if not np.isfinite(values).all() or np.linalg.norm(T) < 1e-6:
        raise ValueError("Calibration produced invalid camera parameters.")
    data = {
        "calibration_type": "measured",
        "image_size": list(image_size),
        "image_size1": list(image_size),
        "image_size2": list(image_size),
        "K1": K1.tolist(), "D1": D1.ravel().tolist(),
        "K2": K2.tolist(), "D2": D2.ravel().tolist(),
        "R": R.tolist(), "T": T.ravel().tolist(),
        "square_size_mm": args.square_mm,
        "checkerboard_inner_corners": [cols, rows],
        "usable_pairs": len(object_points),
        "rms_pixels": {"camera1": rms1, "camera2": rms2, "stereo": stereo_rms},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}: {len(object_points)} pairs, stereo RMS {stereo_rms:.3f} px")
    if rejected:
        print(f"Checkerboard not found in {len(rejected)} pairs: {', '.join(rejected)}")
    if stereo_rms > 1.0:
        print("Warning: high reprojection error; recapture varied, sharp board poses.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("capture", "calibrate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--images", type=Path, default=Path("calibration_images"))
        sub.add_argument("--cols", type=int, default=9, help="Checkerboard inner corners horizontally.")
        sub.add_argument("--rows", type=int, default=6, help="Checkerboard inner corners vertically.")
        sub.add_argument("--square-mm", type=float, default=25.0)
        if command == "capture":
            sub.add_argument("--camera1", default="0")
            sub.add_argument("--camera2", default="1")
            sub.add_argument("--count", type=int, default=25)
        else:
            sub.add_argument("--output", type=Path, default=Path("calibration_params.json"))
    args = parser.parse_args()
    if min(args.cols, args.rows) < 2 or args.square_mm <= 0 or (args.command == "capture" and args.count < 10):
        parser.error("Use at least 2 inner corners per side, positive square size, and at least 10 pairs.")
    args.pattern = (args.cols, args.rows)
    try:
        if args.command == "capture":
            capture_pairs(args)
        else:
            calibrate(args)
    except (OSError, ValueError, cv2.error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
