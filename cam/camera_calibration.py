"""Reusable Sinoseen camera-calibration library (single Python file).

Place this file beside the Python program that uses it. There is no need to
install *this library* as a package. Third-party requirements are OpenCV,
NumPy and cv2-enumerate-cameras.

Example
-------
    import camera_calibration
    values = camera_calibration.calib(1, "12x12", 20)

`calib()` runs one calibration, displays/saves the 5x3 result matrix, then
asks whether another calibration is needed. If yes, enter the next three
parameters on one line, e.g. 2,15x15,30, or press Enter to repeat them.
`calibration()` remains available for a SINGLE run without any repeat prompt.

Inputs:
    camera:           1 = Sinoseen USB 2.0 (2304:1689)
                      2 = Sinoseen USB 3.0 (1BCF:2D52)
    chessboard_size:  printed SQUARES, e.g. "12x12", "7x9", or (12, 12)
    image_count:      number of images to capture, inclusive range 5..40
    board_type:       "chessboard" (default) or "charuco" / "chboard"

Supported printed-square sizes for both patterns:
    5x5, 7x7, 9x9, 12x12, and 15x15.

Call examples:
    calib(1, "12x12", 20)               # Regular chessboard
    calib(1, "12x12", 20, "charuco")    # ChArUco board

ChArUco calibration uses an EXISTING printed board. The board must MATCH
the selected dictionary (default DICT_5X5_1000), marker-to-square ratio
(default 0.70), marker IDs/layout, and legacy-pattern setting. This library
does NOT generate or save ChArUco board images.

Capture controls: SPACE saves an original frame, Q / ESC cancels.
Once the chosen number of images is captured, the function reports which
chessboards were detected. If some are missing, the operator can retake those
images or, with at least 3 valid detections, continue using the detected images.
The result is this 5x3 NumPy matrix:

    [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1],
     [k1, k2, p1],
     [p2, k3, rms]]

Object-point spacing is 1 unit (no physical square size is assumed).
The top three rows are the ordinary OpenCV camera matrix. The bottom two rows
are a CUSTOM layout of the distortion coefficients and RMS reprojection error.
For cv2.undistort, pass the camera matrix and the five coefficients separately.

Every call saves its original frames and one calibration_results.txt under
camera_calibrations/Sinoseen_USB_{2.0|3.0}/Calibration_N/.
No Excel, CSV, NPZ or annotated/detected-image directories are created.

NOTE: For accurate intrinsics, capture the board at DIFFERENT positions,
orientations and distances. Identical board poses cannot constrain intrinsic
and distortion parameters reliably, even with a small RMS error.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional, Union, Tuple

import cv2
import numpy as np


# Values carried forward from the user's working calibration script.
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
# Use unit spacing between adjacent pattern corners. Physical square length
# is not needed for intrinsics/distortion. Board-to-camera translations in
# robot-world units require the real printed square length.
SUPPORTED_SIZES = {(5, 5), (7, 7), (9, 9), (12, 12), (15, 15)}
DEFAULT_ARUCO_DICTIONARY = "DICT_5X5_1000"
DEFAULT_MARKER_RATIO = 0.70  # marker side / square side
MIN_IMAGES = 5           # Minimum images REQUESTED per calibration
MAX_IMAGES = 40
MIN_VALID_IMAGES = 3     # Minimum successfully detected boards to attempt calibration

CAMERAS = {
    1: {"name": "Sinoseen USB 2.0", "vid": "2304", "pid": "1689", "folder": "Sinoseen_USB_2.0"},
    2: {"name": "Sinoseen USB 3.0", "vid": "1bcf", "pid": "2d52", "folder": "Sinoseen_USB_3.0"},
}

# By default the output is alongside this library. An environment variable
# allows a robot project to use a writable data directory without code edits.
OUTPUT_ROOT = Path(os.environ.get(
    "CAMERA_CALIBRATION_OUTPUT",
    str(Path(__file__).resolve().parent / "camera_calibrations")
)).expanduser().resolve()


class CalibrationError(RuntimeError):
    """Raised when identification, capture or calibration cannot complete."""


class _Capture(NamedTuple):
    image_size: Tuple[int, int]
    object_points: Optional[np.ndarray]
    image_points: Optional[np.ndarray]


def _usb_hex(value) -> str:
    """Normalize a camera metadata USB ID into 4-digit lowercase hex."""
    if value is None or value == "":
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):04x}"
    text = str(value).strip().lower().removeprefix("0x")
    return text.zfill(4) if re.fullmatch(r"[0-9a-f]+", text) else ""


def _ids(device) -> Tuple[str, str]:
    vid = _usb_hex(getattr(device, "vid", None))
    pid = _usb_hex(getattr(device, "pid", None))
    # On some Windows backends, the USB ID is embedded in the device path.
    path = str(getattr(device, "path", "") or "")
    match = re.search(r"vid_([0-9a-fA-F]{4})[&/#\\]+pid_([0-9a-fA-F]{4})", path, re.I)
    if match:
        vid = vid or match.group(1).lower()
        pid = pid or match.group(2).lower()
    return vid, pid


def _enumerate_devices(backend):
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError as exc:
        raise CalibrationError(
            "Missing camera discovery dependency. Run: "
            "python -m pip install opencv-python numpy cv2-enumerate-cameras"
        ) from exc
    try:
        return list(enumerate_cameras(backend))
    except Exception as exc:
        raise CalibrationError(f"Camera discovery failed: {exc}") from exc


def _select_camera(camera: int):
    spec = CAMERAS[camera]
    if sys.platform == "win32":
        backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF)
    elif sys.platform.startswith("linux"):
        backends = (cv2.CAP_V4L2,)
    else:
        backends = (cv2.CAP_ANY,)

    seen = []
    for backend in backends:
        try:
            devices = _enumerate_devices(backend)
        except CalibrationError:
            # Dependency failure must be reported (do not hide it).
            raise
        matches = [device for device in devices if _ids(device) ==
                   (spec["vid"], spec["pid"])]
        if len(matches) == 1:
            chosen = matches[0]
            # Validate that an index and backend have been supplied.
            if int(getattr(chosen, "index", -1)) < 0:
                raise CalibrationError("Camera enumerator returned an invalid index.")
            return chosen
        if len(matches) > 1:
            raise CalibrationError(
                f"Multiple cameras report the same VID/PID for {spec['name']}. "
                "Cannot safely choose one; use a unique serial/device path."
            )
        seen.extend(
            f"{getattr(d, 'name', 'Unknown')} ({':'.join(_ids(d))})"
            for d in devices
        )
    raise CalibrationError(
        f"{spec['name']} (VID:PID {spec['vid'].upper()}:{spec['pid'].upper()}) "
        "not found. Connected cameras: " + (", ".join(seen) or "none")
    )


def _open_camera(device):
    index = int(getattr(device, "index", -1))
    backend = int(getattr(device, "backend", cv2.CAP_ANY))
    # The index is obtained from the matching VID/PID, NEVER supplied by the user.
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture.release()
        raise CalibrationError("Cannot open the selected Sinoseen camera.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    return capture


def _board_size(value: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert printed squares to (columns, rows)."""
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", value)
        if not match:
            raise ValueError('chessboard_size must look like "12x12" or "7x9".')
        cols, rows = int(match.group(1)), int(match.group(2))
    elif isinstance(value, (tuple, list)) and len(value) == 2 and all(
        isinstance(n, int) and not isinstance(n, bool) for n in value
    ):
        cols, rows = value
    else:
        raise ValueError('chessboard_size must look like "12x12" or be (12, 12).')
    if (cols, rows) not in SUPPORTED_SIZES:
        options = ", ".join(f"{x}x{y}" for x, y in sorted(SUPPORTED_SIZES))
        raise ValueError(f"Supported printed-square sizes: {options}.")
    return (cols, rows)


def _pattern_type(value: str) -> str:
    """Accept both the user-facing 'chboard' and standard 'charuco' names."""
    if not isinstance(value, str):
        raise ValueError('board_type must be "chessboard" or "charuco".')
    normalized = value.strip().lower().replace("-", "").replace("_", "")
    if normalized in ("chess", "chessboard", "cb"):
        return "chessboard"
    if normalized in ("charuco", "chboard", "charucoboard"):
        return "charuco"
    raise ValueError('board_type must be "chessboard" or "charuco" (also accepts "chboard").')


def _make_charuco_board(square_count, dictionary=DEFAULT_ARUCO_DICTIONARY,
                        marker_ratio=DEFAULT_MARKER_RATIO,
                        legacy_pattern=False):
    """Build the EXACT reference pattern expected by the ChArUco detector.

    Virtual square spacing = 1.0 unit. Printed geometry must match marker
    ratio, dictionary, marker IDs/layout, and legacy-pattern convention.
    """
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, "CharucoBoard"):
        raise CalibrationError(
            "Your OpenCV build lacks ChArUco. Install an OpenCV build with "
            "the aruco module, e.g. opencv-contrib-python."
        )
    if not isinstance(dictionary, str) or not hasattr(aruco, dictionary):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary!r}.")
    if not (0.0 < float(marker_ratio) < 1.0):
        raise ValueError("marker_ratio must be between 0 and 1 (exclusive).")
    dictionary_obj = aruco.getPredefinedDictionary(getattr(aruco, dictionary))
    try:
        board = aruco.CharucoBoard(
            tuple(square_count), 1.0, float(marker_ratio), dictionary_obj
        )
    except cv2.error as exc:
        raise CalibrationError(f"Could not create ChArUco board: {exc}") from exc
    if legacy_pattern:
        if not hasattr(board, "setLegacyPattern"):
            raise CalibrationError("This OpenCV version cannot use a legacy ChArUco board.")
        board.setLegacyPattern(True)
    return board


def _allocate_run(camera: int) -> Tuple[int, Path]:
    parent = OUTPUT_ROOT / CAMERAS[camera]["folder"]
    parent.mkdir(parents=True, exist_ok=True)
    existing = []
    for child in parent.iterdir():
        if child.is_dir():
            match = re.fullmatch(r"Calibration_(\d+)", child.name)
            if match:
                existing.append(int(match.group(1)))
    # mkdir(exist_ok=False) ensures an existing run is never overwritten.
    run_number = max(existing, default=0) + 1
    while True:
        folder = parent / f"Calibration_{run_number}"
        try:
            (folder / "images").mkdir(parents=True, exist_ok=False)
            return run_number, folder
        except FileExistsError:
            run_number += 1


def _detect(frame, square_count: Tuple[int, int], board_type="chessboard",
            charuco_board=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if board_type == "charuco":
        if charuco_board is None:
            raise CalibrationError("Missing ChArUco board configuration.")
        aruco = cv2.aruco
        try:
            if hasattr(aruco, "CharucoDetector"):
                corners, ids, _markers, _marker_ids = aruco.CharucoDetector(
                    charuco_board
                ).detectBoard(gray)
            else:  # Older OpenCV ArUco bindings
                marker_corners, marker_ids, _ = aruco.detectMarkers(
                    gray, charuco_board.getDictionary()
                )
                if marker_ids is None:
                    return None, None
                _, corners, ids = aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, charuco_board
                )
            if corners is None or ids is None or len(ids) < 6:
                return None, None
            if hasattr(charuco_board, "checkCharucoCornersCollinear") and \
                    charuco_board.checkCharucoCornersCollinear(ids):
                return None, None
            obj, img = charuco_board.matchImagePoints(corners, ids)
        except cv2.error:
            return None, None
        obj = np.asarray(obj, dtype=np.float32).reshape(-1, 3)
        img = np.asarray(img, dtype=np.float32).reshape(-1, 1, 2)
        if len(obj) < 6 or len(obj) != len(img):
            return None, None
        return obj, img

    pattern = (square_count[0] - 1, square_count[1] - 1)
    success, corners = False, None

    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            success, corners = cv2.findChessboardCornersSB(gray, pattern)
        except cv2.error:
            success = False

    if not success:
        success, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if success:
            corners = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
    if not success or corners is None:
        return None, None

    count = pattern[0] * pattern[1]
    object_points = np.zeros((count, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[
        0:pattern[0], 0:pattern[1]
    ].T.reshape(-1, 2)
    # Consecutive inner corners are one arbitrary unit apart.
    image_points = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    return object_points, image_points


def _capture_images(device, camera: int, square_count, image_count: int,
                    folder: Path, board_type="chessboard", charuco_board=None) -> Optional[list[_Capture]]:
    cap = _open_camera(device)
    window = f"{CAMERAS[camera]['name']} Calibration - SPACE: Capture  Q: Cancel"
    shots: list[_Capture] = []
    image_size = None
    try:
        print("\nCamera opened. SPACE = capture image; Q = cancel this calibration.")
        while len(shots) < image_count:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise CalibrationError("Camera stopped returning frames.")
            size = (int(frame.shape[1]), int(frame.shape[0]))
            if image_size is None:
                image_size = size
                print(f"Actual image resolution: {size[0]} x {size[1]}")
            if size != image_size:
                raise CalibrationError("Camera image resolution changed during capture.")

            preview = frame.copy()
            detected = sum(shot.image_points is not None for shot in shots)
            cv2.putText(preview,
                        f"Captured {len(shots)}/{image_count} | Detected {detected}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65,
                        (0, 255, 255), 2)
            cv2.putText(preview, "SPACE = Capture    Q = Cancel", (12, 57),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return None
            if key != ord(" "):
                continue

            image_path = folder / "images" / f"image_{len(shots)+1:03d}.png"
            if not cv2.imwrite(str(image_path), frame):
                raise CalibrationError(f"Failed to save image: {image_path}")
            obj, img = _detect(frame, square_count, board_type, charuco_board)
            shots.append(_Capture(size, obj, img))
            print(f"Image {len(shots)}/{image_count}: " +
                  ("DETECTED" if img is not None else "NOT DETECTED"))
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return shots


def _retake_failed_images(device, camera: int, square_count, shots: list[_Capture],
                          folder: Path, board_type="chessboard",
                          charuco_board=None) -> Optional[list[_Capture]]:
    """Recapture only slots with missing corners, keeping original file names.

    Replacements must actually detect the board. Q cancels this run without
    presenting a previous result as a new calibration.
    """
    missing = [i for i, shot in enumerate(shots) if shot.image_points is None]
    if not missing:
        return shots

    cap = _open_camera(device)
    window = f"{CAMERAS[camera]['name']} - Retake undetected images"
    expected_size = shots[0].image_size
    print(f"\nRetake {len(missing)} undetected image(s). SPACE = capture; Q = cancel.")
    try:
        for index in missing:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise CalibrationError("Camera stopped returning frames during retake.")
                size = (int(frame.shape[1]), int(frame.shape[0]))
                if size != expected_size:
                    raise CalibrationError("Camera resolution changed during retake.")

                preview = frame.copy()
                cv2.putText(preview, f"Retake image {index+1}/{len(shots)}",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65,
                            (0, 255, 255), 2)
                cv2.putText(preview, "SPACE = Capture   Q = Cancel",
                            (12, 57), cv2.FONT_HERSHEY_SIMPLEX, .55,
                            (255, 255, 255), 2)
                cv2.imshow(window, preview)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    return None
                if key != 32:
                    continue

                obj, img = _detect(frame, square_count, board_type, charuco_board)
                if img is None:
                    print(f"Image {index+1}/{len(shots)}: NOT DETECTED - try again")
                    continue

                image_path = folder / "images" / f"image_{index+1:03d}.png"
                if not cv2.imwrite(str(image_path), frame):
                    raise CalibrationError(f"Failed to replace image: {image_path}")
                shots[index] = _Capture(size, obj, img)
                print(f"Image {index+1}/{len(shots)}: DETECTED (replaced)")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return shots


def _review_detections(device, camera: int, square_count, shots: list[_Capture],
                       folder: Path, board_type="chessboard",
                       charuco_board=None) -> Optional[list[_Capture]]:
    """Offer retakes or a 3+ detected-view calibration without crashing."""
    while True:
        detected = sum(shot.image_points is not None for shot in shots)
        failed = len(shots) - detected
        print(f"\nCaptured: {len(shots)} | Detected: {detected} | Not detected: {failed}")
        if failed == 0:
            return shots
        if detected >= MIN_VALID_IMAGES:
            print(f"You can calibrate using the {detected} detected images, "
                  "or retake the undetected ones.")
            print("C = Calibrate with detected images | R = Retake missed images | Q = Cancel")
            prompt = "Choose (C/R/Q): "
        else:
            print(f"At least {MIN_VALID_IMAGES} detections are required. "
                  "Retake undetected images to continue.")
            print("R = Retake missed images | Q = Cancel")
            prompt = "Choose (R/Q): "

        answer = input(prompt).strip().lower()
        if answer == "q":
            return None
        if answer == "c" and detected >= MIN_VALID_IMAGES:
            return shots
        if answer == "r":
            updated = _retake_failed_images(
                device, camera, square_count, shots, folder,
                board_type, charuco_board
            )
            if updated is None:
                return None
            shots = updated
            continue
        print("Invalid choice. Select one of the displayed options.")


def _calibrate(shots: list[_Capture]):
    valid = [shot for shot in shots if shot.image_points is not None]
    if len(valid) < MIN_VALID_IMAGES:
        raise CalibrationError(
            f"Only {len(valid)} valid board detections; at least "
            f"{MIN_VALID_IMAGES} are needed to attempt calibration."
        )
    if len({shot.image_size for shot in valid}) != 1:
        raise CalibrationError("Captured images do not have a consistent resolution.")

    try:
        rms, intrinsic, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
            [shot.object_points for shot in valid],
            [shot.image_points for shot in valid],
            valid[0].image_size, None, None,
        )
    except cv2.error as exc:
        raise CalibrationError(
            "OpenCV could not calibrate with these views. Capture more images "
            "at varied board angles and distances."
        ) from exc
    d = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if (len(d) < 5 or not np.isfinite(d).all() or
            not np.isfinite(intrinsic).all() or not np.isfinite(rms)):
        raise CalibrationError("OpenCV returned invalid calibration values.")

    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    k1, k2, p1, p2, k3 = (float(d[i]) for i in range(5))
    output = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
        [k1, k2, p1],
        [p2, k3, float(rms)],
    ], dtype=np.float64)
    return output, len(valid)


def _format_matrix_decimal(matrix: np.ndarray) -> str:
    """Display all matrix entries in decimal notation, without e+ or e-."""
    return np.array2string(
        np.asarray(matrix, dtype=np.float64),
        formatter={"float_kind": lambda value: np.format_float_positional(
            value, unique=True, trim="-"
        )},
        max_line_width=160,
    )


def _write_result(folder: Path, camera: int, run_number: int,
                  square_count: Tuple[int, int], shots: list[_Capture],
                  matrix: np.ndarray, used_count: int, board_type="chessboard",
                  dictionary=DEFAULT_ARUCO_DICTIONARY,
                  marker_ratio=DEFAULT_MARKER_RATIO, legacy_pattern=False):
    path = folder / "calibration_results.txt"
    formatted = _format_matrix_decimal(matrix)
    with path.open("w", encoding="utf-8") as file:
        file.write(f"Camera: {CAMERAS[camera]['name']}\n")
        file.write(f"Calibration: {run_number}\n")
        file.write(f"Date: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write(f"Pattern: {board_type}\n")
        file.write(f"Board: {square_count[0]}x{square_count[1]} printed squares\n")
        if board_type == "charuco":
            file.write(f"ArUco dictionary: {dictionary}\n")
            file.write(f"Marker/square side ratio: {marker_ratio}\n")
            file.write(f"Legacy pattern: {legacy_pattern}\n")
        file.write(f"Inner corners: {square_count[0]-1}x{square_count[1]-1}\n")
        file.write(f"Images captured: {len(shots)}\n")
        file.write(f"Images detected: {used_count}\n")
        file.write(f"Images not detected: {len(shots) - used_count}\n")
        file.write(f"Resolution: {shots[0].image_size[0]} x {shots[0].image_size[1]}\n\n")
        file.write("Combined calibration matrix (5x3):\n")
        file.write("[[fx, 0, cx], [0, fy, cy], [0, 0, 1], "
                   "[k1, k2, p1], [p2, k3, RMS]]\n")
        file.write(formatted + "\n")


def calibration(camera: int, chessboard_size: Union[str, Tuple[int, int]],
                image_count: int, board_type="chessboard", *,
                dictionary=DEFAULT_ARUCO_DICTIONARY,
                marker_ratio=DEFAULT_MARKER_RATIO,
                legacy_pattern=False) -> Optional[np.ndarray]:
    """Capture and calibrate chessboard or ChArUco; return a 5x3 NumPy matrix.

    Examples: calibration(1, "12x12", 20)
              calibration(2, "15x15", 30, "charuco")

    Images are captured by pressing SPACE in the camera preview. Pressing Q or
    Escape cancels and returns None. Camera identification is automatic by VID
    and PID, not by user-provided OpenCV indices.

    Raises:
        ValueError: invalid camera number, chessboard size, or image count.
        CalibrationError: camera not found, low detection count, capture failure,
                          or invalid calibration result.
    """
    if type(camera) is not int or camera not in CAMERAS:
        raise ValueError("camera must be 1 (Sinoseen USB 2.0) or 2 (Sinoseen USB 3.0).")
    squares = _board_size(chessboard_size)
    if type(image_count) is not int or not MIN_IMAGES <= image_count <= MAX_IMAGES:
        raise ValueError(f"image_count must be a whole number from {MIN_IMAGES} to {MAX_IMAGES}.")

    mode = _pattern_type(board_type)
    board = (_make_charuco_board(squares, dictionary, marker_ratio, legacy_pattern)
             if mode == "charuco" else None)
    spec = CAMERAS[camera]
    print(f"\nCamera: {spec['name']} ({spec['vid'].upper()}:{spec['pid'].upper()})")
    print(f"Pattern: {mode} | Board: {squares[0]} x {squares[1]} printed squares")
    if mode == "charuco":
        print(f"ChArUco dictionary: {dictionary} | Marker ratio: {marker_ratio}")
    print(f"Images to capture: {image_count}")
    device = _select_camera(camera)
    run_number, folder = _allocate_run(camera)
    print(f"Calibration {run_number} | Folder: {folder}")

    try:
        shots = _capture_images(device, camera, squares, image_count, folder,
                                mode, board)
        if shots is None:
            print("Calibration cancelled.")
            # Leave partial captures intact, but do not consume a run number
            # when nothing was captured.
            if not any((folder / "images").iterdir()):
                shutil.rmtree(folder)
            return None

        shots = _review_detections(device, camera, squares, shots, folder,
                                   mode, board)
        if shots is None:
            print("Calibration cancelled. No new results calculated.")
            return None
        matrix, used = _calibrate(shots)
        _write_result(folder, camera, run_number, squares, shots, matrix, used,
                      mode, dictionary, marker_ratio, legacy_pattern)
    except Exception:
        # Preserve captured images for diagnosis if the run fails.
        if not any((folder / "images").iterdir()):
            shutil.rmtree(folder)
        raise

    print("\nCombined calibration matrix (5 x 3):")
    print(_format_matrix_decimal(matrix))
    print(f"\nSaved images: {folder / 'images'}")
    print(f"Saved results: {folder / 'calibration_results.txt'}")
    if used < 15:
        print(f"WARNING: Only {used} detected views. Results may be unstable, "
              "particularly with only 3 views. Do not use these parameters "
              "for precision robot vision without further validation.")
    return matrix


def _next_parameters(text: str, current: tuple) -> tuple:
    """Parse the next 3/4 positional inputs; never evaluate arbitrary Python."""
    if not text.strip():
        return current
    stripped = text.strip()
    command_match = re.fullmatch(
        r"(?:camera_calibration\.)?(?:calib|calibration)\s*\((.*)\)",
        stripped, flags=re.IGNORECASE,
    )
    if command_match:
        stripped = command_match.group(1)
    plain = re.fullmatch(
        r"\s*([12])\s*,\s*(\d+)\s*[xX×]\s*(\d+)\s*,\s*(\d+)"
        r"(?:\s*,\s*(chessboard|chboard|charuco))?\s*",
        stripped, flags=re.I,
    )
    if plain:
        parameters = (
            int(plain.group(1)),
            f"{plain.group(2)}x{plain.group(3)}",
            int(plain.group(4)),
            *( [plain.group(5)] if plain.group(5) else [] ),
        )
    else:
        try:
            parameters = ast.literal_eval("(" + stripped + ")")
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as exc:
            raise ValueError(
                'Use: 1,12x12,20 or 1,12x12,20,charuco'
            ) from exc
    if not isinstance(parameters, tuple) or len(parameters) not in (3, 4):
        raise ValueError("Enter camera, board size, images, [optional board type].")
    camera, size, count = parameters[:3]
    # For 3 fields, preserve previous pattern, not always chessboard.
    mode = parameters[3] if len(parameters) == 4 else current[3]
    if type(camera) is not int or camera not in CAMERAS:
        raise ValueError("Camera must be 1 (USB 2.0) or 2 (USB 3.0).")
    _board_size(size)
    if type(count) is not int or not MIN_IMAGES <= count <= MAX_IMAGES:
        raise ValueError("Image count must be a whole number from 5 to 40.")
    return camera, size, count, _pattern_type(mode)


def calib(camera: int, chessboard_size: Union[str, Tuple[int, int]],
          image_count: int, board_type="chessboard", *,
          dictionary=DEFAULT_ARUCO_DICTIONARY,
          marker_ratio=DEFAULT_MARKER_RATIO,
          legacy_pattern=False) -> Optional[np.ndarray]:
    """Run one or more calibrations, returning last successful 5x3 matrix.

    calib(1, "12x12", 20)             # chessboard, backward compatible
    calib(1, "12x12", 20, "chboard")  # ChArUco board

    After each run, reply Y/N, then enter the next inputs in one line.
    Enter alone repeats the previous camera/size/count/pattern.
    """
    settings = _next_parameters("", (camera, chessboard_size, image_count,
                                    _pattern_type(board_type)))
    last_result = None
    while True:
        try:
            result = calibration(*settings, dictionary=dictionary,
                                 marker_ratio=marker_ratio,
                                 legacy_pattern=legacy_pattern)
        except CalibrationError as exc:
            print(f"Calibration could not be completed: {exc}")
            while True:
                retry = input("Start a new calibration with the same settings? (Y/N): ").strip().lower()
                if retry in ("y", "yes"):
                    break
                if retry in ("n", "no", "q"):
                    return None
                print("Please enter Y or N.")
            continue
        if result is None:
            print("Calibration cancelled. No new calibration result was produced.")
            return None
        last_result = result
        while True:
            again = input("\nRun another calibration? (Y/N): ").strip().lower()
            if again in ("n", "no", "q"):
                return last_result
            if again in ("y", "yes"):
                break
            print("Please enter Y or N.")
        while True:
            entered = input(
                "Next calibration (camera, board, images[, pattern]) "
                "[Enter = repeat same settings]: "
            )
            try:
                settings = _next_parameters(entered, settings)
                break
            except ValueError as exc:
                print("Invalid inputs:", exc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sinoseen camera calibration library")
    parser.add_argument("camera", type=int, help="1 = USB 2.0; 2 = USB 3.0")
    parser.add_argument("chessboard_size", help='Printed squares, e.g. "12x12"')
    parser.add_argument("image_count", type=int, help="5 to 40 images")
    parser.add_argument("--once", action="store_true", help="Run only once, without the repeat question")
    parser.add_argument("--pattern", choices=["chessboard", "charuco", "chboard"],
                        default="chessboard", help="Calibration target pattern")
    args = parser.parse_args()
    try:
        func = calibration if args.once else calib
        func(args.camera, args.chessboard_size, args.image_count, args.pattern)
    except (CalibrationError, ValueError, cv2.error) as exc:
        print("Error:", exc, file=sys.stderr)
        sys.exit(1)
