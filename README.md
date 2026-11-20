# Camera Calibration Library

A reusable Python library for calibrating **Sinoseen USB 2.0 and USB 3.0 cameras** using standard Chessboard and ChArUco calibration patterns.

The library uses OpenCV to capture images, detect calibration patterns, calculate intrinsic camera parameters, estimate lens distortion coefficients, and return the results as a single **5 × 3 calibration matrix**.

It is designed for camera calibration testing and integration into computer vision and robotics applications.

## Features

- Automatic Sinoseen camera identification using USB VID/PID.
- Chessboard and ChArUco calibration support.
- Five supported board sizes: 5×5, 7×7, 9×9, 12×12 and 15×15.
- Capture 5–40 images per calibration.
- Live camera preview and calibration-pattern detection.
- Retake images when pattern detection fails.
- Automatic calibration result calculation.
- Single 5×3 output matrix with full decimal values.
- Continuous calibration sessions.
- Automatic numbered folders for each calibration.
- Saves original images and calibration results in a text file.
- No Excel output or ChArUco board generation.

## 1. Supported Cameras

| Camera ID | Camera | VID | PID | Requested Resolution |
|---|---|---|---|---|
| 1 | Sinoseen USB 2.0 | `2304` | `1689` | 640 × 480 |
| 2 | Sinoseen USB 3.0 | `1BCF` | `2D52` | 640 × 480 |

Cameras are selected using their USB Vendor ID (VID) and Product ID (PID).

The user does not need to manually specify an OpenCV camera index.

**Note:** If multiple cameras share the same VID/PID, the library cannot uniquely identify them without an additional identifier, such as a unique serial number or device path.

## 2. Requirements

- Python 3.9 or newer
- OpenCV with ArUco/ChArUco support
- NumPy
- cv2-enumerate-cameras
- Supported Sinoseen USB camera
- Printed chessboard or ChArUco calibration pattern

Install the required dependencies:

```bash
python -m pip install numpy opencv-contrib-python cv2-enumerate-cameras
```

Use `opencv-contrib-python` for ChArUco support. Avoid installing multiple competing OpenCV Python packages in the same environment.

## 3. Using the Library

Place `camera_calibration.py` in the same directory as the Python program using it.

No separate installation of the custom calibration library is required.

Start Python:

```bash
python -B
```

Import the calibration function:

```python
from camera_calibration import calib
```

### Chessboard Calibration

USB 2.0 camera, 12×12 chessboard, 20 images:

```python
calib(1, "12x12", 20)
```

USB 3.0 camera, 9×9 chessboard, 30 images:

```python
calib(2, "9x9", 30)
```

### ChArUco Calibration

USB 2.0 camera, 12×12 ChArUco board, 20 images:

```python
calib(1, "12x12", 20, "chboard")
```

The library also accepts `"charuco"`.

### Function Parameters

```python
calib(camera, chessboard_size, image_count, board_type="chessboard")
```

| Parameter | Description |
|---|---|
| `camera` | 1 = USB 2.0, 2 = USB 3.0 |
| `chessboard_size` | Number of printed squares |
| `image_count` | Number of images, from 5 to 40 |
| `board_type` | chessboard or charuco |

## 4. Supported Calibration Board Sizes

| Board Size | Internal Corners |
|---|---|
| 5×5 | 4×4 |
| 7×7 | 6×6 |
| 9×9 | 8×8 |
| 12×12 | 11×11 |
| 15×15 | 14×14 |

These sizes are supported for both standard chessboard and ChArUco calibration.

The library uses unit square spacing for calibration. Physical square dimensions are not required to estimate intrinsic parameters and lens distortion coefficients.

However, actual board-to-camera distances in millimetres cannot be recovered without knowing the physical square dimensions.

### ChArUco Configuration

Default ChArUco parameters:

| Parameter | Default |
|---|---|
| ArUco dictionary | `DICT_5X5_1000` |
| Marker-to-square ratio | 0.70 |
| Legacy pattern | False |

The printed ChArUco board must match the configured dictionary, marker IDs, layout and size ratio.

Different marker configurations can be supplied using optional function arguments.

## 5. Image Capture Procedure

After executing `calib()`, the selected camera opens automatically.

**Camera controls**

| Key | Action |
|---|---|
| Space | Capture image |
| Q | Cancel current calibration |
| Esc | Cancel current calibration |

The terminal displays the image-capture and pattern-detection status.

Example:

```text
Camera opened.
SPACE = capture image
Q = cancel calibration

Image 1/20: DETECTED
Image 2/20: DETECTED
Image 3/20: NOT DETECTED
```

After capturing the requested number of images, the library displays the total detected and undetected images.

If detection fails for some images, the user can retake them or attempt calibration with at least three successfully detected images.

**For reliable calibration, 20–30 well-distributed detected images are recommended. Three images may produce unreliable calibration values.**

## 6. Recommended Calibration Procedure

Keep the camera completely fixed throughout calibration.

Move only the chessboard or ChArUco board.

Recommended image positions and orientations:

| Images | Board Position / Orientation |
|---|---|
| 1–5 | Center, left, right, top, bottom |
| 6–9 | Four image corners |
| 10–13 | Along four image edges |
| 14–17 | Yaw left/right at 15° and 30° |
| 18–21 | Pitch up/down at 15° and 30° |
| 22–25 | Combined yaw and pitch |
| 26–28 | Near, medium and far distances |
| 29–30 | Clockwise and anticlockwise roll |

### Capture Guidelines

- Keep the camera stationary.
- Use a rigid, flat calibration board.
- Capture images at different board distances.
- Cover the entire camera image area.
- Use uniform, diffuse lighting.
- Avoid strong reflections and shadows.
- Avoid blurry images.
- Keep camera focus, zoom and resolution unchanged.
- Ensure sufficient calibration-pattern corners are detected.

## 7. Calibration Output

The library estimates:

**Camera intrinsics:** `fx`, `fy`, `cx`, `cy`

**Distortion coefficients:** `k1`, `k2`, `p1`, `p2`, `k3`

**Reprojection error:** `RMS`

All parameters are returned in one combined 5×3 matrix:

```text
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1],
 [k1, k2, p1],
 [p2, k3, RMS]]
```

The first three rows represent the standard intrinsic camera matrix.

The final two rows contain distortion coefficients and RMS reprojection error in a custom format.

Values are displayed in standard decimal notation without scientific notation.

**Note:** The combined matrix is a custom output format. When using OpenCV functions such as `cv2.undistort()`, extract the intrinsic matrix and distortion coefficients separately.

## 8. Continuous Calibration

After completing Calibration 1, the library asks:

```text
Run another calibration? (Y/N):
```

Selecting `Y` allows another calibration with new input parameters.

Example:

```text
Next calibration (camera, board, images[, pattern]):
2,12x12,30
```

Press Enter without typing new values to repeat the previous settings.

Selecting `N` ends the calibration session.

The library creates numbered folders automatically for each camera.

## 9. Saved Files

```text
camera_calibrations/
│
├── Sinoseen_USB_2.0/
│   ├── Calibration_1/
│   │   ├── images/
│   │   └── calibration_results.txt
│   └── Calibration_2/
│       ├── images/
│       └── calibration_results.txt
│
└── Sinoseen_USB_3.0/
    ├── Calibration_1/
    │   ├── images/
    │   └── calibration_results.txt
    └── Calibration_2/
        ├── images/
        └── calibration_results.txt
```

Each completed calibration stores the original captured images and calibration results.

By default, these folders are created beside the library file. The output directory can be changed using the `CAMERA_CALIBRATION_OUTPUT` environment variable before importing the library.

## 10. Troubleshooting

| Problem | Solution |
|---|---|
| Camera not detected | Check USB connection and camera VID/PID |
| Multiple identical cameras | Use a unique device identifier |
| Chessboard not detected | Check board dimensions, focus and visibility |
| ChArUco not detected | Check dictionary, marker IDs, layout and ratio |
| Not enough valid images | Retake failed images |
| Calibration results vary | Capture multiple board positions and angles |
| `cv2` not found | Install the OpenCV dependency |
| `__pycache__` created | Start Python with `python -B` |

## 11. Important Notes

- Camera calibration should be performed using images with varied positions, orientations and distances.
- Calibration parameters should be validated before using them for precision measurements or robot control.
- The camera's optical settings should remain unchanged after calibration.
- ChArUco and chessboard patterns use different detection methods.
- The library is intended for interactive use with an OpenCV camera preview.
- Physical camera compatibility must be verified on the target operating system.

---

**Library:** `camera_calibration.py`  
**Main Function:** `calib()`  
**Single-Run Function:** `calibration()`  
**Supported Patterns:** Chessboard and ChArUco  
**Output:** 5×3 NumPy calibration matrix  
**Programming Language:** Python  
**Computer Vision Framework:** OpenCV
