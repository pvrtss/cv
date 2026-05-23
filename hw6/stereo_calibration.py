import cv2
import numpy as np
import os

# =========================================================
# SETTINGS
# =========================================================

VIDEO_PATH = "chess.mp4"

CHESSBOARD_SIZE = (9, 6)

SQUARE_SIZE = 35  # mm

# =========================================================
# PREPARE OBJECT POINTS
# =========================================================

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)

objp[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

# =========================================================
# STORAGE
# =========================================================

objpoints = []

imgpoints_left = []
imgpoints_right = []

# =========================================================
# OPEN VIDEO
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0

success_count = 0

# =========================================================
# PROCESS VIDEO
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Берем каждый 10 кадр
    if frame_count % 10 != 0:
        continue

    height, width, _ = frame.shape

    half_width = width // 2

    left = frame[:, :half_width]
    right = frame[:, half_width:]

    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # =====================================================
    # FIND CHESSBOARD
    # =====================================================

    ret_left, corners_left = cv2.findChessboardCorners(
        gray_left,
        CHESSBOARD_SIZE,
        None
    )

    ret_right, corners_right = cv2.findChessboardCorners(
        gray_right,
        CHESSBOARD_SIZE,
        None
    )

    if ret_left and ret_right:

        success_count += 1

        print(f"Chessboard found: {success_count}")

        objpoints.append(objp)

        # refine corners
        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        corners_left = cv2.cornerSubPix(
            gray_left,
            corners_left,
            (11, 11),
            (-1, -1),
            criteria
        )

        corners_right = cv2.cornerSubPix(
            gray_right,
            corners_right,
            (11, 11),
            (-1, -1),
            criteria
        )

        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

        # draw
        cv2.drawChessboardCorners(
            left,
            CHESSBOARD_SIZE,
            corners_left,
            ret_left
        )

        cv2.drawChessboardCorners(
            right,
            CHESSBOARD_SIZE,
            corners_right,
            ret_right
        )

        cv2.imshow("LEFT", left)
        cv2.imshow("RIGHT", right)

        cv2.waitKey(200)

cap.release()

cv2.destroyAllWindows()

print()
print("================================")
print(f"Valid stereo pairs: {success_count}")
print("================================")

# =========================================================
# CALIBRATE LEFT CAMERA
# =========================================================

ret_left, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(
    objpoints,
    imgpoints_left,
    gray_left.shape[::-1],
    None,
    None
)

# =========================================================
# CALIBRATE RIGHT CAMERA
# =========================================================

ret_right, K2, D2, rvecs2, tvecs2 = cv2.calibrateCamera(
    objpoints,
    imgpoints_right,
    gray_right.shape[::-1],
    None,
    None
)

# =========================================================
# STEREO CALIBRATION
# =========================================================

flags = cv2.CALIB_FIX_INTRINSIC

criteria_stereo = (
    cv2.TERM_CRITERIA_MAX_ITER +
    cv2.TERM_CRITERIA_EPS,
    100,
    1e-5
)

ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    K1,
    D1,
    K2,
    D2,
    gray_left.shape[::-1],
    criteria=criteria_stereo,
    flags=flags
)

print()
print("================================")
print("STEREO CALIBRATION DONE")
print("================================")

print()
print("Translation Vector T:")
print(T)

print()
print("Rotation Matrix R:")
print(R)

# =========================================================
# SAVE
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAVE_PATH = os.path.join(
    BASE_DIR,
    "stereo_calibration.npz"
)

np.savez(
    SAVE_PATH,
    K1=K1,
    D1=D1,
    K2=K2,
    D2=D2,
    R=R,
    T=T
)

print()
print("================================")
print("Calibration saved:")
print(SAVE_PATH)
print("================================")