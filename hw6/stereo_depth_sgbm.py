import cv2
import numpy as np

# =========================================================
# PATHS
# =========================================================

VIDEO_PATH = "video.mp4"
CALIB_PATH = "stereo_calibration.npz"

# =========================================================
# LOAD CALIBRATION
# =========================================================

data = np.load(CALIB_PATH)

K1 = data["K1"]
D1 = data["D1"]

K2 = data["K2"]
D2 = data["D2"]

R = data["R"]
T = data["T"]

# =========================================================
# OPEN VIDEO
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

ret, frame = cap.read()

if not ret:
    print("Ошибка чтения видео")
    exit()

# =========================================================
# SPLIT LEFT / RIGHT
# =========================================================

height, width, _ = frame.shape

half_width = width // 2

left = frame[:, :half_width]
right = frame[:, half_width:]

h, w = left.shape[:2]

# =========================================================
# STEREO RECTIFICATION
# =========================================================

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1,
    D1,
    K2,
    D2,
    (w, h),
    R,
    T
)

# =========================================================
# RECTIFICATION MAPS
# =========================================================

map1x, map1y = cv2.initUndistortRectifyMap(
    K1,
    D1,
    R1,
    P1,
    (w, h),
    cv2.CV_32FC1
)

map2x, map2y = cv2.initUndistortRectifyMap(
    K2,
    D2,
    R2,
    P2,
    (w, h),
    cv2.CV_32FC1
)

# =========================================================
# SGBM
# =========================================================

stereo = cv2.StereoSGBM_create(

    minDisparity=0,

    numDisparities=16 * 8,

    blockSize=5,

    P1=8 * 3 * 5**2,

    P2=32 * 3 * 5**2,

    disp12MaxDiff=1,

    uniquenessRatio=10,

    speckleWindowSize=100,

    speckleRange=32
)

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # =====================================================
    # SPLIT
    # =====================================================

    left = frame[:, :half_width]
    right = frame[:, half_width:]

    # =====================================================
    # RECTIFICATION
    # =====================================================

    rect_left = cv2.remap(
        left,
        map1x,
        map1y,
        cv2.INTER_LINEAR
    )

    rect_right = cv2.remap(
        right,
        map2x,
        map2y,
        cv2.INTER_LINEAR
    )

    # =====================================================
    # GRAYSCALE
    # =====================================================

    gray_left = cv2.cvtColor(
        rect_left,
        cv2.COLOR_BGR2GRAY
    )

    gray_right = cv2.cvtColor(
        rect_right,
        cv2.COLOR_BGR2GRAY
    )

    # =====================================================
    # DISPARITY
    # =====================================================

    disparity = stereo.compute(
        gray_left,
        gray_right
    ).astype(np.float32)

    disparity = disparity / 16.0

    # =====================================================
    # NORMALIZE
    # =====================================================

    disparity_visual = cv2.normalize(
        disparity,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    disparity_visual = np.uint8(disparity_visual)

    # =====================================================
    # COLORMAP
    # =====================================================

    disparity_color = cv2.applyColorMap(
        disparity_visual,
        cv2.COLORMAP_JET
    )

    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow("LEFT RECTIFIED", rect_left)

    cv2.imshow("RIGHT RECTIFIED", rect_right)

    cv2.imshow("DISPARITY MAP", disparity_color)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()