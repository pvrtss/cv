import cv2
import numpy as np
import sys


def order_points(pts):

    rect = np.zeros((4,2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


insert_video = sys.argv[1]
camera_video = sys.argv[2]

cap = cv2.VideoCapture(camera_video)
insert_cap = cv2.VideoCapture(insert_video)

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writer = cv2.VideoWriter(
    'result.mp4',
    fourcc,
    fps,
    (width, height)
)

ret, first_frame = cap.read()

if not ret:
    print("Ошибка чтения видео")
    exit()

gray0 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray0,50,150)

contours,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

screen = None
max_area = 0

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 10000:
        continue

    peri = cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,0.02*peri,True)

    if len(approx)==4 and area>max_area:

        screen = approx
        max_area = area


if screen is None:
    print("Экран не найден")
    exit()

screen_pts = order_points(screen.reshape(4,2)).astype(np.float32)

mask = np.zeros(gray0.shape,np.uint8)
cv2.fillConvexPoly(mask,screen_pts.astype(int),255)

orb = cv2.ORB_create(2000)

kp1, des1 = orb.detectAndCompute(gray0,mask)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ret, insert_frame = insert_cap.read()

h,w = insert_frame.shape[:2]

video_pts = np.array([
    [0,0],
    [w,0],
    [w,h],
    [0,h]
],dtype=np.float32)

H_screen = cv2.getPerspectiveTransform(video_pts,screen_pts)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    ret2, insert_frame = insert_cap.read()

    if not ret2:
        insert_cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        ret2, insert_frame = insert_cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    kp2, des2 = orb.detectAndCompute(gray,None)

    if des2 is None:
        writer.write(frame)
        continue

    matches = bf.match(des1,des2)

    matches = sorted(matches,key=lambda x:x.distance)

    good = matches[:150]

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    ).reshape(-1,1,2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    ).reshape(-1,1,2)

    H_track,_ = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5)

    if H_track is None:
        writer.write(frame)
        continue

    H_total = H_track @ H_screen

    warped = cv2.warpPerspective(
        insert_frame,
        H_total,
        (frame.shape[1],frame.shape[0])
    )

    dst = cv2.perspectiveTransform(
        video_pts.reshape(-1,1,2),
        H_total
    )

    mask = np.zeros(frame.shape[:2],dtype=np.uint8)
    cv2.fillConvexPoly(mask,dst.astype(int),255)

    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(frame,frame,mask=mask_inv)
    fg = cv2.bitwise_and(warped,warped,mask=mask)

    result = cv2.add(bg,fg)

    cv2.imshow("Overlay",result)

    writer.write(result)
    
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
        break


cap.release()
insert_cap.release()
writer.release()

cv2.destroyAllWindows()