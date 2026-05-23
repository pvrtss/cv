import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

VIDEO_PATH = "video.mp4"
OUTPUT_PATH = "output.mp4"

# Более точная модель для crowded scenes
model = YOLO("yolov8m.pt")

tracker = DeepSort(
    max_age=60,
    n_init=5,
    max_cosine_distance=0.2,
    nn_budget=100
)

line_points = []
drawing_finished = False


def mouse_callback(event, x, y, flags, param):

    global line_points
    global drawing_finished

    if event == cv2.EVENT_LBUTTONDOWN and not drawing_finished:

        line_points.append((x, y))

        if len(line_points) == 2:
            drawing_finished = True


def side_of_line(point, line_start, line_end):

    x, y = point

    x1, y1 = line_start
    x2, y2 = line_end

    return (
        (x - x1) * (y2 - y1)
        -
        (y - y1) * (x2 - x1)
    )

cap = cv2.VideoCapture(VIDEO_PATH)

ret, first_frame = cap.read()

if not ret:
    print("Ошибка чтения видео")
    exit()

cv2.namedWindow("Select Line")

cv2.setMouseCallback("Select Line", mouse_callback)

while True:

    frame_copy = first_frame.copy()

    cv2.putText(
        frame_copy,
        "Click 2 points to draw counting line",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Рисуем точки
    for point in line_points:

        cv2.circle(
            frame_copy,
            point,
            5,
            (0, 0, 255),
            -1
        )

    # Рисуем линию
    if len(line_points) == 2:

        cv2.line(
            frame_copy,
            line_points[0],
            line_points[1],
            (0, 255, 0),
            3
        )

    cv2.imshow("Select Line", frame_copy)

    key = cv2.waitKey(1)

    if drawing_finished:
        break

cv2.destroyWindow("Select Line")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

counted_ids = set()
track_history = {}
counter = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    results = model(
        frame,
        conf=0.65,
        iou=0.45
    )[0]

    detections = []

    for result in results.boxes:

        cls = int(result.cls[0])

        # class 0 = person
        if cls != 0:
            continue

        x1, y1, x2, y2 = map(int, result.xyxy[0])

        confidence = float(result.conf[0])

        width_box = x2 - x1
        height_box = y2 - y1

        area = width_box * height_box

        if width_box < 35 or height_box < 70:
            continue

        if area > 300000:
            continue

        aspect_ratio = width_box / height_box

        if aspect_ratio > 1.2:
            continue

        margin = 10

        if (
            x1 <= margin or
            y1 <= margin or
            x2 >= frame_w - margin or
            y2 >= frame_h - margin
        ):
            continue

        detections.append(
            (
                [x1, y1, width_box, height_box],
                confidence,
                "person"
            )
        )

    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    cv2.line(
        frame,
        line_points[0],
        line_points[1],
        (0, 255, 0),
        3
    )

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        bottom_center_x = int((x1 + x2) / 2)
        bottom_center_y = y2

        current_point = (
            bottom_center_x,
            bottom_center_y
        )

        cv2.circle(
            frame,
            current_point,
            5,
            (0, 0, 255),
            -1
        )

        if track_id not in track_history:

            track_history[track_id] = current_point

        previous_point = track_history[track_id]

        prev_side = side_of_line(
            previous_point,
            line_points[0],
            line_points[1]
        )

        current_side = side_of_line(
            current_point,
            line_points[0],
            line_points[1]
        )

        crossed = (
            prev_side * current_side < 0
        )

        if crossed and track_id not in counted_ids:
            counter += 1
            counted_ids.add(track_id)

        track_history[track_id] = current_point

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    cv2.putText(
        frame,
        f"Count: {counter}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        3
    )

    small_frame = cv2.resize(frame, (width // 2, height // 2))
    cv2.imshow("People Counter", small_frame)


    out.write(frame)

    key = cv2.waitKey(1)

    # ESC
    if key == 27:
        break

cap.release()

out.release()

cv2.destroyAllWindows()

print()
print("====================================")
print("ГОТОВО")
print(f"Итоговое количество людей: {counter}")
print(f"Видео сохранено: {OUTPUT_PATH}")
print("====================================")