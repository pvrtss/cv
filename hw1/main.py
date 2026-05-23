import cv2
import sys

squares = []
square_size = 20

def show_clicked(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: 
        squares.append((x, y, square_size))
        print(f"Квадрат ({x}, {y})")
    

script_name = sys.argv[0]
args = str(sys.argv[1])
print (f"args : {args}")

capture = cv2.VideoCapture(str(args))

if not capture.isOpened():
    print("ОШИБКА: Не удалось открыть видеофайл")
    exit()

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', show_clicked)

while(True):
    ret, frame = capture.read()

    if not ret: 
        print("Конец видео")
        exit()
    # на каждый фрейм
    for (x, y, size) in squares:
        half = size // 2
        # Рисуем квадрат
        cv2.rectangle(frame, 
                    (x - half, y - half), 
                    (x + half, y + half), 
                    (0, 0, 255), 1)

    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF == ord('c') or cv2.waitKey(1) & 0xFF == ord('C'):
        squares.clear()

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
        break

capture.release()
cv2.destroyAllWindows()