import cv2
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Использование: python task1.py <путь_к_видео>")
    sys.exit(1)

cap = cv2.VideoCapture(sys.argv[1])
detector = cv2.QRCodeDetector()

scale = 0.4

max_angle_no_corr = 0
max_angle_with_corr = 0
loss_angle = None
final_loss_angle = None
was_decoding = False
frame_count = 0
last_angle = 0

def calculate_angle_from_points(points):
    """Расчет угла наклона по точкам"""
    if points is None or len(points) < 4:
        return 0
    
    pts = points.astype(np.float32)
    
    side1 = np.linalg.norm(pts[1] - pts[0])  # верхняя
    side2 = np.linalg.norm(pts[2] - pts[1])  # правая
    side3 = np.linalg.norm(pts[3] - pts[2])  # нижняя
    side4 = np.linalg.norm(pts[0] - pts[3])  # левая
    
    avg_width = (side1 + side3) / 2
    avg_height = (side2 + side4) / 2
    
    if avg_width > 0 and avg_height > 0:
        ratio = min(avg_width, avg_height) / max(avg_width, avg_height)
        angle = (1 - ratio) * 90
        
        width_diff = abs(side1 - side3) / avg_width
        height_diff = abs(side2 - side4) / avg_height
        distortion = (width_diff + height_diff) / 2
        
        angle = angle * (1 + distortion)
        angle = min(90, max(0, angle))
        
        return angle
    
    return 0

print("Обработка видео... Нажмите 'q' для выхода")
print("="*60)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    display = frame.copy()
    current_angle = 0
    
    data, points, _ = detector.detectAndDecode(frame)
    
    if data and points is not None:
        pts = points[0].astype(int)
        for i in range(len(pts)):
            cv2.line(display, tuple(pts[i]), tuple(pts[(i+1) % len(pts)]), (0, 255, 0), 3)
        
        current_angle = calculate_angle_from_points(points[0])
        last_angle = current_angle
        
        if current_angle > max_angle_no_corr:
            max_angle_no_corr = current_angle
        
        print(f"Кадр {frame_count}: РАСПОЗНАНО | Угол: {current_angle:.1f}°")
        cv2.putText(display, "qr decoded", (250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display, f"ang: {current_angle:.1f} deg", (250, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        was_decoding = True
        
    elif points is not None:
        pts = points[0].astype(int)
        for i in range(len(pts)):
            cv2.line(display, tuple(pts[i]), tuple(pts[(i+1) % len(pts)]), (0, 255, 255), 3)
        
        current_angle = calculate_angle_from_points(points[0])
        last_angle = current_angle
        
        cv2.putText(display, "QR FOUND but NOT DECODED", (250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display, f"Angle: {current_angle:.1f} deg", (250, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if was_decoding and loss_angle is None:
            loss_angle = current_angle
            print(f"\n!!! ПЕРВАЯ ПОТЕРЯ РАСПОЗНАВАНИЯ при угле {current_angle:.1f}° (кадр {frame_count}) !!!\n")
            was_decoding = False
        
        pts_f = points[0].astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts_f.sum(axis=1)
        diff = np.diff(pts_f, axis=1)
        rect[0] = pts_f[np.argmin(s)]
        rect[2] = pts_f[np.argmax(s)]
        rect[1] = pts_f[np.argmin(diff)]
        rect[3] = pts_f[np.argmax(diff)]
        
        width = int(max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3])))
        height = int(max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1])))
        
        if width > 10 and height > 10:
            dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(frame, M, (width, height))
            
            if warped is not None:
                data_corr, _, _ = detector.detectAndDecode(warped)
                if data_corr:
                    if current_angle > max_angle_with_corr:
                        max_angle_with_corr = current_angle
                    print(f"  -> КОРРЕКЦИЯ ПОМОГЛА! Угол: {current_angle:.1f}°")
                    cv2.putText(display, "Correction: SUCCESS", (250, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    warped_small = cv2.resize(warped, (150, 150))
                    cv2.imshow('Corrected QR', warped_small)
                else:
                    cv2.putText(display, "Correction: FAILED", (250, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    if final_loss_angle is None and loss_angle is not None:
                        final_loss_angle = current_angle
                        print(f"\n!!! ОКОНЧАТЕЛЬНАЯ ПОТЕРЯ РАСПОЗНАВАНИЯ при угле {current_angle:.1f}° !!!")
                        print(f"QR код больше не распознается\n")
    else:
        cv2.putText(display, "NO QR CODE", (250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display, f"Last angle: {last_angle:.1f} deg", (250, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        
        if was_decoding and loss_angle is None:
            loss_angle = last_angle
            print(f"\n!!! ПОТЕРЯ РАСПОЗНАВАНИЯ (QR пропал) при угле {last_angle:.1f}° !!!\n")
            was_decoding = False
    
    cv2.putText(display, f"Max angle: {max_angle_no_corr:.1f} deg", (10, display.shape[0] - 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(display, f"Max with corr: {max_angle_with_corr:.1f} deg", (10, display.shape[0] - 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    if loss_angle:
        cv2.putText(display, f"First loss: {loss_angle:.1f} deg", (10, display.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    if final_loss_angle:
        cv2.putText(display, f"Final loss: {final_loss_angle:.1f} deg", (10, display.shape[0] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    small_display = cv2.resize(display, (int(display.shape[1]*scale), int(display.shape[0]*scale)))
    cv2.imshow('QR Code Detection', small_display)
    
    if cv2.waitKey(30) & 0xFF == ord('q') or cv2.waitKey(30) & 0xFF == ord('Q'):
        break

print("\n" + "="*60)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
print("="*60)
print(f"Максимальный угол распознавания БЕЗ коррекции: {max_angle_no_corr:.1f}°")
print(f"Максимальный угол распознавания С коррекцией: {max_angle_with_corr:.1f}°")

if loss_angle:
    print(f"\n!!! ПЕРВАЯ ПОТЕРЯ РАСПОЗНАВАНИЯ: {loss_angle:.1f}°")

if final_loss_angle:
    print(f"!!! ОКОНЧАТЕЛЬНАЯ ПОТЕРЯ (больше не распозналось): {final_loss_angle:.1f}°")

if max_angle_with_corr > max_angle_no_corr:
    improvement = max_angle_with_corr - max_angle_no_corr
    print(f"\nУЛУЧШЕНИЕ ОТ КОРРЕКЦИИ: +{improvement:.1f}°")

cap.release()
cv2.destroyAllWindows()