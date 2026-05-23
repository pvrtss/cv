# -*- coding: utf-8 -*-
import numpy as np
import cv2
import json
import os
import glob
import argparse
import uuid
from datetime import datetime


class Base():
    """Базовый класс для калибровки"""
    def __init__(self):
        pass

    def get_board(self):
        return {}

    def calibrate_by_images(self, source, calib_source, type):
        images = glob.glob(calib_source + "*.jpg") + glob.glob(calib_source + "*.jpeg") + \
                 glob.glob(calib_source + "*.png") + glob.glob(calib_source + "*.bmp")
        if len(images):
            img = cv2.imread(images[0])
            calibration_result = self._calibrate(img)
            calibration_result['type'] = type
            calibration_result['source'] = source
            calibration_result['calibration_source'] = calib_source
            return calibration_result
        return None

    def calibrate_by_video(self, source, calib_source, type, all_frames=False):
        cam = calib_source
        if isinstance(cam, str) and cam.isdigit():
            cam = int(cam)
        cap = cv2.VideoCapture(cam)
        flag, img = cap.read()
        cap.release()
        if flag:
            calibration_result = self._calibrate(img)
            calibration_result['type'] = type
            calibration_result['source'] = source
            calibration_result['calibration_source'] = calib_source
            return calibration_result
        return None

    def _calibrate(self, img):
        focal_length = img.shape
        center = (focal_length[1]/2, focal_length[0]/2)
        camera_matrix = np.array(
                                [[focal_length[1], 0, center[0]],
                                [0, focal_length[0], center[1]],
                                [0, 0, 1]], dtype="double"
                                )
        dist_coeffs = np.zeros((4,1))
        rvecs = []
        tvecs = []
        resolution = {"w": focal_length[0], "h": focal_length[1]}

        print("-"*20)
        print("Camera Matrix: ", camera_matrix)
        print("Distortion Coefficients : ", dist_coeffs)
        print("-"*20)
        optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix,
                dist_coeffs,
                (resolution['w'], resolution['h']),
                1,
                (resolution['w'], resolution['h']),
        )
        return {
            "id": str(uuid.uuid4()),
            "camera_matrix": camera_matrix.tolist(),
            "optimal_camera_matrix": optimal_camera_matrix.tolist(),
            "roi": roi,
            "distortion": dist_coeffs.tolist(),
            "rvecs": [vec.tolist() for vec in rvecs],
            "tvecs": [vec.tolist() for vec in tvecs],
            "resolution": resolution
        }


class Board:
    """Базовый класс для калибровочной доски"""
    def __init__(self, pattern_size, save_images_path=None):
        self.pattern_size = pattern_size
        self.save_images_path = save_images_path
        self.all_corners = []
        self.all_obj_points = []
        self.all_images = []
        
    def _save_image(self, image, name):
        if self.save_images_path:
            os.makedirs(self.save_images_path, exist_ok=True)
            path = os.path.join(self.save_images_path, name)
            cv2.imwrite(path, image)
            
    def get_board(self):
        raise NotImplementedError


class ChessBoard(Board):
    """Класс для калибровки по шахматной доске"""
    def __init__(self, pattern_size=(9, 6), square_size=0.025, save_images_path=None):
        Board.__init__(self, pattern_size, save_images_path)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)
        self.square_size = square_size
        
        self.objp = np.zeros((self.pattern_size[1] * self.pattern_size[0], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2) * self.square_size
        
        self.image_counter = 0

    def get_board(self):
        return {
            'type': 'chess',
            'pattern_size': self.pattern_size,
            'square_size': self.square_size,
        }

    def _find_board(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        cv2.drawChessboardCorners(image, self.pattern_size, corners, ret)
        
        if ret == True and len(corners):
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            return ret, corners, image_size
        return False, None, image_size
    
    def calibrate_by_images(self, images_path, source_type='rgb'):
        """Калибровка по папке с изображениями"""
        print(f"\n{'='*60}")
        print("Начало калибровки по изображениям (шахматная доска)")
        print(f"{'='*60}")
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(images_path, ext)))
            images.extend(glob.glob(os.path.join(images_path, ext.upper())))
        
        if not images:
            print(f"Ошибка: Не найдено изображений в папке {images_path}")
            return None
        
        print(f"Найдено изображений: {len(images)}")
        
        objpoints = []
        imgpoints = []
        image_size = None
        successful_images = []
        
        for idx, img_path in enumerate(images, 1):
            print(f"\nОбработка изображения {idx}/{len(images)}: {os.path.basename(img_path)}")
            
            image = cv2.imread(img_path)
            if image is None:
                print(f"  Не удалось загрузить изображение")
                continue
            
            ret, corners, img_size = self._find_board(image)
            image_size = img_size 
            
            if ret:
                objpoints.append(self.objp)
                imgpoints.append(corners)
                successful_images.append(img_path)
                
                if self.save_images_path:
                    save_name = f"chessboard_calib_{self.image_counter:03d}.jpg"
                    self._save_image(image, save_name)
                    self.image_counter += 1
                
                print(f"  ✓ Углы найдены! Сохранено {len(corners)} углов")
            else:
                print(f"  ✗ Углы не найдены. Проверьте, что доска полностью видна и имеет правильный размер {self.pattern_size}")
        
        print(f"\n{'='*60}")
        print(f"Собрано {len(objpoints)} изображений с найденной доской")
        
        if len(objpoints) < 5:
            print("Ошибка: Недостаточно изображений для калибровки (минимум 5)")
            print("Рекомендации:")
            print("- Убедитесь, что шахматная доска полностью видна на изображениях")
            print("- Используйте разные ракурсы и расстояния")
            print("- Обеспечьте хорошее освещение")
            return None
        
        print("\nВыполняется калибровка камеры...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )
        
        if not ret:
            print("Ошибка калибровки!")
            return None
        
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ КАЛИБРОВКИ")
        print("="*60)
        print(f"Матрица камеры:\n{mtx}")
        print(f"\nКоэффициенты дисторсии:\n{dist.ravel()}")
        print(f"\nСредняя ошибка перепроецирования: {mean_error:.6f}")
        print(f"Количество использованных изображений: {len(objpoints)}")
        print(f"Размер изображения: {image_size}")
        print("="*60)
        
        camera_calibration = {
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "mean_error": mean_error,
            "image_size": image_size,
            "num_images": len(objpoints),
            "pattern_size": list(self.pattern_size),
            "square_size": self.square_size,
            "successful_images": successful_images
        }
        
        return camera_calibration
    
    def calibrate_by_video(self, source, source_type='rgb', all_frames=False):
        """Калибровка по видеопотоку или видеофайлу"""
        print(f"\n{'='*60}")
        print("Начало калибровки по видео (шахматная доска)")
        print(f"{'='*60}")
        
        if isinstance(source, str):
            if source.isdigit():
                source = int(source)
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть источник {source}")
            return None
        
        print("Инструкция:")
        print("- Наведите камеру на шахматную доску")
        print("- Нажмите ENTER для захвата кадра для калибровки")
        print("- Нажмите ESC или Q для завершения и начала калибровки")
        print("- Соберите минимум 10-15 кадров с разных ракурсов")
        print()
        
        objpoints = []
        imgpoints = []
        captured_frames = 0
        image_size = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Не удалось получить кадр")
                break
            
            display_frame = frame.copy()
            
            found, corners, img_size = self._find_board(display_frame)
            image_size = img_size
            
            cv2.putText(display_frame, f"Captured: {captured_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press ENTER to capture, ESC/Q to finish", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if found:
                cv2.putText(display_frame, "Board found!", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Chessboard Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:
                if found and corners is not None:
                    objpoints.append(self.objp)
                    imgpoints.append(corners)
                    captured_frames += 1
                    
                    if self.save_images_path:
                        save_name = f"captured_frame_{captured_frames:03d}.jpg"
                        self._save_image(frame, save_name)
                    
                    print(f"Кадр {captured_frames} сохранен")
                else:
                    print("Доска не найдена! Попробуйте другой ракурс.")
                    
            elif key == 27 or key == ord('q'):  # ESC или Q
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"Собрано {captured_frames} кадров с доской")
        
        if captured_frames < 5:
            print("Ошибка: Недостаточно кадров для калибровки (минимум 5)")
            return None

        print("\nВыполняется калибровка камеры...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )
        
        if not ret:
            print("Ошибка калибровки!")
            return None
        
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ КАЛИБРОВКИ")
        print("="*60)
        print(f"Матрица камеры:\n{mtx}")
        print(f"\nКоэффициенты дисторсии:\n{dist.ravel()}")
        print(f"\nСредняя ошибка перепроецирования: {mean_error:.6f}")
        print("="*60)
        
        camera_calibration = {
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "mean_error": mean_error,
            "image_size": image_size,
            "num_images": captured_frames,
            "pattern_size": list(self.pattern_size),
            "square_size": self.square_size
        }
        
        return camera_calibration


class CircleBoard(Board):
    """Класс для калибровки по круговой доске (асимметричная сетка)"""
    def __init__(self, pattern_size=(4, 11), circle_diameter=0.015, circle_spacing=0.02, save_images_path=None):
        Board.__init__(self, pattern_size, save_images_path)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        self.circle_diameter = circle_diameter
        self.circle_spacing = circle_spacing
        
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 10000
        params.minThreshold = 1
        params.maxThreshold = 200
        params.filterByCircularity = True
        params.minCircularity = 0.1
        params.filterByInertia = True
        params.minInertiaRatio = 0.001
        self.detector = cv2.SimpleBlobDetector_create(params)
        
        self.objp = self.create_board()
        self.image_counter = 0
    
    def create_board(self):
        """Создание 3D точек для асимметричной круговой сетки"""
        objp = np.zeros((self.pattern_size[1] * self.pattern_size[0], 3), np.float32)
        temp_x = 0
        ind = 0
        for i in range(self.pattern_size[1]):
            start_y = 0
            if i % 2 != 0:
                start_y += self.circle_spacing / 2
            for k in range(self.pattern_size[0]):
                x = temp_x
                y = start_y + self.circle_spacing * k
                z = 0
                objp[ind] = (x, y, z)
                ind += 1
            temp_x = temp_x + (self.circle_spacing / 2)
        return objp
    
    def get_board(self):
        return {
            'type': 'circle',
            'pattern_size': self.pattern_size,
            'circle_diameter': self.circle_diameter,
            'circle_spacing': self.circle_spacing,
        }
    
    def _find_board(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        
        ret, corners = cv2.findCirclesGrid(
            gray, self.pattern_size, 
            blobDetector=self.detector, 
            flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
        )
        cv2.drawChessboardCorners(image, self.pattern_size, corners, ret)
        
        if ret == True and len(corners):
            return ret, corners, image_size
        return False, None, image_size
    
    def calibrate_by_images(self, images_path, source_type='rgb'):
        """Калибровка по папке с изображениями (круговая доска)"""
        print(f"\n{'='*60}")
        print("Начало калибровки по изображениям (круговая доска)")
        print(f"{'='*60}")
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(images_path, ext)))
            images.extend(glob.glob(os.path.join(images_path, ext.upper())))
        
        if not images:
            print(f"Ошибка: Не найдено изображений в папке {images_path}")
            return None
        
        print(f"Найдено изображений: {len(images)}")
        
        objpoints = []
        imgpoints = []
        image_size = None
        successful_images = []
        
        for idx, img_path in enumerate(images, 1):
            print(f"\nОбработка {idx}/{len(images)}: {os.path.basename(img_path)}")
            
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            ret, corners, img_size = self._find_board(image)
            image_size = img_size
            
            if ret:
                objpoints.append(self.objp)
                imgpoints.append(corners)
                successful_images.append(img_path)
                
                if self.save_images_path:
                    save_name = f"circleboard_calib_{self.image_counter:03d}.jpg"
                    self._save_image(image, save_name)
                    self.image_counter += 1
                
                print(f"  ✓ Круги найдены! Сохранено {len(corners)} точек")
            else:
                print(f"  ✗ Круги не найдены")
        
        print(f"\nСобрано {len(objpoints)} изображений")
        
        if len(objpoints) < 5:
            print("Ошибка: Недостаточно изображений для калибровки")
            return None

        print("\nВыполняется калибровка камеры...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )
        
        if not ret:
            print("Ошибка калибровки!")
            return None

        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ КАЛИБРОВКИ")
        print("="*60)
        print(f"Матрица камеры:\n{mtx}")
        print(f"\nКоэффициенты дисторсии:\n{dist.ravel()}")
        print(f"\nСредняя ошибка перепроецирования: {mean_error:.6f}")
        print("="*60)
        
        return {
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "mean_error": mean_error,
            "image_size": image_size,
            "num_images": len(objpoints),
            "pattern_size": list(self.pattern_size),
            "circle_diameter": self.circle_diameter,
            "circle_spacing": self.circle_spacing,
            "successful_images": successful_images
        }
    
    def calibrate_by_video(self, source, source_type='rgb', all_frames=False):
        """Калибровка по видеопотоку (круговая доска)"""
        print(f"\n{'='*60}")
        print("Начало калибровки по видео (круговая доска)")
        print(f"{'='*60}")
        
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть источник {source}")
            return None
        
        print("Инструкция:")
        print("- Наведите камеру на круговую доску")
        print("- Нажмите ENTER для захвата кадра")
        print("- Нажмите ESC или Q для завершения")
        print()
        
        objpoints = []
        imgpoints = []
        captured_frames = 0
        image_size = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            found, corners, img_size = self._find_board(display_frame)
            image_size = img_size
            
            cv2.putText(display_frame, f"Captured: {captured_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if found:
                cv2.putText(display_frame, "Circle board found!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Circle Board Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13 and found:
                objpoints.append(self.objp)
                imgpoints.append(corners)
                captured_frames += 1
                print(f"Кадр {captured_frames} сохранен")
            elif key == 27 or key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_frames < 5:
            print("Ошибка: Недостаточно кадров")
            return None
        
        print("\nВыполняется калибровка...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )
        
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ КАЛИБРОВКИ")
        print("="*60)
        print(f"Средняя ошибка: {mean_error:.6f}")
        print("="*60)
        
        return {
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "mean_error": mean_error,
            "image_size": image_size,
            "num_images": captured_frames
        }


class Calibration:
    """Основной класс для управления калибровкой"""
    def __init__(self, calib_input, source_type, board_type, save_images, all_frames, 
                 pattern_size=(9, 6), square_size=0.025, circle_params=None):
        self.calib_input = calib_input
        self.input_type = self.get_input_type(calib_input)
        self.all_frames = all_frames
        self.source_type = source_type
        
        if self.input_type == 'video' and isinstance(calib_input, str) and calib_input.isdigit():
            self.calib_input = int(self.calib_input)
        
        board_type = board_type.lower()
        if "chess" in board_type:
            self.board = ChessBoard(pattern_size=pattern_size, square_size=square_size, 
                                   save_images_path=save_images)
        elif "circle" in board_type:
            if circle_params:
                self.board = CircleBoard(
                    pattern_size=circle_params.get('pattern_size', (4, 11)),
                    circle_diameter=circle_params.get('circle_diameter', 0.015),
                    circle_spacing=circle_params.get('circle_spacing', 0.02),
                    save_images_path=save_images
                )
            else:
                self.board = CircleBoard(save_images_path=save_images)
        else:
            print(f"Предупреждение: Неизвестный тип доски '{board_type}', используется шахматная доска")
            self.board = ChessBoard(pattern_size=pattern_size, square_size=square_size,
                                   save_images_path=save_images)
    
    def get_input_type(self, input_data):
        if isinstance(input_data, str):
            if os.path.isdir(input_data):
                return 'images'
            elif os.path.isfile(input_data):
                return 'video'
        return 'video'
    
    def calibrate(self):
        if self.board is None:
            print("Ошибка: калибровочная доска не инициализирована")
            return None
        
        if self.input_type == 'video':
            return self.board.calibrate_by_video(self.calib_input, self.source_type, self.all_frames)
        elif self.input_type == 'images':
            return self.board.calibrate_by_images(self.calib_input, self.source_type)
        else:
            print("Не удалось определить тип источника")
            return None
    
    def save(self, path, data):
        if data is None:
            print("Нет данных для сохранения")
            return
        
        os.makedirs(path, exist_ok=True)
        
        now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        name = f"calibration__{now}.json"
        full_path = os.path.join(path, name)

        save_data = {
            "cameras": [data],
            "board": self.board.get_board() if self.board else None,
            "calibration_date": now
        }
        
        with open(full_path, "w") as json_file:
            json.dump(save_data, json_file, indent=4)
        
        print(f"\nДанные калибровки сохранены в: {full_path}")

        if 'mean_error' in data:
            print("\n" + "-"*60)
            print("ИТОГИ")
            print("="*60)
            print(f"Тип доски: {save_data['board']['type'] if save_data['board'] else 'unknown'}")
            print(f"Средняя ошибка: {data['mean_error']:.6f}")
            if 'image_size' in data:
                print(f"Размер изображения: {data['image_size']}")
            if 'num_images' in data:
                print(f"Использовано изображений: {data['num_images']}")
            print("="*60)
            
            if data['mean_error'] < 0.1:
                print("✓ Отлично! Ошибка калибровки очень низкая.")
            elif data['mean_error'] < 0.3:
                print("✓ Хорошо. Ошибка калибровки приемлемая.")
            elif data['mean_error'] < 0.5:
                print("⚠ Удовлетворительно. Рекомендуется улучшить качество калибровки.")
            else:
                print("✗ Плохо. Рекомендуется повторить калибровку с лучшими изображениями.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Калибровка камеры по шахматной или круговой доске')
    parser.add_argument('--source', '-s', required=True, 
                       help='Источник: 0 для веб-камеры, путь к видео или папке с изображениями')
    parser.add_argument('--source_type', '-t', default='rgb', 
                       help='Тип камеры (по умолчанию: rgb)')
    parser.add_argument('--board_type', '-b', default='chess', 
                       help='Тип доски: chess (шахматная) или circle (круговая)')
    parser.add_argument('--save_path', default='./calibration_results/', 
                       help='Путь для сохранения результатов калибровки')
    parser.add_argument('--save_images', default='./calibration_images/', 
                       help='Путь для сохранения изображений с отмеченными углами')
    parser.add_argument('--all_frames', '-a', action='store_true', 
                       help='Использовать все кадры видео для калибровки')
    
    parser.add_argument('--pattern_width', type=int, default=9,
                       help='Ширина шахматной доски (внутренние углы)')
    parser.add_argument('--pattern_height', type=int, default=6,
                       help='Высота шахматной доски (внутренние углы)')
    parser.add_argument('--square_size', type=float, default=0.025,
                       help='Размер квадрата в метрах (по умолчанию: 0.025)')

    parser.add_argument('--circle_pattern_width', type=int, default=4,
                       help='Ширина круговой сетки')
    parser.add_argument('--circle_pattern_height', type=int, default=11,
                       help='Высота круговой сетки')
    parser.add_argument('--circle_diameter', type=float, default=0.015,
                       help='Диаметр круга в метрах')
    parser.add_argument('--circle_spacing', type=float, default=0.02,
                       help='Расстояние между кругами в метрах')
    
    args = parser.parse_args()
    
    circle_params = {
        'pattern_size': (args.circle_pattern_width, args.circle_pattern_height),
        'circle_diameter': args.circle_diameter,
        'circle_spacing': args.circle_spacing
    } if args.board_type.lower() == 'circle' else None
    
    calibration = Calibration(
        args.source, 
        args.source_type, 
        args.board_type, 
        args.save_images, 
        args.all_frames,
        pattern_size=(args.pattern_width, args.pattern_height),
        square_size=args.square_size,
        circle_params=circle_params
    )
    
    calibration_data = calibration.calibrate()
    calibration.save(args.save_path, calibration_data)