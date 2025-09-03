import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime

def test_on_hard_cases(model_path, hard_cases_dir):
    # Загружаем модель
    model = YOLO(model_path)
    
    # Создаем директорию для сохранения результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f'results/hard_cases_test_{timestamp}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Получаем список изображений
    hard_cases = list(Path(hard_cases_dir).glob('*.jpg'))
    total_images = len(hard_cases)
    
    print(f"\nТестирование на {total_images} сложных случаях...")
    
    # Статистика детекций
    detections = {
        'detected': 0,
        'not_detected': 0,
        'confidences': []
    }
    
    # Обработка каждого изображения
    for idx, img_path in enumerate(hard_cases, 1):
        print(f"\nОбработка изображения {idx}/{total_images}: {img_path.name}")
        
        # Получаем предсказания
        results = model.predict(str(img_path), save=False, verbose=False)[0]
        
        # Загружаем изображение для визуализации
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Проверяем наличие детекций
        if len(results.boxes) > 0:
            detections['detected'] += 1
            
            # Сохраняем уверенность модели
            confidences = results.boxes.conf.cpu().numpy()
            detections['confidences'].extend(confidences)
            
            # Отрисовка боксов на изображении
            for box, conf in zip(results.boxes.xyxy.cpu().numpy(), confidences):
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Рисуем бокс
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Добавляем текст с уверенностью
                cv2.putText(img, f'{conf:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            detections['not_detected'] += 1
        
        # Сохраняем результат
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Детекции: {len(results.boxes)}, Макс. уверенность: {max(confidences) if len(results.boxes) > 0 else 0:.2f}')
        plt.savefig(save_dir / f'{img_path.stem}_result.jpg')
        plt.close()
    
    # Выводим итоговую статистику
    print("\nИтоговая статистика:")
    print(f"Всего изображений: {total_images}")
    print(f"Успешных детекций: {detections['detected']} ({detections['detected']/total_images*100:.1f}%)")
    print(f"Неудачных детекций: {detections['not_detected']} ({detections['not_detected']/total_images*100:.1f}%)")
    
    if detections['confidences']:
        conf_array = np.array(detections['confidences'])
        print(f"\nСтатистика по уверенности модели:")
        print(f"Средняя уверенность: {np.mean(conf_array):.3f}")
        print(f"Медианная уверенность: {np.median(conf_array):.3f}")
        print(f"Мин. уверенность: {np.min(conf_array):.3f}")
        print(f"Макс. уверенность: {np.max(conf_array):.3f}")
        
        # Строим гистограмму уверенности
        plt.figure(figsize=(10, 6))
        plt.hist(conf_array, bins=20, range=(0, 1))
        plt.title('Распределение уверенности модели')
        plt.xlabel('Уверенность')
        plt.ylabel('Количество детекций')
        plt.savefig(save_dir / 'confidence_distribution.jpg')
        plt.close()
    
    print(f"\nРезультаты сохранены в директории: {save_dir}")

if __name__ == "__main__":
    # Путь к модели и директории с тестовыми изображениями
    MODEL_PATH = '/workspaces/Car_plate_detecting/best (1).pt'
    HARD_CASES_DIR = '/workspaces/Car_plate_detecting/hard_cases'
    
    test_on_hard_cases(MODEL_PATH, HARD_CASES_DIR)
