# <div align="center">Обнаружение номеров автомобилей </div>
## <div align="center"> Описание проекта </div>
![Alt text](resources%2Fimages%2Fyolo9.png)

За основу были взяты модель [YOLO9C](https://github.com/WongKinYiu/yolov9/tree/main), набор данных [nomeroff.net](https://nomeroff.net.ua/) и библиотеки [Ultralytics](https://docs.ultralytics.com/ru/models/yolov9/#introduction-to-yolov9) и PyTorch-2.2.1 Данные были разбиты на подвыборки для обучения, тестирования и валидации:

|Тип выборки данных | Количество изображений |
| :----------------: | :--------------------: |
| train | 20505 (80%) |
| val   | 2563 (10%) |
| test  | 2564 (10%) |

Набор данных, который использовался в данном проекте можно найти на [HuggingFace](https://huggingface.co/datasets/AY000554/Car_plate_detecting_dataset).
<br> Решение по распознаванию обнаруженного номера можно найти в [этом](https://github.com/AY000554/ocr_car_plate) проекте.
<br> Разметка представлена в формате YOLO: ```class x_center y_center width height```.

Примеры изображений номеров:

|![image/png](resources%2Fimages%2F2.jpg) <br> <br> разметка: 0 0.308105 0.796875 0.059570 0.023438|![Alt text](resources%2Fimages%2F63.bmp) <br> <br> разметка: 0 0.754687 0.845833 0.328125 0.104167 |
| :------------------------------------------: | :------------------------------------------: |
|![Alt text](resources%2Fimages%2F255.jpg) <br> <br> разметка: 0 0.605957 0.699854 0.077148 0.046852 |![Alt text](resources%2Fimages%2F323.jpg) <br> <br> разметка: 0 0.414062 0.507812 0.162109 0.080729 |

## <div align="center"> Настройка среды </div>

Для настройки среды используйте ```requirements_yolo9.txt```:
```commandline
pip install -r requirements_yolo9.txt
```
Так же, можно запустить среду в ```Docker```. В таком случае будет использоваться ```requirements_yolo9_for_docker.txt```.
Команда для сборки докера:
```commandline
docker build -t yolo9 .
```
Для работы в контейнере докера нужно подключить к контейнеру папку с данными ```data``` и папку для сохранения логов
и чекпоинтов модели ```logs```, а также папку с конфигурационным файлом ```Config.ini```. Перед запуском кода отредактируйте ```Config.ini ``` под ваш проект.
Пример запуска докер контейнера:
```commandline
docker run -v "$(pwd)"/../data/Car_plate_detecting_dataset:/yolo9/data -v "$(pwd)"/logs:/yolo9/logs -v /yolo9/logs:"$(pwd)"/logs -v "$(pwd)"/Config:/yolo9/Config --shm-size 16gb -it --rm --runtime=nvidia --gpus '"device=0"' yolo9
```

## <div align="center"> Обучение </div>
Для запуска обучения отредактируйте раздел ```Train``` в ```Config.ini``` под свой проект и запустите скрипт ```yolo9_train.py``` в своей среде или докер контейнере:
```commandline
python yolo9_train.py
```
По умолчанию файлы логов обучения записываются в папку ```logs```. Логи сохраняются в формате Tensorboard. Пример команды для запуска Tensorboard:
```commandline
tensorboard --logdir logs --port 6015
```

## <div align="center"> Тестирование </div>
Для запуска тестирования отредактируйте раздел ```Test``` в ```Config.ini``` под свой проект и запустите скрипт ```yolo9_test.py``` в своей среде или докер контейнере:
```commandline
python yolo9_test.py
```
Результаты тестирования сохраняются в папке с моделью.


## <div align="center">  Результаты тестирования обученной модели </div>
Модель обучалась на протяжении 100 эпох с размером батча 32 и размером цветных изображений 640 х 640 пикселей. Значение шага обучения изменялось от 1e-3 до 1e-10 по закону косинусного распада (CosineDecay) с прогревом (warmup) в 4 эпохи.
Результаты тестирования на тестовой выборке данных:

| Название метрики | Доля|
| :--------------- | :-----: |
| Отношение ошибочно распознанных и не распознанных <br> номеров к общему количеству номеров | 0,0391 |

Тестирование проводилось при пороговых значениях confidence=0.7, IoU=0.6.

![Alt text](logs%2FYOLO9c_CPD_ru%2Fconfusion_matrix_.png)


Параметры модели:

| Название параметра | Значение|
| :--------------- | :-----: |
| Размер в формате ONNX |  101.6 MB|
| Размер в формате pt |  51.6 MB|
| Количество параметров | 25.3 M |
| FLOPs | 102.1 G |

