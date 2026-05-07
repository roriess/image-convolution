# image-convolution

Учебная практика по свертке изображений. 1 курс 2 семестр.

## Структура

*   `images/`: хранит исходные изображения.
*   `results/`: хранит результаты свёртки.
*   `src/`: содержит весь исходный код.
    *   `benchmark/`: модуль для тестирования производительности.
        *   `benchmark.py`: скрипт запуска бенчмарка (сравнение Educational и OpenCV).
        *   `benchmark_visualization.png`: график сравнения времени выполнения.
    *   `convolution.py`: реализация свёртка (Educational).
    *   `kernels.py`: набор ядер свёртки.
    *   `main.py`: точка входа в программу.
    *   `padding.py`: набор вариантов обработки края.
*   `tests/`
    *   `for_test_grayscale/`
        *   `golden_grayscale/`: эталонные изображения для ч/б свёртки.
        *   `create_golden_grayscale.py`: генерация эталонов.
    *   `for_test_rgb/`
        *   `golden_rgb/`: эталонные изображения для цветной свёртки.
        *   `create_golden_rgb.py`: генерация эталонов для RGB.
    *   `test_convolution_grayscale.py`: pytest‑тесты для ч/б изображений.
    *   `test_convolution_rgb.py`: pytest‑тесты для цветных изображений.

## Доступные ядра
### 3x3:
*   `sharpen_3x3`: повышение резкости.
*   `blur_3x3`: усредняющее размытие.
*   `gaussian_blur_3x3`: размытие по Гауссу.
*   `highlighting_vertical_borders_3x3`: выделение вертикальных границ.
*   `highlighting_horizontal_borders_3x3`: выделение горизонтальных границ.
*   `embossing_3x3`:  эффект тиснения.
### 5х5:
*   `blur_5x5`: усредняющее размытие.
*   `gaussian_blur_5x5`: размытие по Гауссу.


## Доступные варианты обработки края
Размер итогового изображения меньше размера исходного:
*   `no_padding`: игнорирование краёв.
Размер итогового изображения равен размеру исходного:
*   `zero_padding`: дополнение нулями.
*   `replicate_padding`: копирование крайних пикселей.
*   `mirror_padding`: отражение без дублирования границы.
*   `symmetric_padding`: симметричное отражение с дублированием границы.
*   `tile_padding`: циклическое продолжение.

##
Запуск программы: 
Для свертки изображений:
```markdown
python -m src.main.py run --input_dir INPUT_DIR --output_dir --image_mode IMAGE_MODE OUTPUT_DIR --kernel KERNEL --padding PADDING
```
Для теста производительности:
```markdown
python -m src.main.py benchmark
```
Справка: 
```markdown
python -m src.main run --help
```
```markdown
python -m src.main benchmark --help
```

## 
Изображение для примера взято с данного сайта: https://www.publicdomainarchive.com/ \
Изображения на сайте распространяются под лицензией Public Domain.
