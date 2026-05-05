import numpy as np
from typing import Callable


# игнорирование краёв
def no_padding(arr_img: np.ndarray, kernel_height: int) -> np.ndarray:
    arr_img_height, arr_img_width = arr_img.shape

    padding = kernel_height - 1
    img_height = arr_img_height - padding
    img_width = arr_img_width - padding

    new_img = np.zeros((img_height, img_width), dtype=np.float32)
    new_img[:, :] = arr_img[
        padding : img_height + padding, padding : img_width + padding
    ]

    return new_img


# дополнение нулями
def zero_padding(arr_img: np.ndarray, kernel_height: int) -> np.ndarray:
    arr_img_height, arr_img_width = arr_img.shape

    padding = (kernel_height - 1) // 2
    img_height = arr_img_height + 2 * padding
    img_width = arr_img_width + 2 * padding

    new_img = np.zeros((img_height, img_width), dtype=np.float32)
    new_img[padding : padding + arr_img_height, padding : padding + arr_img_width] = (
        arr_img
    )

    return new_img


# копирование крайних пикселей
def replicate_padding(arr_img: np.ndarray, kernel_height: int) -> np.ndarray:
    _, arr_img_width = arr_img.shape
    new_img = zero_padding(arr_img, kernel_height)
    img_height, img_width = new_img.shape
    padding = (kernel_height - 1) // 2

    for i in range(padding):
        new_img[i, padding : padding + arr_img_width] = arr_img[0, :]

    for i in range(img_height - padding, img_height):
        new_img[i, padding : padding + arr_img_width] = arr_img[-1, :]

    for j in range(padding):
        new_img[:, j] = new_img[:, padding]

    for j in range(img_width - padding, img_width):
        new_img[:, j] = new_img[:, img_width - padding - 1]

    return new_img


# зеркальное копирование пикселей без дублирования крайних
def mirror_padding(arr_img: np.ndarray, kernel_height: int) -> np.ndarray:
    arr_img_height, arr_img_width = arr_img.shape
    new_img = zero_padding(arr_img, kernel_height)
    padding = (kernel_height - 1) // 2

    for i in range(padding):
        new_img[padding - i - 1, padding : padding + arr_img_width] = arr_img[i + 1, :]
        new_img[padding + arr_img_height + i, padding : padding + arr_img_width] = (
            arr_img[arr_img_height - 2 - i, :]
        )

    for j in range(padding):
        new_img[:, padding - j - 1] = new_img[:, padding + j + 1]
        new_img[:, padding + arr_img_width + j] = new_img[
            :, padding + arr_img_width - j - 2
        ]

    return new_img


# зеркальное копирование пикселей с дублированием крайних
def symmetric_padding(arr_img: np.ndarray, kernel_height: int) -> np.ndarray:
    arr_img_height, arr_img_width = arr_img.shape
    new_img = zero_padding(arr_img, kernel_height)
    padding = (kernel_height - 1) // 2

    for i in range(padding):
        new_img[padding - i - 1, padding : padding + arr_img_width] = arr_img[i, :]
        new_img[padding + arr_img_height + i, padding : padding + arr_img_width] = (
            arr_img[arr_img_height - 1 - i, :]
        )

    for j in range(padding):
        new_img[:, padding - j - 1] = new_img[:, padding + j]
        new_img[:, padding + arr_img_width + j] = new_img[
            :, padding + arr_img_width - 1 - j
        ]

    return new_img


# циклическое копирование пикселей
def tile_padding(arr_img: np.ndarray, kernel_height: int) -> np.ndarray:
    arr_img_height, arr_img_width = arr_img.shape
    new_img = zero_padding(arr_img, kernel_height)
    padding = (kernel_height - 1) // 2

    for i in range(padding):
        new_img[padding - i - 1, padding : padding + arr_img_width] = arr_img[
            arr_img_height - i - 1, :
        ]
        new_img[padding + arr_img_height + i, padding : padding + arr_img_width] = (
            arr_img[i, :]
        )

    for j in range(padding):
        new_img[:, padding - j - 1] = new_img[:, padding + arr_img_width - j - 1]
        new_img[:, padding + arr_img_width + j] = new_img[:, padding + j]

    return new_img


PADDINGS = {
    "no_padding": no_padding,
    "zero_padding": zero_padding,
    "replicate_padding": replicate_padding,
    "mirror_padding": mirror_padding,
    "tile_padding": tile_padding,
    "symmetric_padding": symmetric_padding,
}


# применение обработки края к будущему изображению
def add_padding(arr_img: np.ndarray, kernel_height: int, func: Callable) -> np.ndarray:
    return func(arr_img, kernel_height)
