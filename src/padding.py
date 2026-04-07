import numpy as np


# игнорирование краёв
def no_padding(arr_img: np.ndarray, kernel_height: np.ndarray) -> np.ndarray:
    img_height, img_width = arr_img.shape

    img_height = img_height - kernel_height + 1
    img_width = img_width - kernel_height + 1

    return np.zeros((img_height, img_width), dtype=np.float32)


# дополнение нулями
def zero_padding(arr_img):
    pass


# дополнение константой
def constant_padding(arr_img):
    pass


# копирование крайних пикселей
def replicate(arr_img):
    pass


# копирование крайних пикселей
def mirror_padding(arr_img):
    pass


# циклическое продолжение
def tile_padding(arr_img):
    pass


# симметрическое отражение
def symmetric_padding(arr_img):
    pass


PADDINGS = {
    "no_padding": no_padding,
    "zero_padding": zero_padding,
    "constant_padding": constant_padding,
    "replicate": replicate,
    "mirror_padding": mirror_padding,
    "tile_padding": tile_padding,
    "symmetric_padding": symmetric_padding,
}


# применение обработки края к будущему изображению
def add_padding(arr_img: np.ndarray, kernel_height: np.ndarray, func) -> np.ndarray:
    return func(arr_img, kernel_height)
