import numpy as np

# игнорирование краёв
def no_padding(arr_img, kernel_height):
    img_height, img_width = arr_img.shape

    img_height = img_height - kernel_height + 1
    img_width = img_width - kernel_height + 1

    return np.zeros((img_height, img_width))

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

# применение обработки края к будущему изображению
def add_padding(arr_img, kernel_height, padding):
    if padding == "no_padding": 
        return no_padding(arr_img, kernel_height)
    else: 
        print("Use another padding\n") # здесь будут дописаны другие варианты обработки края
