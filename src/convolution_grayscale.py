from PIL import Image 
import numpy as np
from kernels import *
from padding import *
import sys
import os 

def convert_to_grayscale(arr_img: int) -> int:
    gray = 0.299 * arr_img[:,:,0] + 0.587 * arr_img[:,:,1] + 0.114 * arr_img[:,:,2] # коэффициенты luminosity
    return gray.astype(np.uint8)

def convolution_grayscale():
    if len(sys.argv) < 4:
        print("Не указан файл, ядро или способ обработки края.\n")
        return
    else:
        img = os.path.join("images", sys.argv[1]) # перед запуском программы необходимо добавить изображение в images/
        kernel = sys.argv[2]
        padding = sys.argv[3]
    img = Image.open(img)

    arr_img = np.array(img)
    if len(arr_img.shape) == 3: 
        arr_img = convert_to_grayscale(arr_img)

    kernel_height, _ = KERNELS[kernel].shape

    new_img = add_padding(arr_img, kernel_height, PADDINGS[padding])
    img_height, img_width = new_img.shape

    for i in range(img_height):
        for j in range(img_width):
            new_img[i][j] = np.sum(arr_img[i:i + kernel_height, j:j + kernel_height] * KERNELS[kernel])

    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    new_img = Image.fromarray(new_img)

    new_img.save("results/image.jpeg")
