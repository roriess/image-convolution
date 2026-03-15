from PIL import Image 
import numpy as np
from kernels import *
from padding import *
import os

def convert_to_GS(arr_img):
    gray = 0.299 * arr_img[:,:,0] + 0.587 * arr_img[:,:,1] + 0.114 * arr_img[:,:,2] # коэффициенты luminosity
    return gray.astype(np.uint8)

def convolution_GS(image, kernel, padding):
    img = os.path.join("images", image)
    img = Image.open(img)

    arr_img = np.array(img)
    if len(arr_img.shape) == 3: 
        arr_img = convert_to_GS(arr_img)

    kernel_height, _ = kernel.shape

    new_img = add_padding(arr_img, kernel_height, padding)
    img_height, img_width = new_img.shape

    for i in range(img_height):
        for j in range(img_width):
            new_img[i][j] = np.sum(arr_img[i:i + kernel_height, j:j + kernel_height] * kernel)

    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    new_img = Image.fromarray(new_img)

    new_img.save("results/image.jpeg")
