from PIL import Image
import numpy as np

from src.kernels import KERNELS
from src.padding import PADDINGS, add_padding


def _convolve_array(arr_img: np.ndarray, kernel: str, padding: str) -> np.ndarray:
    kernel_height, _ = KERNELS[kernel].shape
    padded = add_padding(arr_img, kernel_height, PADDINGS[padding])

    if padding == "no_padding":
        new_img_height = padded.shape[0]
        new_img_width = padded.shape[1]
        source = arr_img
    else:
        new_img_height = padded.shape[0] - kernel_height + 1
        new_img_width = padded.shape[1] - kernel_height + 1
        source = padded

    new_img = np.zeros((new_img_height, new_img_width), dtype=np.float32)

    kernel_flat = KERNELS[kernel].ravel()
    for i in range(new_img_height):
        for j in range(new_img_width):
            new_img[i, j] = np.dot(
                source[i : i + kernel_height, j : j + kernel_height].ravel(),
                kernel_flat,
            )

    return new_img


def _save_image(new_img: np.ndarray, output_dir: str):
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    result = Image.fromarray(new_img)
    result.save(output_dir)


def convolution_grayscale(
    arr_img: np.ndarray, kernel: str, padding: str, output_dir: str
):
    new_img = _convolve_array(arr_img, kernel, padding)
    _save_image(new_img, output_dir)


def convolution_rgb(arr_img: np.ndarray, kernel: str, padding: str, output_dir: str):
    result_channels = []

    for c in range(3):
        channel = arr_img[:, :, c]
        new_img = _convolve_array(channel, kernel, padding)
        result_channels.append(new_img)

    new_img = np.dstack(result_channels)
    _save_image(new_img, output_dir)
