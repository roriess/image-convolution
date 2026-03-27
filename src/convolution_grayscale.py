from PIL import Image
import numpy as np
from kernels import KERNELS
from padding import PADDINGS, add_padding


def convert_to_grayscale(arr_img: np.ndarray) -> np.ndarray:
    gray = (
        0.299 * arr_img[:, :, 0] + 0.587 * arr_img[:, :, 1] + 0.114 * arr_img[:, :, 2]
    )  # коэффициенты luminosity
    return gray.astype(np.uint8)


def convolution_grayscale(args):
    img = Image.open(args.input_dir)

    arr_img = np.array(img)
    if len(arr_img.shape) == 3:
        arr_img = convert_to_grayscale(arr_img)

    kernel_height, _ = KERNELS[args.kernel].shape

    new_img = add_padding(arr_img, kernel_height, PADDINGS[args.padding])
    img_height, img_width = new_img.shape

    for i in range(img_height):
        for j in range(img_width):
            new_img[i][j] = np.sum(
                arr_img[i : i + kernel_height, j : j + kernel_height]
                * KERNELS[args.kernel]
            )

    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    new_img = Image.fromarray(new_img)

    new_img.save(args.output_dir)
