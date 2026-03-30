from PIL import Image
import numpy as np
from src.kernels import KERNELS
from src.padding import PADDINGS, add_padding


def convolution_grayscale(args):
    img = Image.open(args.input_dir).convert("L")
    arr_img = np.asarray(img, dtype=np.float32)

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
