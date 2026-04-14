from PIL import Image
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kernels import KERNELS
from src.padding import PADDINGS, add_padding


def convolution_grayscale(**args):
    img = Image.open(args["input_dir"]).convert("L")
    arr_img = np.asarray(img, dtype=np.float32)

    kernel_height, _ = KERNELS[args["kernel"]].shape

    padded = add_padding(arr_img, kernel_height, PADDINGS[args["padding"]])

    if args["padding"] == "no_padding":
        new_img_height = padded.shape[0]
        new_img_width = padded.shape[1]
        source = arr_img
    else:
        new_img_height = padded.shape[0] - kernel_height + 1
        new_img_width = padded.shape[1] - kernel_height + 1
        source = padded

    new_img = np.zeros((new_img_height, new_img_width), dtype=np.float32)

    kernel_flat = KERNELS[args["kernel"]].ravel()
    for i in range(new_img_height):
        for j in range(new_img_width):
            new_img[i, j] = np.dot(
                source[i : i + kernel_height, j : j + kernel_height].ravel(),
                kernel_flat,
            )

    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    new_img = Image.fromarray(new_img)

    new_img.save(args["output_dir"])


def convolution_rgb(**args):
    img = Image.open(args["input_dir"]).convert("RGB")
    arr_img = np.asarray(img, dtype=np.float32)

    kernel_height, _ = KERNELS[args["kernel"]].shape

    result_channels = []

    for c in range(3):
        channel = arr_img[:, :, c]

        padded = add_padding(channel, kernel_height, PADDINGS[args["padding"]])

        if args["padding"] == "no_padding":
            new_img_height = padded.shape[0]
            new_img_width = padded.shape[1]
            source = channel
        else:
            new_img_height = padded.shape[0] - kernel_height + 1
            new_img_width = padded.shape[1] - kernel_height + 1
            source = padded

        new_img = np.zeros((new_img_height, new_img_width), dtype=np.float32)

        kernel_flat = KERNELS[args["kernel"]].ravel()
        for i in range(new_img_height):
            for j in range(new_img_width):
                new_img[i, j] = np.dot(
                    source[i : i + kernel_height, j : j + kernel_height].ravel(),
                    kernel_flat,
                )

        result_channels.append(new_img)

    new_img = np.dstack(result_channels)
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    new_img = Image.fromarray(new_img)

    new_img.save(args["output_dir"])
