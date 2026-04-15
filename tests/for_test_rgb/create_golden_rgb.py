import numpy as np
from PIL import Image
from pathlib import Path
from scipy.signal import correlate2d
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.kernels import KERNELS

INPUT_IMAGE = Path("images/image.jpg")
GOLDEN_DIR = Path(__file__).parent / "golden_rgb"


PADDINGS = {
    "zero_padding": "constant",
    "replicate_padding": "edge",
    "mirror_padding": "reflect",
    "tile_padding": "wrap",
}

COMBINATIONS = [
    ("embossing_3x3", "zero_padding"),
    ("gaussian_blur_5x5", "replicate_padding"),
    ("blur_5x5", "mirror_padding"),
    ("embossing_3x3", "tile_padding"),
]


def create_golden_images():
    img = Image.open(INPUT_IMAGE).convert("RGB")
    img_arr = np.asarray(img, dtype=np.float32)

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    for kernel_name, padding_name in COMBINATIONS:
        kernel = KERNELS[kernel_name]
        kernel_height, kernel_width = kernel.shape
        pad_h = (kernel_height - 1) // 2
        pad_w = (kernel_width - 1) // 2
        pad_mode = PADDINGS[padding_name]

        result_channels = []
        for c in range(3):
            channel = img_arr[:, :, c]

            if padding_name == "zero_padding":
                padded = np.pad(
                    channel,
                    ((pad_h, pad_h), (pad_w, pad_w)),
                    mode=pad_mode,
                    constant_values=0,
                )
            else:
                padded = np.pad(
                    channel,
                    ((pad_h, pad_h), (pad_w, pad_w)),
                    mode=pad_mode,
                )
            conv = correlate2d(padded, kernel, mode="valid")
            result_channels.append(conv)

        result = np.dstack(result_channels)
        result = np.clip(result, 0, 255).astype(np.uint8)
        output_img = Image.fromarray(result)
        output_path = GOLDEN_DIR / f"{kernel_name}_{padding_name}.png"
        output_img.save(output_path)


if __name__ == "__main__":
    create_golden_images()
