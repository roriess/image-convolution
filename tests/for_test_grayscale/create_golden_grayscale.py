import numpy as np
from PIL import Image
from pathlib import Path
from scipy.signal import correlate2d
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.kernels import KERNELS

INPUT_IMAGE = Path("images/image.jpg")
GOLDEN_DIR = Path(__file__).parent / "golden_grayscale"

PADDINGS = {
    "no_padding": None,
    "zero_padding": "constant",
    "replicate_padding": "edge",
    "mirror_padding": "reflect",
    "symmetric_padding": "symmetric",
    "tile_padding": "wrap",
}


def create_golden_images():
    img = Image.open(INPUT_IMAGE).convert("L")
    img_arr = np.asarray(img, dtype=np.float32)

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    for padding_name, pad_mode in PADDINGS.items():
        padding_dir = GOLDEN_DIR / padding_name
        padding_dir.mkdir(exist_ok=True)

        for kernel_name, kernel in KERNELS.items():
            kernel_height, kernel_width = kernel.shape
            padding_height = (kernel_height - 1) // 2
            padding_width = (kernel_width - 1) // 2

            if padding_name == "no_padding":
                conv = correlate2d(img_arr, kernel, mode="valid")
            else:
                if padding_name == "zero_padding":
                    padded = np.pad(
                        img_arr,
                        (
                            (padding_height, padding_height),
                            (padding_width, padding_width),
                        ),
                        mode=pad_mode,
                        constant_values=0,
                    )
                else:
                    padded = np.pad(
                        img_arr,
                        (
                            (padding_height, padding_height),
                            (padding_width, padding_width),
                        ),
                        mode=pad_mode,
                    )
                conv = correlate2d(padded, kernel, mode="valid")

            conv = np.clip(conv, 0, 255).astype(np.uint8)
            output_img = Image.fromarray(conv)
            output_path = padding_dir / f"{kernel_name}.png"
            output_img.save(output_path)


if __name__ == "__main__":
    create_golden_images()
