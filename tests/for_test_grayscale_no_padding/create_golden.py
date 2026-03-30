import numpy as np
from PIL import Image
from pathlib import Path
from scipy.signal import correlate2d
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.kernels import KERNELS

INPUT_IMAGE = Path("images/image.jpg")
GOLDEN_DIR = Path(__file__).parent / "golden"


# функция для генерирвции golden изображений
def create_golden_images():
    img = Image.open(INPUT_IMAGE).convert("L")

    img_arr = np.asarray(img, dtype=np.float32)

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    for kernel_name, kernel in KERNELS.items():
        conv = correlate2d(img_arr, kernel, mode="valid", boundary="fill", fillvalue=0)
        conv = np.clip(conv, 0, 255).astype(np.uint8)

        output_img = Image.fromarray(conv)
        output_path = GOLDEN_DIR / f"{kernel_name}.png"
        output_img.save(output_path)


if __name__ == "__main__":
    create_golden_images()
