from src.convolution import convolution_grayscale, convolution_rgb
import argparse
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Convolution image")

    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--image_mode",
        type=str,
        required=True,
        help="Image color mode: 'L' for grayscale, 'RGB' for color",
    )
    parser.add_argument(
        "--kernel", type=str, required=True, help="Selected filter for photo processing"
    )
    parser.add_argument(
        "--padding", type=str, required=True, help="Border processing style"
    )

    args = parser.parse_args()

    if args.image_mode != "L" and args.image_mode != "RGB":
        raise ValueError(
            f"Unsupported image mode: {args['image_mode']}. Use 'L' for grayscale or 'RGB' for color."
        )

    img = Image.open(args.input_dir).convert(args.image_mode)
    arr_img = np.asarray(img, dtype=np.float32)

    func = convolution_grayscale if args.image_mode == "L" else convolution_rgb
    new_img = func(
        arr_img=arr_img,
        kernel=args.kernel,
        padding=args.padding,
    )

    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    result = Image.fromarray(new_img)
    result.save(args.output_dir)


if __name__ == "__main__":
    main()
