from convolution import convolution_grayscale, convolution_rgb
import argparse


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

    if args["image_mode"] == "L":
        convolution_grayscale(**vars(args))
    elif args["image_mode"] == "RGB":
        convolution_rgb(**vars(args))
    else:
        raise ValueError(
            f"Unsupported image mode: {args['image_mode']}. Use 'L' for grayscale or 'RGB' for color."
        )


if __name__ == "__main__":
    main()
