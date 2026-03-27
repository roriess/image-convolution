from convolution_grayscale import convolution_grayscale
import argparse


def main():
    parser = argparse.ArgumentParser(description="Convolution image")

    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--kernel", type=str, required=True, help="Selected filter for photo processing"
    )
    parser.add_argument(
        "--padding", type=str, required=True, help="Border processing style"
    )

    args = parser.parse_args()

    convolution_grayscale(args)


if __name__ == "__main__":
    main()
