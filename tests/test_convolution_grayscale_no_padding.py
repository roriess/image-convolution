import sys
import numpy as np
import pytest
import argparse
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.convolution_grayscale import convolution_grayscale
from src.kernels import KERNELS

GOLDEN_DIR = Path(__file__).parent / "for_test_grayscale_no_padding/golden"


@pytest.fixture
def args():
    return argparse.Namespace(
        input_dir="images/image.jpg",
        output_dir="tests/for_test_grayscale_no_padding/test_images/",
        kernel="",
        padding="no_padding",
    )


@pytest.mark.parametrize("kernel_name", KERNELS.keys())
def test_kernel(kernel_name, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"result_{kernel_name}.png"

    test_args = argparse.Namespace(**vars(args))
    test_args.kernel = kernel_name
    test_args.output_dir = str(output_file)

    convolution_grayscale(test_args)
    assert output_file.exists(), f"Output file not found: {output_file}"

    result_img = Image.open(output_file).convert("L")
    result = np.asarray(result_img)

    expected_path = GOLDEN_DIR / f"{kernel_name}.png"
    assert expected_path.exists(), f"Golden file not found: {expected_path}"

    expected_img = Image.open(expected_path).convert("L")
    expected = np.asarray(expected_img)

    np.testing.assert_allclose(
        result, expected, atol=1, rtol=0, err_msg=f"Mismatch for kernel {kernel_name}"
    )
