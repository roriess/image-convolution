import sys
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.convolution import convolution_rgb
from src.kernels import KERNELS
from src.padding import PADDINGS

GOLDEN_DIR = Path(__file__).parent / "for_test_rgb/golden_rgb"


@pytest.fixture
def args():
    return {
        "input_dir": "images/image.jpg",
        "output_dir": "tests/for_test_rgb/test_images/",
        "kernel": "",
        "padding": "",
    }


@pytest.mark.parametrize("kernel_name", KERNELS.keys())
@pytest.mark.parametrize("padding_name", PADDINGS)
def test_kernel(padding_name, kernel_name, args):
    output_dir = Path(args["output_dir"]) / padding_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"result_{kernel_name}.png"

    test_args = args.copy()
    test_args["kernel"] = kernel_name
    test_args["padding"] = padding_name
    test_args["output_dir"] = str(output_file)

    convolution_rgb(**test_args)
    assert output_file.exists(), f"Output file not found: {output_file}"

    result_img = Image.open(output_file).convert("RGB")
    result = np.asarray(result_img)

    expected_path = GOLDEN_DIR / padding_name / f"{kernel_name}.png"
    assert expected_path.exists(), f"Golden file not found: {expected_path}"

    expected_img = Image.open(expected_path).convert("RGB")
    expected = np.asarray(expected_img)

    np.testing.assert_allclose(
        result,
        expected,
        atol=1,
        rtol=0,
        err_msg=f"Mismatch for kernel {kernel_name}, padding {padding_name}",
    )
