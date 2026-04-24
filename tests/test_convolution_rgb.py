import numpy as np
import pytest
from pathlib import Path
from PIL import Image


from src.convolution import convolution_rgb

GOLDEN_DIR = Path("tests/for_test_rgb/golden_rgb")


@pytest.fixture
def args():
    return {
        "input_dir": "images/image.jpg",
        "output_dir": "tests/for_test_rgb/test_images/",
        "kernel": "",
        "padding": "",
    }


COMBINATIONS = [
    ("embossing_3x3", "zero_padding"),
    ("gaussian_blur_5x5", "replicate_padding"),
    ("blur_5x5", "mirror_padding"),
    ("embossing_3x3", "tile_padding"),
]


@pytest.mark.parametrize("kernel_name, padding_name", COMBINATIONS)
def test_kernel(padding_name, kernel_name, args):
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"result_{kernel_name}_{padding_name}.png"

    img = Image.open(args["input_dir"]).convert("RGB")
    arr_img = np.asarray(img, dtype=np.float32)

    new_img = convolution_rgb(
        arr_img=arr_img,
        kernel=kernel_name,
        padding=padding_name,
    )

    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    result = Image.fromarray(new_img)
    result.save(output_file)

    assert output_file.exists(), f"Output file not found: {output_file}"

    result_img = Image.open(output_file).convert("RGB")
    result = np.asarray(result_img)

    expected_path = GOLDEN_DIR / f"{kernel_name}_{padding_name}.png"
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
