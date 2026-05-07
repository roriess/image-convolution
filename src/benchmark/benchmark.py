import pytest
import numpy as np
from src.convolution import convolution_grayscale, convolution_rgb
from src.kernels import KERNELS
import cv2

SIZES = [128, 512, 2048]

COMBINATIONS_GRAY = [
    ("embossing_3x3", "zero_padding"),
    ("blur_5x5", "mirror_padding"),
]

COMBINATIONS_RGB = [
    ("sharpen_3x3", "replicate_padding"),
    ("gaussian_blur_5x5", "zero_padding"),
]

PADDING_FOR_OPENCV = {
    "zero_padding": cv2.BORDER_CONSTANT,
    "replicate_padding": cv2.BORDER_REPLICATE,
    "mirror_padding": cv2.BORDER_REFLECT_101,
}


@pytest.fixture(params=SIZES, ids=[f"{size}x{size}" for size in SIZES])
def size(request):
    return request.param


@pytest.fixture
def grayscale_image(size):
    return np.random.rand(size, size).astype(np.float32)


@pytest.fixture
def rgb_image(size):
    return np.random.rand(size, size, 3).astype(np.float32)


@pytest.mark.parametrize("kernel,padding", COMBINATIONS_GRAY)
def test_grayscale_edu(benchmark, grayscale_image, kernel, padding):
    benchmark(convolution_grayscale, grayscale_image, kernel, padding)


@pytest.mark.parametrize("kernel,padding", COMBINATIONS_GRAY)
def test_grayscale_cv(benchmark, grayscale_image, kernel, padding):
    border = PADDING_FOR_OPENCV.get(padding, cv2.BORDER_DEFAULT)

    def run():
        cv2.filter2D(grayscale_image, -1, KERNELS[kernel], borderType=border)

    benchmark(run)


@pytest.mark.parametrize("kernel,padding", COMBINATIONS_RGB)
def test_rgb_edu(benchmark, rgb_image, kernel, padding):
    benchmark(convolution_rgb, rgb_image, kernel, padding)


@pytest.mark.parametrize("kernel,padding", COMBINATIONS_RGB)
def test_rgb_cv(benchmark, rgb_image, kernel, padding):
    border = PADDING_FOR_OPENCV.get(padding, cv2.BORDER_DEFAULT)

    def run():
        cv2.filter2D(rgb_image, -1, KERNELS[kernel], borderType=border)

    benchmark(run)
