import time
import numpy as np
import matplotlib.pyplot as plt
from src.convolution import convolution_grayscale, convolution_rgb
from src.kernels import KERNELS
import cv2
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    edu: List[float]
    cv: List[float]


sizes = [128, 1024, 4096]

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


def _measure_convolution(
    img: np.ndarray, kernel_name: str, padding_name: str, func: Callable
) -> tuple[float, float]:
    kernel = KERNELS[kernel_name]
    border = PADDING_FOR_OPENCV.get(padding_name, cv2.BORDER_DEFAULT)

    start = time.perf_counter()
    func(img, kernel_name, padding_name)
    edu_time = time.perf_counter() - start

    start = time.perf_counter()
    cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=border)
    cv_time = time.perf_counter() - start

    return edu_time, cv_time


def _testing_image(
    combinations: List[Tuple[str, str]], func: Callable
) -> Dict[Tuple[str, str], BenchmarkResult]:
    results = {
        combination: BenchmarkResult(edu=[], cv=[]) for combination in combinations
    }
    for size in sizes:
        print(f"\nTesting {func.__name__}, size {size}x{size}")
        if func == convolution_rgb:
            img = np.random.rand(size, size, 3).astype(np.float32)
        else:
            img = np.random.rand(size, size).astype(np.float32)

        for combination in combinations:
            kernel, padding = combination
            educational, opencv = _measure_convolution(img, kernel, padding, func)
            results[combination].edu.append(educational)
            results[combination].cv.append(opencv)
            print(f"  {func.__name__}: {kernel}+{padding} done")

    for combination in combinations:
        results[combination].edu = [time * 1000 for time in results[combination].edu]
        results[combination].cv = [time * 1000 for time in results[combination].cv]

    return results


def _print_results(
    title: str,
    results: Dict[Tuple[str, str], BenchmarkResult],
    combinations: List[Tuple[str, str]],
    sizes: List[int],
):
    print(f"\n{title}")
    for combination in combinations:
        print(f"{combination[0]} + {combination[1]}:")
        for index, size in enumerate(sizes):
            educational = results[combination].edu[index]
            opencv = results[combination].cv[index]
            print(
                f"  {size}x{size}: Educational = {educational:.3f} ms, OpenCV = {opencv:.3f} ms"
            )


def _visualization(
    results_gray: Dict[Tuple[str, str], BenchmarkResult],
    results_rgb: Dict[Tuple[str, str], BenchmarkResult],
):
    # список [(kernel, padding, img_type), ...]
    all_info = []
    for combination in COMBINATIONS_GRAY:
        all_info.append((combination[0], combination[1], "Grayscale"))
    for combination in COMBINATIONS_RGB:
        all_info.append((combination[0], combination[1], "RGB"))

    # список объектов BenchmarkResult
    all_results = []
    for combination in COMBINATIONS_GRAY:
        all_results.append(results_gray[combination])
    for combination in COMBINATIONS_RGB:
        all_results.append(results_rgb[combination])

    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for index, (info, result) in enumerate(zip(all_info, all_results)):
        axis = axes[index]
        kernel, padding, img_type = info
        edu_values = result.edu
        cv_values = result.cv
        column_position = np.arange(len(sizes))
        width = 0.35
        bar_1 = axis.bar(
            column_position - width / 2,
            edu_values,
            width,
            label="Educational",
            color="#1f77b4",
        )
        bar_2 = axis.bar(
            column_position + width / 2,
            cv_values,
            width,
            label="OpenCV",
            color="#ff7f0e",
        )
        axis.bar_label(bar_1, fmt="%.3f", padding=1)
        axis.bar_label(bar_2, fmt="%.3f", padding=1)
        axis.set_xticks(column_position)
        axis.set_xticklabels(sizes)
        axis.set_yscale("log")
        axis.set_xlabel("Image size")
        axis.set_ylabel("Time (ms)")
        axis.set_title(f"{kernel}\n{padding}\n({img_type})")
        axis.legend()
        axis.grid(axis="y", linestyle="--", alpha=0.7)

    plt.suptitle("Performance comparison: Educational vs OpenCV", fontsize=14)
    plt.tight_layout()
    plt.savefig("src/benchmark/benchmark_graphs.png", dpi=150)
    plt.show()

    print("\nRESULTS")
    _print_results("Grayscale", results_gray, COMBINATIONS_GRAY, sizes)
    _print_results("RGB", results_rgb, COMBINATIONS_RGB, sizes)


def run_benchmark():
    results_gray = _testing_image(COMBINATIONS_GRAY, convolution_grayscale)
    results_rgb = _testing_image(COMBINATIONS_RGB, convolution_rgb)

    _visualization(results_gray, results_rgb)


if __name__ == "__main__":
    run_benchmark()
