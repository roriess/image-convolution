import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import numpy as np
import matplotlib.pyplot as plt
from src.convolution import convolution_grayscale, convolution_rgb
from src.kernels import KERNELS
import cv2

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


def measure_gray(
    img: np.ndarray, kernel_name: str, padding_name: str
) -> tuple[float, float]:
    kernel = KERNELS[kernel_name]
    border = PADDING_FOR_OPENCV.get(padding_name, cv2.BORDER_DEFAULT)

    start = time.perf_counter()
    convolution_grayscale(img, kernel_name, padding_name)
    edu_time = time.perf_counter() - start

    start = time.perf_counter()
    cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=border)
    cv_time = time.perf_counter() - start

    return edu_time, cv_time


def measure_rgb(
    img: np.ndarray, kernel_name: str, padding_name: str
) -> tuple[float, float]:
    kernel = KERNELS[kernel_name]
    border = PADDING_FOR_OPENCV.get(padding_name, cv2.BORDER_DEFAULT)

    start = time.perf_counter()
    convolution_rgb(img, kernel_name, padding_name)
    educational_time = time.perf_counter() - start

    start = time.perf_counter()
    cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=border)
    opencv_time = time.perf_counter() - start

    return educational_time, opencv_time


def run_benchmark():
    results_gray = {
        combination: {"edu": [], "cv": []} for combination in COMBINATIONS_GRAY
    }
    for size in sizes:
        print(f"\nTesting grayscale, size {size}x{size}")
        img = np.random.rand(size, size).astype(np.float32)
        for combination in COMBINATIONS_GRAY:
            kernel, padding = combination
            educational_time, opencv_time = measure_gray(kernel, padding, img)
            results_gray[combination]["edu"].append(educational_time)
            results_gray[combination]["cv"].append(opencv_time)
            print(f"  Gray: {kernel}+{padding} done")

    results_rgb = {
        combination: {"edu": [], "cv": []} for combination in COMBINATIONS_RGB
    }
    for size in sizes:
        print(f"\nTesting RGB, size {size}x{size}")
        img = np.random.rand(size, size, 3).astype(np.float32)
        for combination in COMBINATIONS_RGB:
            kernel, padding = combination
            educational_time, opencv_time = measure_rgb(kernel, padding, img)
            results_rgb[combination]["edu"].append(educational_time)
            results_rgb[combination]["cv"].append(opencv_time)
            print(f"  RGB: {kernel}+{padding} done")

    for combination in COMBINATIONS_GRAY:
        results_gray[combination]["edu_ms"] = [
            time * 1000 for time in results_gray[combination]["edu"]
        ]
        results_gray[combination]["cv_ms"] = [
            time * 1000 for time in results_gray[combination]["cv"]
        ]
    for combination in COMBINATIONS_RGB:
        results_rgb[combination]["edu_ms"] = [
            time * 1000 for time in results_rgb[combination]["edu"]
        ]
        results_rgb[combination]["cv_ms"] = [
            time * 1000 for time in results_rgb[combination]["cv"]
        ]

    all_info = []
    for combination in COMBINATIONS_GRAY:
        all_info.append((combination[0], combination[1], "Grayscale"))
    for combination in COMBINATIONS_RGB:
        all_info.append((combination[0], combination[1], "RGB"))

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
        edu_values = result["edu_ms"]
        cv_values = result["cv_ms"]
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
    print("\nGrayscale")
    for combination in COMBINATIONS_GRAY:
        print(f"{combination[0]} + {combination[1]}:")
        for index, size in enumerate(sizes):
            edu = results_gray[combination]["edu_ms"][index]
            cv = results_gray[combination]["cv_ms"][index]
            print(f"{size}x{size}: Educational = {edu:.3f} ms, OpenCV = {cv:.3f} ms")
    print("\nRGB")
    for combination in COMBINATIONS_RGB:
        print(f"{combination[0]} + {combination[1]}:")
        for index, size in enumerate(sizes):
            edu = results_rgb[combination]["edu_ms"][index]
            cv = results_rgb[combination]["cv_ms"][index]
            print(f"{size}x{size}: Educational = {edu:.3f} ms, OpenCV = {cv:.3f} ms")
