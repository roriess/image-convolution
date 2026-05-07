import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

COMBINATIONS_GRAY = [
    ("embossing_3x3", "zero_padding"),
    ("blur_5x5", "mirror_padding"),
]
COMBINATIONS_RGB = [
    ("sharpen_3x3", "replicate_padding"),
    ("gaussian_blur_5x5", "zero_padding"),
]


def visualization():
    with open("results.json", "r") as f:
        data = json.load(f)

    results = {}

    for benchmark in data["benchmarks"]:
        name = benchmark["name"]

        bracket_start = name.find("[")
        bracket_end = name.find("]")

        if bracket_start == -1 or bracket_end == -1:
            continue

        test_name = name[:bracket_start]

        parameters = name[bracket_start + 1 : bracket_end]

        image_type = "grayscale" if "grayscale" in test_name else "rgb"
        implementation = "edu" if "edu" in test_name else "cv"

        parts = parameters.split("-")
        if len(parts) < 3:
            continue

        size = int(parts[0].split("x")[0])

        kernel_plus_padding = "-".join(parts[1:])
        last_hyphen = kernel_plus_padding.rfind("-")
        kernel = kernel_plus_padding[:last_hyphen]
        padding = kernel_plus_padding[last_hyphen + 1 :]

        mean = benchmark["stats"]["mean"] * 1000
        std = benchmark["stats"]["stddev"] * 1000

        key = (image_type, kernel, padding, size, implementation)
        results[key] = (mean, std)

    diagram_data = defaultdict(
        lambda: {
            "sizes": [],
            "edu_mean": [],
            "edu_std": [],
            "cv_mean": [],
            "cv_std": [],
        }
    )

    for (image_type, kernel, padding, size, implementation), (
        mean,
        std,
    ) in results.items():
        key = (image_type, kernel, padding)
        if implementation == "edu":
            diagram_data[key]["edu_mean"].append((size, mean))
            diagram_data[key]["edu_std"].append((size, std))
        else:
            diagram_data[key]["cv_mean"].append((size, mean))
            diagram_data[key]["cv_std"].append((size, std))

    for key in diagram_data:
        edu = sorted(diagram_data[key]["edu_mean"], key=lambda x: x[0])
        diagram_data[key]["sizes"] = [size for size, _ in edu]
        diagram_data[key]["edu_mean"] = [mean for _, mean in edu]
        edu_std = sorted(diagram_data[key]["edu_std"], key=lambda x: x[0])
        diagram_data[key]["edu_std"] = [size for _, size in edu_std]

        cv = sorted(diagram_data[key]["cv_mean"], key=lambda x: x[0])
        diagram_data[key]["cv_mean"] = [mean for _, mean in cv]
        cv_std = sorted(diagram_data[key]["cv_std"], key=lambda x: x[0])
        diagram_data[key]["cv_std"] = [size for _, size in cv_std]

    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    order = []
    for kernel, padding in COMBINATIONS_GRAY:
        order.append(("grayscale", kernel, padding))
    for kernel, padding in COMBINATIONS_RGB:
        order.append(("rgb", kernel, padding))

    for index, key in enumerate(order):
        axis = axes[index]
        data = diagram_data[key]
        image_type, kernel, padding = key
        column_position = np.arange(len(data["sizes"]))
        width = 0.35

        bars_1 = axis.bar(
            column_position - width / 2,
            data["edu_mean"],
            width,
            yerr=data["edu_std"],
            label="Educational",
            color="#1f77b4",
            capsize=5,
            ecolor="black",
            linewidth=1.5,
        )
        bars_2 = axis.bar(
            column_position + width / 2,
            data["cv_mean"],
            width,
            yerr=data["cv_std"],
            label="OpenCV",
            color="#ff7f0e",
            capsize=5,
            ecolor="black",
            linewidth=1.5,
        )

        axis.set_xticks(column_position)
        axis.set_xticklabels(data["sizes"])
        axis.set_yscale("log")
        axis.set_xlabel("Image size")
        axis.set_ylabel("Time (ms)")
        axis.set_title(f"{kernel}\n{padding}\n({image_type.capitalize()})")
        axis.legend()
        axis.grid(axis="y", linestyle="--", alpha=0.7)

        for bar, mean_value, std_value in zip(
            bars_1, data["edu_mean"], data["edu_std"]
        ):
            height = bar.get_height()
            axis.annotate(
                f"{mean_value:.4f}±{std_value:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

        for bar, mean_value, std_value in zip(bars_2, data["cv_mean"], data["cv_std"]):
            height = bar.get_height()
            axis.annotate(
                f"{mean_value:.4f}±{std_value:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    plt.suptitle(
        "Performance comparison: Educational vs OpenCV (mean ± std)", fontsize=14
    )
    plt.tight_layout()
    plt.savefig("src/benchmark/benchmark_graphs.png", dpi=150)
    plt.show()
