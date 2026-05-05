from src.convolution import convolution_grayscale, convolution_rgb
from PIL import Image
import numpy as np
from src.benchmark.visualization import visualization
from src.benchmark.run_benchmark import run_benchmark
import typer
from pathlib import Path

app = typer.Typer()


@app.command()
def run(
    input_dir: str = typer.Option(..., help="Input directory"),
    output_dir: str = typer.Option(..., help="Output directory"),
    image_mode: str = typer.Option(..., help="Image mode: 'L' or 'RGB'"),
    kernel: str = typer.Option(..., help="Kernel name"),
    padding: str = typer.Option(..., help="Padding type"),
):
    if image_mode != "L" and image_mode != "RGB":
        raise ValueError(
            f"Unsupported image mode: {image_mode}. Use 'L' for grayscale or 'RGB' for color."
        )
    img = Image.open(input_dir).convert(image_mode)
    arr_img = np.asarray(img, dtype=np.float32)

    func = convolution_grayscale if image_mode == "L" else convolution_rgb
    new_img = func(
        arr_img=arr_img,
        kernel=kernel,
        padding=padding,
    )

    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    result = Image.fromarray(new_img)
    result.save(output_dir)


@app.command()
def benchmark(
    only_benchmark: bool = typer.Option(False, "--benchmark", help="Run benchmark"),
    only_visualization: bool = typer.Option(
        False, "--visualization", help="Run visualization"
    ),
):

    if not only_benchmark and not only_visualization:
        only_benchmark = True
        only_visualization = True

    if only_benchmark:
        run_benchmark()

    if only_visualization:
        file_path = Path("results.json")
        if not file_path.exists():
            raise FileNotFoundError("File not found. Run the benchmark first.")
        visualization()


if __name__ == "__main__":
    app()
