import sys
import subprocess
from pathlib import Path


def run_benchmark():
    file = Path("src/benchmark/benchmark.py")
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")

    command = [
        sys.executable,
        "-m",
        "pytest",
        str(file),
        "--benchmark-json",
        "results.json",  # сохранение результатов в файл results.json
        "--benchmark-only",  # игнорирование обычных тестов
        "--benchmark-disable-gc",  # отключение сборщика мусора
        "--benchmark-min-rounds=3",  # минимальное число прогонов каждого теста
        "--benchmark-warmup=on",  # выполнение прогревочного прогона
    ]
    subprocess.run(command, check=True)
