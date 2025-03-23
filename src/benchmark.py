import matplotlib.pyplot as plt
import numpy as np
from .models import ModelInterface, YoloDetector, PlainYoloDetector, FastDetector
from .cameras import CameraInterface
from .detect import run_object_detection
from time import perf_counter_ns
from tqdm import tqdm


def single_benchmark(
    camera: CameraInterface, model: ModelInterface, frames=100, desc="Benchmark"
):

    # Log FPS every `log_frame`` frames
    log_frame = max(frames // 10, 10)

    start_time = perf_counter_ns()

    measurement_count = frames // log_frame

    # One additional for warmups
    measurements: list[float] = [0.0] * (measurement_count + 1)

    for frame in tqdm(range(0, frames + log_frame), unit="frame", desc=desc):
        run_object_detection(camera, model)

        if frame % log_frame == 0:
            end_time = perf_counter_ns()
            measurements[(int)(frame // log_frame)] = log_frame / (
                end_time - start_time
            )
            start_time = perf_counter_ns()

    # Remove the warmup measurement
    return np.multiply(measurements[1:], 1e9)


def run_benchmarks(camera: CameraInterface, frames=100):
    log_frame = max(frames // 10, 10)

    models = {
        "Plain Yolo f32": PlainYoloDetector("models/best_float32.tflite"),
        "Plain Yolo v5 f16": PlainYoloDetector("models/yolov5nu_float16.tflite"),
        "Custom f16": FastDetector("models/best_saved_model/best_float16.tflite"),
        "Custom f32": FastDetector("models/best_float32.tflite"),
        "Yolo f32": YoloDetector("models/best_float32.tflite"),
        "Yolo f16": YoloDetector(
            "models/best_saved_model/best_float16.tflite", half=True
        ),
    }

    fig, ax = plt.subplots()
    for name, model in models.items():
        try:
            x = np.linspace(log_frame, frames, log_frame)
            y = single_benchmark(camera, model, frames, name)

            average = np.average(y)

            (line,) = ax.plot(x, y)
            line.set_label(f"{name} - {round(average, 2)} fps")
        except:
            continue

    plt.xlabel("Frame")
    plt.ylabel("Average FPS")
    plt.legend(loc="best")

    plt.show()
