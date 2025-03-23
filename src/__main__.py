from .cameras import CameraInterface
from .models import ModelInterface, FastDetector, YoloDetector, PlainYoloDetector
from .detect import run_object_detection
from .benchmark import run_benchmarks

def main():
    camera: CameraInterface
    model: ModelInterface

    try:
        from .cameras.picamera import PiCamera

        camera = PiCamera()
    except ImportError:
        from .cameras.cvcamera import Cv2Camera

        camera = Cv2Camera()

    # model = PlainYoloDetector("models/best_float32.tflite")
    # model = YoloDetector("models/best_float32.tflite")
    # model = FastDetector("models/yolov5nu_float16.tflite")
    # model = PlainYoloDetector("models/yolov5nu_float16.tflite")
    # model = FastDetector("models/best_saved_model/best_float16.tflite")

    # run_object_detection(camera, model)
    run_benchmarks(camera)

    camera.stop_camera()

if __name__ == "__main__":
    main()
