from .modelinterface import ModelInterface
from .yolo import YoloDetector
from .new import FastDetector
import sys
import cv2
from numpy import float32, expand_dims
import time

# Attempt to load the pi camera, otherwise fall back to normal webcams.
# TODO: refactor this into a class
try:
    from picamera2 import Picamera2
    from libcamera import Transform

    camera: Picamera2

    def prepare_camera():
        camera = Picamera2()
        # BGR888 is actually RGB, yes, I know, it's confusing.
        # RGB is what the model can take as an input.
        configuration = camera.create_preview_configuration(
            main={"size": (640, 640), "format": "BGR888"}, transform=Transform(hflip=1)
        )

        camera.configure(configuration)

        camera.start()

    def capture():
        value = camera.capture_array()
        return value, value

    def stop_camera():
        camera.close()

    print("INFO: Using PiCamera.")

except ImportError:
    camera: cv2.VideoCapture

    def prepare_camera():
        global camera

        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        # Test it once to see if it works.
        capture()

    def capture():
        global camera

        success, frame = camera.read()

        if not success:
            sys.exit(
                "ERROR: Unable to read from webcam. Please verify your webcam settings."
            )

        rame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        float_array = rgb.astype(float32)
        float_array /= 255.0
        float_array = expand_dims(float_array, axis=0)

        return float_array, frame

    def stop_camera():
        global camera

        camera.release()

    print("INFO: Using OpenCV camera.")

    pass


def main():
    # Load the model
    model: ModelInterface = YoloDetector("models/best_float32.tflite")
    # model: ModelInterface = FastDetector("models/best_float32.tflite")

    # Variables to calculate FPS
    counter, fps = 0, 0.0
    start_time = time.time()
    fps_avg_frame_count = 10

    prepare_camera()

    while True:
        array, image = capture()

        results = model.detect(array)

        for result in results:
            cv2.rectangle(
                img=image,
                pt1=result.point1,
                pt2=result.point2,
                color=(10, 255, 0),
                thickness=2,
            )

            label = f"{result.name}: {int(result.confidence * 100)}%"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            label_ymin = max(result.point1[1], label_size[1] + 10)
            cv2.rectangle(
                image,
                (int(result.point1[1]), label_ymin - label_size[1] - 10),
                (int(result.point1[1]) + label_size[0], label_ymin + 2),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                image,
                label,
                (int(result.point1[1]), label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = "FPS = {:.1f}".format(fps)
        text_location = (24, 24)
        cv2.putText(
            image,
            fps_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Camera", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop_camera()


if __name__ == "__main__":
    main()
