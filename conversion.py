from ultralytics import YOLO

model = YOLO("models/best.pt")
model.half()
model.export(format="tflite", half=True)

# model.export(format="saved_model")
# model.export(format="tflite", int8=True)
