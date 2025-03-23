import tensorflow as tf
from tflite_support import metadata as _metadata
from tensorflow.lite.tools.flatbuffer_utils import *
from tflite_support import flatbuffers


# Load the original Ultralytics model
model = tf.saved_model.load("models/best_saved_model")

# Convert to TFLite with `float16` weights but `float32` inputs
converter = tf.lite.TFLiteConverter.from_saved_model("models/best_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # Keep weights float16
converter.inference_input_type = tf.float32  # Ensure model accepts float32 input

tflite_model = converter.convert()

# Save new model
with open("new_model.tflite", "wb") as f:
    f.write(tflite_model)

reader = _metadata.MetadataDisplayer.with_model_file("models/best_float16.tflite")
print(reader.get_metadata_json())

writer = _metadata.MetadataPopulator.with_model_file("new_model.tflite")
writer.load_metadata_and_associated_files("models/best_float16.tflite")
