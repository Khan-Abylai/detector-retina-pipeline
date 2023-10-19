# worked with tensorflow-gpu 2.3.0 version
import os
import tensorflow as tf
import numpy as np

def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 3, 480, 480)
      yield [data.astype(np.float32)]

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

converter = tf.lite.TFLiteConverter.from_saved_model(f"weights/PlateDetector_mobilenet.pb")

# float16
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

# int8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8

# way 2
#loaded = tf.saved_model.load("weights/PlateDetector_mobilenet.pb")
#concrete_func = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
#concrete_func.inputs[0].set_shape([1, 3, 480, 480])
#converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
#converter.target_spec.supported_ops = [
   #tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
   #tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
   #]
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open('weights/PlateDetector-mobilenet_int8.tflite', 'wb') as f:
  f.write(tflite_model)

print("Converted succesfully!")
