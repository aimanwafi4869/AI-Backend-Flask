import torch
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import os

print("Loading YOLOv11 .pt model...")
model = YOLO("Yolo_v11_Model.pt")
model.eval()

# Dummy input (1, 3, 640, 640)
dummy_input = torch.randn(1, 3, 640, 640)

print("Tracing model with TorchScript...")
traced_model = torch.jit.trace(model.model, dummy_input)

print("Converting to TensorFlow via torch2trt or manual...")
# We'll use a simple wrapper since Ultralytics has no direct TF export

class TFWrapper(tf.keras.Model):
    def __init__(self, torch_model):
        super().__init__()
        self.torch_model = torch_model

    def call(self, inputs):
        # Convert TF tensor → PyTorch
        x = tf.transpose(inputs, [0, 3, 1, 2])  # NCHW
        x = torch.from_numpy(x.numpy())
        with torch.no_grad():
            output = self.torch_model(x)
        # Convert back to TF
        if isinstance(output, tuple):
            output = output[0]
        output = output.numpy()
        output = tf.transpose(output, [0, 2, 3, 1])  # NHWC
        return tf.convert_to_tensor(output)

print("Wrapping in Keras model...")
tf_model = TFWrapper(traced_model)

# Test inference
print("Testing inference...")
test_input = tf.random.normal([1, 640, 640, 3])
output = tf_model(test_input)
print(f"Output shape: {output.shape}")

# Save as SavedModel
print("Saving as TensorFlow SavedModel...")
tf.saved_model.save(tf_model, "saved_model_tf")

print("SavedModel ready in 'saved_model_tf/'!")