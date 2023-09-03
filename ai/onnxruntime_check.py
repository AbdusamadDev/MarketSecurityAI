import onnx
import os


# Load the ONNX model
onnx_model = onnx.load(os.path.join("resources", "single_relu.onnx"))
print(onnx_model)