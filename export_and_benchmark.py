# export_and_benchmark.py

import os
import time
import torch
import numpy as np
from model import ArterySegModel, device
from config import SHAPE, MODEL_NAME
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Load trained model and weights
weight_path = f"logs/0_finalized_seg_{MODEL_NAME}.pth"
model = ArterySegModel(in_ch=1, sobel_ch=64)
model.load_state_dict(torch.load(weight_path, map_location=device))
model.to(device)
model.eval()

# Create dummy input for export and benchmarking
dummy_input = torch.randn(1, 1, SHAPE[0], SHAPE[1]).to(device)

# Export model to ONNX
onnx_file = "model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_file,
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

# Build TensorRT engine from ONNX
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_file, "rb") as model_file:
    if not parser.parse(model_file.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
engine = builder.build_engine(network, config)

# Allocate buffers for TensorRT engine execution
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        shape = engine.get_binding_shape(binding)
        size = trt.volume(shape) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})
    return inputs, outputs, bindings, stream

inputs, outputs, bindings, stream = allocate_buffers(engine)
context = engine.create_execution_context()

# Function to run inference with TensorRT engine
def run_trt_inference(context, inputs, outputs, bindings, stream, input_data):
    np.copyto(inputs[0]["host"], input_data.ravel())
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
    stream.synchronize()
    return outputs[0]["host"]

# Benchmark Inference Times

n_runs = 50

# PyTorch Inference Timing
with torch.no_grad():
    for _ in range(10):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(n_runs):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / n_runs

# TensorRT Inference Timing
dummy_input_np = dummy_input.cpu().numpy()
for _ in range(10):
    _ = run_trt_inference(context, inputs, outputs, bindings, stream, dummy_input_np)
start_time = time.time()
for _ in range(n_runs):
    _ = run_trt_inference(context, inputs, outputs, bindings, stream, dummy_input_np)
trt_time = (time.time() - start_time) / n_runs

print("Average Inference Time per Run:")
print(f"PyTorch: {pytorch_time:.6f} seconds")
print(f"TensorRT: {trt_time:.6f} seconds")

