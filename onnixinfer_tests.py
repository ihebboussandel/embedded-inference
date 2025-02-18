import onnxruntime as ort
import numpy as np
import time
import resource  # To track memory usage on Linux
import gc  # To ensure garbage collection

# Path to your ONNX model
model_path = "Neoprene.onnx"

# Specify the providers for ONNX Runtime
providers = ['CPUExecutionProvider', 'XnnpackExecutionProvider']  # Add or remove providers as needed

# Create an ONNX Runtime session with specified providers
session = ort.InferenceSession(model_path, providers=providers)

# Get the input name for the model
input_name = session.get_inputs()[0].name  # 'X' is the name of the input

# Function to measure memory usage before and after inference
def measure_memory():
    # Get memory usage before inference (in KB)
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024  # Convert to MB

# Function to run inference and measure time and memory usage
def run_inference(input_data):
    # Ensure garbage collection to clean up any unused memory
    gc.collect()

    # Measure the start time and memory usage before inference
    start_time = time.time()
    memory_before = measure_memory()

    # Run inference
    output = session.run(None, {input_name: input_data})

    # Measure the end time and memory usage after inference
    end_time = time.time()
    memory_after = measure_memory()

    # Calculate time and memory usage
    inference_time = end_time - start_time
    memory_used = memory_after - memory_before

    return output, inference_time, memory_used

# Batch size testing
batch_sizes = [1, 5, 10, 50, 100]  # Different batch sizes to test
num_iterations = 5  # Number of iterations to average memory usage and inference time
for batch_size in batch_sizes:
    total_inference_time = 0
    total_memory_used = 0

    # Run multiple iterations to average the results
    for _ in range(num_iterations):
        # Create a batch of random input data of size `batch_size`
        input_data = np.random.rand(batch_size, 3).astype(np.float64)  # Adjust to match model input shape

        # Run inference and measure time and memory
        _, inference_time, memory_used = run_inference(input_data)

        # Accumulate time and memory usage
        total_inference_time += inference_time
        total_memory_used += memory_used

    # Calculate average results
    avg_inference_time = total_inference_time / num_iterations
    avg_memory_used = total_memory_used / num_iterations

    # Calculate throughput
    throughput = batch_size / avg_inference_time

    # Print results for each batch size
    print(f"Batch size: {batch_size}")
    print(f"  Avg inference time: {avg_inference_time:.6f} seconds")
    print(f"  Avg memory used: {avg_memory_used:.9f} MB")
    print(f"  Throughput: {throughput:.2f} inferences per second")
    print("=" * 50)
