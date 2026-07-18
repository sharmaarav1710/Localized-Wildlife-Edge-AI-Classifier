import os
import time
import numpy as np
import onnxruntime as ort

# ImageNet normalization statistics
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def generate_dummy_input(batch_size=1, channels=3, height=224, width=224):
    """Generates a standardized dummy tensor matching edge device runtime shapes"""
    return np.random.randn(batch_size, channels, height, width).astype(np.float32)

def benchmark_model(model_path, num_runs=100, warmup_runs=10):
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}. Skipping.")
        return None

    # Force ONNX to utilize a single CPU core to mimic ultra-low resource edge hardware
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(model_path, sess_options=opts)
    input_name = session.get_inputs()[0].name
    dummy_data = generate_dummy_input()

    # Warmup runs to allow the CPU cache and JIT to stabilize
    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: dummy_data})

    # Actual latency evaluation loop
    latencies = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = session.run(None, {input_name: dummy_data})
        latencies.append((time.perf_counter() - start_time) * 1000.0) # convert to ms

    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    p95_latency = np.percentile(latencies, 95)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    return {
        "file_size": file_size_mb,
        "mean": mean_latency,
        "std": std_latency,
        "p95": p95_latency
    }

def main():
    print("⏱️ Initiating Edge AI Latency & Footprint Benchmarker...")
    print("   ↳ System constraints configured: CPU Cores locked to $1$ thread (mimicking edge limits).")
    
    fp32_path = "models/wildlife_model.onnx"
    int8_path = "models/wildlife_model_quantized.onnx"

    fp32_results = benchmark_model(fp32_path)
    int8_results = benchmark_model(int8_path)

    print("\n📊 BENCHMARK COMPARISON REPORT:")
    print("=" * 65)
    print(f"{'Metric':<25} | {'Base FP32 Model':<16} | {'Quantized INT8 Model':<16}")
    print("=" * 65)
    
    if fp32_results:
        print(f"{'Model File Size (MB)':<25} | {fp32_results['file_size']:>12.2f} MB | {int8_results['file_size']:>12.2f} MB")
        print(f"{'Avg Latency (ms)':<25} | {fp32_results['mean']:>12.2f} ms | {int8_results['mean']:>12.2f} ms")
        print(f"{'p95 Latency (ms)':<25} | {fp32_results['p95']:>12.2f} ms | {int8_results['p95']:>12.2f} ms")
        print(f"{'Latency StdDev (ms)':<25} | {fp32_results['std']:>12.2f} ms | {int8_results['std']:>12.2f} ms")
    print("=" * 65)

    if fp32_results and int8_results:
        speed_diff = ((fp32_results["mean"] - int8_results["mean"]) / fp32_results["mean"]) * 100.0
        if speed_diff > 0:
            print(f"🚀 INT8 model is {speed_diff:.1f}% faster than the FP32 base model.")
        else:
            print(f"ℹ️ FP32 base model is {abs(speed_diff):.1f}% faster on this host architecture.")
            print("   ↳ Reason: Dynamic quantization has CPU scaling overhead on modern multi-core devices.")
            print("     For lightweight networks (like MobileNetV3-Small), standard FP32 remains the edge sweet spot!")

if __name__ == "__main__":
    main()