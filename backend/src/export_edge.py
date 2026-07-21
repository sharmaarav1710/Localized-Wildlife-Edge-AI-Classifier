import os
import torch
from train import EdgeClassifier

try:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process
    ONNX_QUANT_AVAILABLE = True
except ImportError:
    ONNX_QUANT_AVAILABLE = False

def main():

    os.makedirs("models", exist_ok=True)
    pytorch_model_path = "models/wildlife_model.pth"
    onnx_fp32_path = "models/wildlife_model.onnx"
    onnx_prep_path = "models/wildlife_model_prep.onnx"
    onnx_int8_path = "models/wildlife_model_quantized.onnx"

    if not os.path.exists(pytorch_model_path):
        raise FileNotFoundError(f"Could not find trained PyTorch model weights at {pytorch_model_path}!")

    print(" Loading trained PyTorch weights...")
    model = EdgeClassifier(num_classes=2)
    model.load_state_dict(torch.load(pytorch_model_path, map_location=torch.device("cpu")))
    model.eval()


    dummy_input = torch.randn(1, 3, 224, 224)

    print(" Exporting to standard ONNX representation (Static Batch Size = 1, Ops 18)...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_fp32_path,
        export_params=True,
        opset_version=18,  
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )
    print(f"FP32 ONNX model saved to {onnx_fp32_path}")

    if ONNX_QUANT_AVAILABLE:
        print("⚡ Pre-processing model with ONNX shape inference...")
        try:

            quant_pre_process(
                input_model_path=onnx_fp32_path,
                output_model_path=onnx_prep_path,
                skip_symbolic_shape=False
            )
            quant_target_input = onnx_prep_path
            print("✓ Graph pre-processing completed.")
        except Exception as e:
            print(f"Pre-processing optimization skipped due to: {e}. Falling back to direct quantization.")
            quant_target_input = onnx_fp32_path

        print("Performing INT8 dynamic CPU quantization...")
        try:
            quantize_dynamic(
                model_input=quant_target_input,
                model_output=onnx_int8_path,
                weight_type=QuantType.QUInt8
            )
            
            orig_size = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
            quant_size = os.path.getsize(onnx_int8_path) / (1024 * 1024)
            
            print("\n Quantization Success!")
            print(f"   ↳ Original ONNX Size: {orig_size:.2f} MB")
            print(f"   ↳ Quantized ONNX Size: {quant_size:.2f} MB")
            print(f"   ↳ Compression Ratio:  {(1 - quant_size/orig_size)*100:.1f}% space saved!")
            print(f"   ↳ Optimized Edge model saved to {onnx_int8_path}")
            

            if os.path.exists(onnx_prep_path):
                os.remove(onnx_prep_path)

        except Exception as e:
            print(f"❌ Failed to run dynamic quantization: {e}")
    else:
        print("\nWarning: onnxruntime is required for the optional quantization step.")
        print("You can run dynamic quantization by installing: pip install onnxruntime")

if __name__ == "__main__":
    main()