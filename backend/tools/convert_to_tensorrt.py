#!/usr/bin/env python3
"""Convert ONNX models to TensorRT engines.

This tool converts YOLO ONNX models to optimized TensorRT engines for
faster inference on NVIDIA GPUs.

Usage:
    source /opt/venv/bin/activate
    cd /workspace/ronin-nvr/backend

    # Convert with FP16 precision (recommended)
    python tools/convert_to_tensorrt.py --model /opt3/ronin/ml_models/yolov8n.onnx

    # Convert with specific precision
    python tools/convert_to_tensorrt.py --model /opt3/ronin/ml_models/yolov8s.onnx --precision fp32

    # Convert with custom batch size
    python tools/convert_to_tensorrt.py --model model.onnx --batch-size 16

Output:
    Creates .engine file in same directory as input ONNX model.
    E.g., yolov8n.onnx -> yolov8n.engine
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("ERROR: TensorRT not installed")
    sys.exit(1)


def create_engine(
    onnx_path: Path,
    precision: str = "fp16",
    max_batch_size: int = 8,
    workspace_size_gb: float = 2.0,
    verbose: bool = False,
) -> bytes:
    """Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model
        precision: Inference precision ("fp32", "fp16", "int8")
        max_batch_size: Maximum batch size for dynamic batching
        workspace_size_gb: GPU workspace memory in GB
        verbose: Enable verbose logging

    Returns:
        Serialized TensorRT engine bytes
    """
    # Set up logging
    log_level = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
    trt_logger = trt.Logger(log_level)

    # Create builder
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    # Parse ONNX model
    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        onnx_data = f.read()

    if not parser.parse(onnx_data):
        print("ERROR: Failed to parse ONNX model")
        for i in range(parser.num_errors):
            print(f"  Error {i}: {parser.get_error(i)}")
        return None

    # Show network info
    print(f"\nNetwork info:")
    print(f"  Inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    {inp.name}: {inp.shape} ({inp.dtype})")

    print(f"  Outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    {out.name}: {out.shape} ({out.dtype})")

    print(f"  Layers: {network.num_layers}")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_size_gb * (1 << 30))
    )

    # Set precision
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print(f"\nUsing FP16 precision")
        else:
            print("WARNING: FP16 not supported on this GPU, using FP32")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print(f"\nUsing INT8 precision")
            print("WARNING: INT8 requires calibration data for best accuracy")
        else:
            print("WARNING: INT8 not supported on this GPU, using FP32")
    else:
        print(f"\nUsing FP32 precision")

    # Set up optimization profile for dynamic batch
    profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)

        # Handle dynamic dimensions
        min_shape = shape.copy()
        opt_shape = shape.copy()
        max_shape = shape.copy()

        # Set batch dimension
        if shape[0] == -1:
            min_shape[0] = 1
            opt_shape[0] = max_batch_size // 2 or 1
            max_shape[0] = max_batch_size

        profile.set_shape(inp.name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
        print(f"\nOptimization profile for {inp.name}:")
        print(f"  Min: {min_shape}")
        print(f"  Opt: {opt_shape}")
        print(f"  Max: {max_shape}")

    config.add_optimization_profile(profile)

    # Build engine
    print(f"\nBuilding TensorRT engine (this may take several minutes)...")
    t0 = time.time()

    serialized_engine = builder.build_serialized_network(network, config)

    build_time = time.time() - t0
    print(f"Build completed in {build_time:.1f} seconds")

    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return None

    return serialized_engine


def verify_engine(engine_path: Path) -> bool:
    """Verify a TensorRT engine can be loaded and run.

    Args:
        engine_path: Path to .engine file

    Returns:
        True if engine is valid
    """
    print(f"\nVerifying engine: {engine_path}")

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)

    with open(engine_path, "rb") as f:
        engine_data = f.read()

    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        print("ERROR: Failed to deserialize engine")
        return False

    print(f"  Engine loaded successfully")
    print(f"  IO tensors: {engine.num_io_tensors}")

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"    {name}: {shape} ({dtype}) - {mode}")

    # Create execution context
    context = engine.create_execution_context()
    if context is None:
        print("ERROR: Failed to create execution context")
        return False

    print("  Execution context created")

    # Run dummy inference
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        # Allocate buffers
        stream = cuda.Stream()

        # Get input shape
        input_name = engine.get_tensor_name(0)
        input_shape = list(engine.get_tensor_shape(input_name))
        if input_shape[0] == -1:
            input_shape[0] = 1
        input_shape = tuple(input_shape)

        # Create dummy input
        input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
        dummy_input = np.random.randn(*input_shape).astype(input_dtype)

        # Allocate device memory
        d_input = cuda.mem_alloc(dummy_input.nbytes)
        cuda.memcpy_htod_async(d_input, dummy_input, stream)

        # Set input shape
        context.set_input_shape(input_name, input_shape)
        context.set_tensor_address(input_name, int(d_input))

        # Allocate output
        output_name = engine.get_tensor_name(1)
        output_shape = context.get_tensor_shape(output_name)
        output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
        output_size = int(np.prod(output_shape))
        h_output = cuda.pagelocked_empty(output_size, output_dtype)
        d_output = cuda.mem_alloc(h_output.nbytes)
        context.set_tensor_address(output_name, int(d_output))

        # Run inference
        context.execute_async_v3(stream.handle)

        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        print(f"  Dummy inference successful")
        print(f"  Output shape: {output_shape}")

        return True

    except ImportError:
        print("  PyCUDA not available, skipping inference test")
        return True
    except Exception as e:
        print(f"ERROR: Inference test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX models to TensorRT engines"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output engine path (default: same as input with .engine extension)"
    )
    parser.add_argument(
        "--precision", "-p",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Inference precision (default: fp16)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Maximum batch size (default: 8)"
    )
    parser.add_argument(
        "--workspace", "-w",
        type=float,
        default=2.0,
        help="GPU workspace size in GB (default: 2.0)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify engine after building"
    )

    args = parser.parse_args()

    # Validate input
    onnx_path = Path(args.model)
    if not onnx_path.exists():
        print(f"ERROR: Model file not found: {onnx_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        engine_path = Path(args.output)
    else:
        engine_path = onnx_path.with_suffix(".engine")

    print("=" * 60)
    print("TensorRT Model Conversion")
    print("=" * 60)
    print(f"Input:      {onnx_path}")
    print(f"Output:     {engine_path}")
    print(f"Precision:  {args.precision}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workspace:  {args.workspace} GB")
    print("=" * 60)

    # Build engine
    engine_data = create_engine(
        onnx_path=onnx_path,
        precision=args.precision,
        max_batch_size=args.batch_size,
        workspace_size_gb=args.workspace,
        verbose=args.verbose,
    )

    if engine_data is None:
        print("\nERROR: Engine build failed")
        sys.exit(1)

    # Save engine
    print(f"\nSaving engine to: {engine_path}")

    # Handle both bytes and IHostMemory types
    if hasattr(engine_data, "tobytes"):
        engine_bytes = engine_data.tobytes()
    elif isinstance(engine_data, bytes):
        engine_bytes = engine_data
    else:
        # TensorRT 10.x returns IHostMemory
        engine_bytes = bytes(engine_data)

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    engine_size_mb = len(engine_bytes) / (1024 * 1024)
    print(f"Engine size: {engine_size_mb:.1f} MB")

    # Verify if requested
    if args.verify:
        if not verify_engine(engine_path):
            print("\nWARNING: Engine verification failed")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
