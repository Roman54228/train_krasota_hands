"""
Export ONNX models to TensorRT engine.
Supports FP16 and INT8 quantization with calibration.
"""

import argparse
import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from glob import glob
from tqdm import tqdm


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator for TensorRT."""
    
    def __init__(self, calibration_images, batch_size, img_size, cache_file):
        super().__init__()
        self.calibration_images = calibration_images
        self.batch_size = batch_size
        self.img_size = img_size
        self.current_index = 0
        self.cache_file = cache_file
        self.device_input = None
        self.data_shape = (3, img_size, img_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.calibration_images):
            return None

        # Lazy allocation of device memory
        if self.device_input is None:
            self.device_input = cuda.mem_alloc(
                trt.volume(self.data_shape) * self.batch_size * np.float32().nbytes
            )

        # Collect batch
        batch = []
        for i in range(self.batch_size):
            if self.current_index >= len(self.calibration_images):
                break
            batch.append(self.calibration_images[self.current_index])
            self.current_index += 1

        if not batch:
            return None

        # Stack and copy to device
        batch = np.ascontiguousarray(np.stack(batch)).astype(np.float32)
        cuda.memcpy_htod(self.device_input, batch)

        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def load_calibration_images(image_paths, img_size, grayscale=False, max_images=500):
    """Load and preprocess calibration images.
    
    Args:
        image_paths: List of image paths or directory
        img_size: Target image size
        grayscale: Convert to grayscale
        max_images: Maximum number of images to use
        
    Returns:
        List of preprocessed image arrays [N, 3, H, W]
    """
    if isinstance(image_paths, str):
        # Directory path
        image_paths = glob(os.path.join(image_paths, "*.jpg")) + \
                     glob(os.path.join(image_paths, "*.png")) + \
                     glob(os.path.join(image_paths, "*.jpeg"))
    
    image_paths = image_paths[:max_images]
    
    print(f"Loading {len(image_paths)} calibration images...")
    
    images = []
    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img = cv2.resize(img, (img_size, img_size))
        
        if grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.stack([gray] * 3, axis=-1)
        
        # Convert to RGB and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # To CHW format
        img = img.transpose(2, 0, 1)
        images.append(img)
    
    print(f"Loaded {len(images)} images")
    return images


def build_engine(args):
    """Build TensorRT engine from ONNX."""
    
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    
    # Parse ONNX
    print(f"Parsing ONNX model: {args.onnx_path}")
    parser = trt.OnnxParser(network, logger)
    
    with open(args.onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print("ONNX parsed successfully")
    
    # Set memory pool
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_size << 30)
    
    # FP16
    if args.fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled")
        else:
            print("Warning: FP16 not supported on this platform")
    
    # INT8
    if args.int8:
        if not builder.platform_has_fast_int8:
            print("Warning: INT8 not supported on this platform")
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            print("INT8 mode enabled")
            
            # Load calibration data
            if not args.calib_images:
                raise ValueError("--calib_images required for INT8 mode")
            
            calib_images = load_calibration_images(
                args.calib_images,
                args.img_size,
                grayscale=args.grayscale,
                max_images=args.max_calib_images
            )
            
            calibrator = Int8Calibrator(
                calib_images,
                args.calib_batch_size,
                args.img_size,
                args.calib_cache
            )
            config.int8_calibrator = calibrator
    
    # Optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(args.min_batch, 3, args.img_size, args.img_size),
        opt=(args.opt_batch, 3, args.img_size, args.img_size),
        max=(args.max_batch, 3, args.img_size, args.img_size)
    )
    config.add_optimization_profile(profile)
    
    # Build engine
    print("Building TensorRT engine... (this may take several minutes)")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except Exception as e:
        print(f"ERROR: Failed to build engine: {e}")
        return None
    
    if serialized_engine is None:
        print("ERROR: Engine build returned None")
        return None
    
    # Save engine
    print(f"Saving engine to: {args.output_path}")
    with open(args.output_path, "wb") as f:
        f.write(serialized_engine)
    
    print("✅ TensorRT engine built successfully!")
    print(f"   Output: {args.output_path}")
    print(f"   FP16: {args.fp16}")
    print(f"   INT8: {args.int8}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export ONNX to TensorRT engine')
    
    # Input/Output
    parser.add_argument('--onnx_path', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output TensorRT engine path')
    
    # Model settings
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--min_batch', type=int, default=1,
                        help='Minimum batch size')
    parser.add_argument('--opt_batch', type=int, default=2,
                        help='Optimal batch size')
    parser.add_argument('--max_batch', type=int, default=4,
                        help='Maximum batch size')
    
    # Precision
    parser.add_argument('--fp16', action='store_true',
                        help='Enable FP16 precision')
    parser.add_argument('--int8', action='store_true',
                        help='Enable INT8 precision (requires calibration)')
    
    # INT8 Calibration
    parser.add_argument('--calib_images', type=str, default=None,
                        help='Path to calibration images directory (for INT8)')
    parser.add_argument('--calib_batch_size', type=int, default=1,
                        help='Calibration batch size')
    parser.add_argument('--max_calib_images', type=int, default=500,
                        help='Maximum number of calibration images')
    parser.add_argument('--calib_cache', type=str, default='calibration.cache',
                        help='Calibration cache file')
    parser.add_argument('--grayscale', action='store_true',
                        help='Convert images to grayscale')
    
    # TensorRT settings
    parser.add_argument('--workspace_size', type=int, default=1,
                        help='Workspace size in GB')
    
    args = parser.parse_args()
    build_engine(args)

