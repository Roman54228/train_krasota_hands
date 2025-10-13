"""
Export PyTorch models to ONNX format.
"""

import argparse
import torch
import torch.onnx
import timm


def export_to_onnx(args):
    """Export model to ONNX."""
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    if args.model_type == 'blazehand':
        from MediaPipePyTorch.blazehand_landmark import BlazeHandLandmark
        model = BlazeHandLandmark()
    elif args.model_type == 'timm':
        model = timm.create_model(
            args.model_name,
            pretrained=False,
            num_classes=args.num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load weights
    if args.weights:
        print(f"Loading weights from: {args.weights}")
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict, strict=args.strict_load)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Create dummy input
    dummy_input = torch.randn(
        args.batch_size, 3, args.img_size, args.img_size,
        device=device
    )
    
    # Export to ONNX
    print(f"Exporting to ONNX: {args.output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        args.output_path,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        } if args.dynamic_batch else None
    )
    
    print(f"Model exported successfully to: {args.output_path}")
    
    # Verify ONNX model
    if args.verify:
        import onnx
        print("Verifying ONNX model...")
        onnx_model = onnx.load(args.output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    
    # Model
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['blazehand', 'timm'],
                        help='Model type')
    parser.add_argument('--model_name', type=str, default='mobilenetv3_small_100',
                        help='Model name (for timm models)')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of classes (for classification models)')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--strict_load', action='store_true',
                        help='Strict loading of weights')
    
    # Export
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output ONNX file path')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for dummy input')
    parser.add_argument('--opset_version', type=int, default=13,
                        help='ONNX opset version')
    parser.add_argument('--dynamic_batch', action='store_true',
                        help='Enable dynamic batch size')
    parser.add_argument('--verify', action='store_true',
                        help='Verify ONNX model after export')
    
    # System
    parser.add_argument('--use_cuda', action='store_true',
                        help='Use CUDA for export')
    
    args = parser.parse_args()
    export_to_onnx(args)

