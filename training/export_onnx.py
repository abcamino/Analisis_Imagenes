"""
Export PyTorch model to ONNX format.
Requires Python 3.11/3.12 with PyTorch.

Usage:
    python export_onnx.py --checkpoint models/best_model.pth --output models/onnx/mobilenetv3_aneurysm.onnx
"""

import argparse
from pathlib import Path

try:
    import torch
    import torch.onnx
    import timm
    import onnx
    import onnxruntime as ort
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Error: {e}")
    print("\nThis script requires Python 3.11 or 3.12 with PyTorch.")
    TORCH_AVAILABLE = False


def create_model(num_classes=2):
    """Create MobileNetV3 model."""
    model = timm.create_model(
        'mobilenetv3_large_100',
        pretrained=False,
        num_classes=num_classes
    )
    return model


def export_to_onnx(checkpoint_path, output_path, num_classes=2):
    """Export PyTorch model to ONNX format."""

    # Create model
    model = create_model(num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_acc']*100:.2f}%")

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"\nExported ONNX model to: {output_path}")

    # Verify the model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    # Test with ONNX Runtime
    print("\nTesting with ONNX Runtime...")
    session = ort.InferenceSession(str(output_path))
    input_name = session.get_inputs()[0].name

    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    output = session.run(None, {input_name: test_input})

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output[0].shape}")
    print(f"Output values: {output[0]}")

    # Softmax to get probabilities
    probs = np.exp(output[0]) / np.sum(np.exp(output[0]), axis=1, keepdims=True)
    print(f"Probabilities: {probs}")

    print("\nONNX export successful!")
    print(f"\nTo use the model in Python 3.14:")
    print(f"  Copy {output_path} to the main project's models/onnx/ directory")


def export_pretrained_onnx(output_path, num_classes=2):
    """Export pretrained model directly (without training)."""

    print("Creating pretrained MobileNetV3 model...")
    model = timm.create_model(
        'mobilenetv3_large_100',
        pretrained=True,
        num_classes=num_classes
    )
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use dynamo=False to use legacy exporter (more compatible)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamo=False
    )

    print(f"Exported pretrained ONNX model to: {output_path}")

    # Verify
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    # Test with ONNX Runtime
    print("Testing with ONNX Runtime...")
    session = ort.InferenceSession(str(output_path))
    input_name = session.get_inputs()[0].name

    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    output = session.run(None, {input_name: test_input})

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output[0].shape}")

    print("NOTE: This is a pretrained ImageNet model, NOT trained for aneurysm detection.")
    print("For real aneurysm detection, you need to train on medical imaging data.")


def main():
    if not TORCH_AVAILABLE:
        return

    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--checkpoint', type=str, help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='models/onnx/mobilenetv3_aneurysm.onnx',
                        help='Output ONNX path')
    parser.add_argument('--pretrained', action='store_true',
                        help='Export pretrained model without training (for testing)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    args = parser.parse_args()

    if args.pretrained:
        export_pretrained_onnx(args.output, args.num_classes)
    elif args.checkpoint:
        export_to_onnx(args.checkpoint, args.output, args.num_classes)
    else:
        print("Please specify --checkpoint or --pretrained")
        print("\nExamples:")
        print("  python export_onnx.py --checkpoint models/best_model.pth")
        print("  python export_onnx.py --pretrained")


if __name__ == '__main__':
    main()
