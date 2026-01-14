"""Export PyTorch models to ONNX format."""

import torch
from pathlib import Path
from typing import Tuple


def export_to_onnx(
    model_name: str = "mobilenetv3_small_100",
    num_classes: int = 2,
    input_size: Tuple[int, int] = (224, 224),
    output_path: str = "models/onnx/mobilenetv3_aneurysm.onnx",
    weights_path: str = None
):
    """
    Export a timm model to ONNX format.

    Args:
        model_name: Name of the model from timm library
        num_classes: Number of output classes
        input_size: Input image size (height, width)
        output_path: Path to save the ONNX model
        weights_path: Optional path to fine-tuned weights
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm not installed. Run: pip install timm")

    print(f"Creating model: {model_name}")

    # Create model
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        in_chans=1  # Grayscale input
    )

    # Load fine-tuned weights if provided
    if weights_path and Path(weights_path).exists():
        print(f"Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)

    model.eval()

    # Create dummy input (batch=1, channels=1, H, W)
    dummy_input = torch.randn(1, 1, input_size[0], input_size[1])

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    print(f"Exporting to: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model exported successfully!")
    print(f"Input shape: (batch, 1, {input_size[0]}, {input_size[1]})")
    print(f"Output shape: (batch, {num_classes})")

    # Verify the model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation: PASSED")
    except ImportError:
        print("onnx package not installed, skipping validation")

    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model", default="mobilenetv3_small_100", help="Model name from timm")
    parser.add_argument("--classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--size", type=int, default=224, help="Input size")
    parser.add_argument("--output", default="models/onnx/mobilenetv3_aneurysm.onnx", help="Output path")
    parser.add_argument("--weights", default=None, help="Fine-tuned weights path")

    args = parser.parse_args()

    export_to_onnx(
        model_name=args.model,
        num_classes=args.classes,
        input_size=(args.size, args.size),
        output_path=args.output,
        weights_path=args.weights
    )
