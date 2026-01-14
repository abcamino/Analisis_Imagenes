"""ONNX Runtime inference for aneurysm detection."""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import time


class ONNXInference:
    """Optimized inference using ONNX Runtime."""

    def __init__(self, model_path: str, num_classes: int = 2):
        """
        Initialize ONNX inference session.

        Args:
            model_path: Path to .onnx model file
            num_classes: Number of output classes
        """
        self.model_path = Path(model_path)
        self.num_classes = num_classes
        self.session = None
        self.input_name = None
        self.input_shape = None

        if self.model_path.exists():
            self._load_model()

    def _load_model(self):
        """Load ONNX model into session."""
        try:
            import onnxruntime as ort

            # Use CPU execution provider
            providers = ['CPUExecutionProvider']

            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers
            )

            # Get input details
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            self.input_shape = input_info.shape

            print(f"Model loaded: {self.model_path.name}")
            print(f"Input: {self.input_name}, shape: {self.input_shape}")

        except ImportError:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")

    def predict(self, tensor: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Run inference on preprocessed tensor.

        Args:
            tensor: Preprocessed image tensor (NCHW format)

        Returns:
            Tuple of (predicted_class, probabilities)
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Check model path.")

        # Ensure correct shape and dtype
        if tensor.ndim == 2:
            # (H, W) -> (1, 1, H, W)
            tensor = tensor[np.newaxis, np.newaxis, :, :]
        elif tensor.ndim == 3:
            # (H, W, C) -> (1, C, H, W)
            tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, :, :, :]

        tensor = tensor.astype(np.float32)

        # Run inference
        outputs = self.session.run(None, {self.input_name: tensor})
        logits = outputs[0]

        # Apply softmax
        probs = self._softmax(logits[0])
        pred_class = int(np.argmax(probs))

        return pred_class, probs

    def predict_batch(self, tensors: List[np.ndarray]) -> List[Tuple[int, np.ndarray]]:
        """Run inference on multiple tensors."""
        return [self.predict(t) for t in tensors]

    def predict_with_timing(self, tensor: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        Run inference and return timing info.

        Returns:
            Tuple of (predicted_class, probabilities, inference_time_ms)
        """
        start = time.perf_counter()
        pred_class, probs = self.predict(tensor)
        elapsed = (time.perf_counter() - start) * 1000
        return pred_class, probs, elapsed

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.session is not None


class OpenCVDNNInference:
    """
    Inference using OpenCV DNN module.
    Alternative to ONNX Runtime when it's not available (e.g., Python 3.14+).
    """

    def __init__(self, model_path: str, num_classes: int = 2):
        self.model_path = Path(model_path)
        self.num_classes = num_classes
        self.net = None

        if self.model_path.exists():
            self._load_model()

    def _load_model(self):
        """Load ONNX model using OpenCV DNN."""
        import cv2

        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        print(f"Model loaded with OpenCV DNN: {self.model_path.name}")

    def predict(self, tensor: np.ndarray) -> Tuple[int, np.ndarray]:
        """Run inference using OpenCV DNN."""
        if self.net is None:
            raise RuntimeError("Model not loaded")

        # Prepare blob - ensure shape is (1, C, H, W)
        if tensor.ndim == 2:
            # Grayscale (H, W) -> (1, 3, H, W)
            tensor = np.stack([tensor, tensor, tensor], axis=0)
            tensor = tensor[np.newaxis, :, :, :]
        elif tensor.ndim == 3:
            # (H, W, C) or (C, H, W)
            if tensor.shape[2] == 3:  # (H, W, C)
                tensor = np.transpose(tensor, (2, 0, 1))
            tensor = tensor[np.newaxis, :, :, :]
        # tensor.ndim == 4: already (1, C, H, W)

        tensor = tensor.astype(np.float32)

        self.net.setInput(tensor)
        output = self.net.forward()

        probs = self._softmax(output[0])
        pred_class = int(np.argmax(probs))

        return pred_class, probs

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def is_loaded(self) -> bool:
        return self.net is not None


class FallbackInference:
    """
    Heuristic-based fallback when no model is available.
    Uses image statistics for simple classification (for testing only).
    """

    def __init__(self):
        self.loaded = True
        print("Using heuristic fallback (no trained model)")

    def predict(self, tensor: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Simple heuristic-based prediction.
        NOT a real model - just for testing the pipeline.
        """
        mean_val = np.mean(tensor)
        std_val = np.std(tensor)

        # Heuristic: high contrast regions may indicate anomalies
        # This is a placeholder - real detection needs a trained model
        score = (std_val * 2 + (1 - abs(mean_val - 0.5))) / 3

        if score > 0.4:
            prob_aneurysm = min(0.3 + score * 0.5, 0.85)
            return 1, np.array([1 - prob_aneurysm, prob_aneurysm])
        else:
            return 0, np.array([0.7, 0.3])

    def is_loaded(self) -> bool:
        return True


def create_inference_engine(model_path: str, num_classes: int = 2):
    """
    Factory function to create the best available inference engine.

    Priority:
    1. ONNX Runtime (if available)
    2. OpenCV DNN (fallback for Python 3.14+)
    3. Heuristic fallback (no model)
    """
    model_exists = Path(model_path).exists()

    # Try ONNX Runtime first
    try:
        import onnxruntime
        if model_exists:
            return ONNXInference(model_path, num_classes)
    except ImportError:
        pass

    # Try OpenCV DNN
    if model_exists:
        try:
            return OpenCVDNNInference(model_path, num_classes)
        except Exception as e:
            print(f"OpenCV DNN failed: {e}")

    # Fallback
    return FallbackInference()
