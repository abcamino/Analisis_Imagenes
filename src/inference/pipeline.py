"""Main detection pipeline orchestrating C++ preprocessing and ONNX inference."""

import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

from .onnx_inference import create_inference_engine
from .postprocessor import PostProcessor, Detection, create_result_dict


class DetectionPipeline:
    """
    Complete aneurysm detection pipeline.

    Orchestrates:
    1. C++ preprocessing (fast image loading, CLAHE, normalization)
    2. ONNX Runtime inference (optimized model execution)
    3. Python post-processing (NMS, filtering)
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the detection pipeline.

        Args:
            config_path: Path to config.yaml
            config: Direct configuration dictionary (overrides config_path)
        """
        self.config = self._load_config(config_path, config)
        self.preprocessor = None
        self.inference = None
        self.postprocessor = None

        self._init_components()

    def _load_config(self, config_path: Optional[str], config: Optional[Dict]) -> Dict:
        """Load configuration from file or use provided dict."""
        if config:
            return config

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)

        # Default configuration
        return {
            'preprocessing': {
                'target_width': 224,
                'target_height': 224,
                'clahe': {'enabled': True, 'clip_limit': 2.0}
            },
            'model': {
                'onnx_path': 'models/onnx/mobilenetv3_aneurysm.onnx',
                'num_classes': 2,
                'class_names': ['normal', 'aneurysm']
            },
            'inference': {
                'confidence_threshold': 0.5
            },
            'postprocessing': {
                'nms_threshold': 0.45,
                'min_detection_size': 10,
                'max_detections': 10
            }
        }

    def _init_components(self):
        """Initialize pipeline components."""
        # Use Python+OpenCV preprocessing (fastest due to zero-copy numpy sharing)
        # C++ module available but slower due to data copy overhead
        self._use_cpp = False
        try:
            import cv2
            print(f"Using Python+OpenCV preprocessing (optimized)")
        except ImportError:
            print("OpenCV not available")

        # Initialize inference engine (auto-selects best available)
        model_path = self.config['model']['onnx_path']
        self.inference = create_inference_engine(
            model_path,
            self.config['model']['num_classes']
        )

        # Initialize post-processor
        self.postprocessor = PostProcessor(
            confidence_threshold=self.config['inference']['confidence_threshold'],
            nms_threshold=self.config['postprocessing']['nms_threshold'],
            min_size=self.config['postprocessing']['min_detection_size'],
            max_detections=self.config['postprocessing']['max_detections'],
            class_names=self.config['model'].get('class_names', ['normal', 'aneurysm'])
        )

    def run(self, image_path: str) -> Dict[str, Any]:
        """
        Run full detection pipeline on a single image.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with detection results and timing info
        """
        total_start = time.perf_counter()
        timings = {}

        # 1. Preprocessing
        preprocess_start = time.perf_counter()
        if self._use_cpp:
            tensor, rois = self._preprocess_cpp(image_path)
        else:
            tensor, rois = self._preprocess_python(image_path)
        timings['preprocess_ms'] = (time.perf_counter() - preprocess_start) * 1000

        # 2. Inference
        inference_start = time.perf_counter()
        class_id, probs = self.inference.predict(tensor)
        timings['inference_ms'] = (time.perf_counter() - inference_start) * 1000

        # 3. Post-processing
        postprocess_start = time.perf_counter()
        predictions = [(class_id, probs)]
        detections = self.postprocessor.process(predictions, rois)
        timings['postprocess_ms'] = (time.perf_counter() - postprocess_start) * 1000

        # Total time
        timings['total_ms'] = (time.perf_counter() - total_start) * 1000

        # Create result
        result = create_result_dict(detections, timings['total_ms'], image_path)
        result['timings'] = timings
        result['preprocessed_tensor_shape'] = tensor.shape

        return result

    def _preprocess_cpp(self, image_path: str):
        """Preprocess using C++ module."""
        import aneurysm_cpp
        import cv2

        # Load image with OpenCV (faster than pure Python)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Use C++ preprocessing
        target_h = self.config['preprocessing']['target_height']
        target_w = self.config['preprocessing']['target_width']
        processed = aneurysm_cpp.preprocess_image(image, target_h, target_w)
        tensor = processed['normalized']

        return tensor, []

    def _preprocess_python(self, image_path: str):
        """Fallback preprocessing using Python/OpenCV."""
        import cv2

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Resize
        target_w = self.config['preprocessing']['target_width']
        target_h = self.config['preprocessing']['target_height']
        image = cv2.resize(image, (target_w, target_h))

        # CLAHE
        if self.config['preprocessing']['clahe']['enabled']:
            clahe = cv2.createCLAHE(
                clipLimit=self.config['preprocessing']['clahe']['clip_limit'],
                tileGridSize=(8, 8)
            )
            image = clahe.apply(image)

        # Normalize to [0, 1]
        tensor = image.astype(np.float32) / 255.0

        # Convert grayscale to 3-channel (model expects RGB)
        # Replicate the grayscale channel 3 times
        tensor = np.stack([tensor, tensor, tensor], axis=0)  # (3, H, W)

        # Add batch dimension: (1, 3, H, W)
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, []

    def run_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Run pipeline on multiple images."""
        return [self.run(path) for path in image_paths]

    def benchmark(self, image_path: str, n_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark pipeline performance.

        Args:
            image_path: Test image path
            n_runs: Number of runs for averaging

        Returns:
            Dictionary with average timings
        """
        timings = {
            'preprocess_ms': [],
            'inference_ms': [],
            'postprocess_ms': [],
            'total_ms': []
        }

        # Warmup
        self.run(image_path)

        # Benchmark runs
        for _ in range(n_runs):
            result = self.run(image_path)
            for key in timings:
                timings[key].append(result['timings'][key])

        # Calculate averages
        return {
            key: np.mean(values)
            for key, values in timings.items()
        }
