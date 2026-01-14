"""Post-processing for detection results."""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[float, float]


class PostProcessor:
    """Post-processing for detection results."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        min_size: int = 10,
        max_detections: int = 10,
        class_names: List[str] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.min_size = min_size
        self.max_detections = max_detections
        self.class_names = class_names or ["normal", "aneurysm"]

    def process(
        self,
        predictions: List[Tuple[int, np.ndarray]],
        rois: List[Any] = None
    ) -> List[Detection]:
        """
        Process raw predictions into detection results.

        Args:
            predictions: List of (class_id, probabilities) tuples
            rois: Optional list of ROI objects from C++ module

        Returns:
            List of Detection objects
        """
        detections = []

        for i, (class_id, probs) in enumerate(predictions):
            confidence = float(probs[class_id])

            # Filter by confidence
            if confidence < self.confidence_threshold:
                continue

            # Get bbox from ROI if available
            if rois and i < len(rois):
                roi = rois[i]
                bbox = roi.bbox if hasattr(roi, 'bbox') else (0, 0, 0, 0)
                center = roi.center if hasattr(roi, 'center') else (0.0, 0.0)
            else:
                bbox = (0, 0, 0, 0)
                center = (0.0, 0.0)

            # Filter by size
            if bbox[2] < self.min_size or bbox[3] < self.min_size:
                continue

            detection = Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=class_id,
                class_name=self.class_names[class_id],
                center=center
            )
            detections.append(detection)

        # Apply NMS
        detections = self._apply_nms(detections)

        # Limit number of detections
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        detections = detections[:self.max_detections]

        return detections

    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping detections
            detections = [
                d for d in detections
                if self._iou(best.bbox, d.bbox) < self.nms_threshold
            ]

        return keep

    @staticmethod
    def _iou(box1: Tuple[int, ...], box2: Tuple[int, ...]) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0


def create_result_dict(
    detections: List[Detection],
    inference_time_ms: float,
    image_path: str = ""
) -> Dict[str, Any]:
    """Create a structured result dictionary."""
    has_aneurysm = any(d.class_name == "aneurysm" for d in detections)
    max_confidence = max((d.confidence for d in detections), default=0.0)

    return {
        "image_path": image_path,
        "has_aneurysm": has_aneurysm,
        "num_detections": len(detections),
        "max_confidence": max_confidence,
        "inference_time_ms": inference_time_ms,
        "detections": [
            {
                "bbox": d.bbox,
                "confidence": d.confidence,
                "class_id": d.class_id,
                "class_name": d.class_name,
                "center": d.center
            }
            for d in detections
        ]
    }
