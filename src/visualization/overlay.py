"""Visualization utilities for detection results."""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class DetectionOverlay:
    """Draw detection results on images."""

    def __init__(
        self,
        bbox_color: Tuple[int, int, int] = (255, 0, 0),
        bbox_thickness: int = 2,
        font_scale: float = 0.6,
        overlay_alpha: float = 0.4,
        show_confidence: bool = True
    ):
        self.bbox_color = bbox_color
        self.bbox_thickness = bbox_thickness
        self.font_scale = font_scale
        self.overlay_alpha = overlay_alpha
        self.show_confidence = show_confidence

    def draw_detections_cv2(
        self,
        image: np.ndarray,
        detections: List[Dict],
        copy: bool = True
    ) -> np.ndarray:
        """
        Draw detections using OpenCV.

        Args:
            image: Input image (BGR or grayscale)
            detections: List of detection dictionaries
            copy: Whether to copy the image first

        Returns:
            Image with drawn detections
        """
        import cv2

        if copy:
            image = image.copy()

        # Convert grayscale to BGR if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']

            # Draw bounding box
            x, y, w, h = bbox
            color = self.bbox_color if class_name == 'aneurysm' else (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, self.bbox_thickness)

            # Draw label
            if self.show_confidence:
                label = f"{class_name}: {confidence:.1%}"
            else:
                label = class_name

            # Background for label
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y), color, -1)
            cv2.putText(
                image, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 1
            )

        return image

    def draw_detections_matplotlib(
        self,
        image: np.ndarray,
        detections: List[Dict],
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Draw detections using Matplotlib.

        Args:
            image: Input image
            detections: List of detection dictionaries
            figsize: Figure size
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Display image
        if len(image.shape) == 3:
            ax.imshow(image[:, :, ::-1])  # BGR to RGB
        else:
            ax.imshow(image, cmap='gray')

        # Draw detections
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']

            x, y, w, h = bbox
            color = 'red' if class_name == 'aneurysm' else 'green'

            # Draw rectangle
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=self.bbox_thickness,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)

            # Draw label
            if self.show_confidence:
                label = f"{class_name}: {confidence:.1%}"
            else:
                label = class_name

            ax.text(
                x, y - 5, label,
                color='white',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7)
            )

        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")

        return fig

    def create_comparison_view(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        detections: List[Dict],
        result_info: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a side-by-side comparison view.

        Args:
            original: Original image
            processed: Processed/enhanced image
            detections: Detection results
            result_info: Dictionary with timing and other info
            save_path: Optional path to save

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        if len(original.shape) == 3:
            axes[0].imshow(original[:, :, ::-1])
        else:
            axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')

        # Processed image
        if len(processed.shape) == 3:
            axes[1].imshow(processed[:, :, ::-1])
        else:
            axes[1].imshow(processed, cmap='gray')
        axes[1].set_title('Preprocessed (CLAHE)')
        axes[1].axis('off')

        # Detections overlay
        if len(original.shape) == 3:
            axes[2].imshow(original[:, :, ::-1])
        else:
            axes[2].imshow(original, cmap='gray')

        for det in detections:
            bbox = det['bbox']
            x, y, w, h = bbox
            color = 'red' if det['class_name'] == 'aneurysm' else 'green'
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            axes[2].add_patch(rect)

        axes[2].set_title('Detections')
        axes[2].axis('off')

        # Add info text
        info_text = (
            f"Aneurysm detected: {result_info.get('has_aneurysm', False)}\n"
            f"Confidence: {result_info.get('max_confidence', 0):.1%}\n"
            f"Time: {result_info.get('inference_time_ms', 0):.1f}ms"
        )
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightgray'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def generate_report(
        self,
        results: List[Dict[str, Any]],
        output_dir: str = "outputs/reports"
    ) -> str:
        """
        Generate a simple text report of detection results.

        Args:
            results: List of detection results
            output_dir: Output directory

        Returns:
            Path to report file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "detection_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ANEURYSM DETECTION REPORT\n")
            f.write("=" * 60 + "\n\n")

            total_aneurysms = sum(1 for r in results if r.get('has_aneurysm', False))
            f.write(f"Total images analyzed: {len(results)}\n")
            f.write(f"Images with aneurysm detected: {total_aneurysms}\n\n")

            for i, result in enumerate(results):
                f.write(f"--- Image {i + 1} ---\n")
                f.write(f"Path: {result.get('image_path', 'N/A')}\n")
                f.write(f"Aneurysm detected: {result.get('has_aneurysm', False)}\n")
                f.write(f"Max confidence: {result.get('max_confidence', 0):.1%}\n")
                f.write(f"Processing time: {result.get('inference_time_ms', 0):.1f}ms\n")

                if result.get('detections'):
                    f.write("Detections:\n")
                    for det in result['detections']:
                        f.write(f"  - {det['class_name']}: {det['confidence']:.1%}\n")
                f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("NOTE: This is for educational purposes only.\n")
            f.write("NOT for clinical diagnosis.\n")
            f.write("=" * 60 + "\n")

        print(f"Report saved to: {report_path}")
        return str(report_path)
