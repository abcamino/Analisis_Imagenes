"""Service layer that wraps the existing DetectionPipeline."""

import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import sys

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.inference.pipeline import DetectionPipeline
from src.visualization.overlay import DetectionOverlay
from webapp.config import settings


class AnalysisService:
    """Wraps DetectionPipeline for web application use."""

    _instance: Optional["AnalysisService"] = None
    _pipeline: Optional[DetectionPipeline] = None
    _overlay: Optional[DetectionOverlay] = None

    def __new__(cls):
        """Singleton pattern to reuse pipeline instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize service with pipeline and overlay."""
        if self._pipeline is None:
            self._pipeline = DetectionPipeline(config_path=str(settings.PIPELINE_CONFIG_PATH))
            self._overlay = DetectionOverlay()

            # Ensure directories exist
            settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            settings.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def pipeline(self) -> DetectionPipeline:
        return self._pipeline

    @property
    def overlay(self) -> DetectionOverlay:
        return self._overlay

    def save_upload(self, file_content: bytes, original_filename: str) -> Tuple[str, str]:
        """
        Save uploaded file with UUID-based name.

        Args:
            file_content: Raw file bytes
            original_filename: Original filename from upload

        Returns:
            Tuple of (stored_filename, full_path)
        """
        ext = Path(original_filename).suffix.lower()
        stored_filename = f"{uuid.uuid4()}{ext}"
        full_path = settings.UPLOAD_DIR / stored_filename

        with open(full_path, "wb") as f:
            f.write(file_content)

        return stored_filename, str(full_path)

    def run_analysis(
        self,
        image_path: str,
        generate_visualization: bool = True
    ) -> Dict[str, Any]:
        """
        Run detection pipeline on image.

        Args:
            image_path: Path to image file
            generate_visualization: Whether to create visualization

        Returns:
            Extended result dictionary with visualization path
        """
        # Run existing pipeline
        result = self.pipeline.run(image_path)

        # Generate visualization if requested
        if generate_visualization:
            import cv2

            image = cv2.imread(image_path)
            if image is not None:
                viz_filename = f"{Path(image_path).stem}_viz.png"
                viz_path = settings.VISUALIZATION_DIR / viz_filename

                self.overlay.draw_detections_matplotlib(
                    image,
                    result.get("detections", []),
                    save_path=str(viz_path)
                )
                result["visualization_path"] = str(viz_path)

        return result

    def delete_analysis_files(
        self,
        image_path: str,
        viz_path: Optional[str] = None
    ) -> None:
        """
        Remove analysis files from disk.

        Args:
            image_path: Path to original image
            viz_path: Optional path to visualization
        """
        Path(image_path).unlink(missing_ok=True)
        if viz_path:
            Path(viz_path).unlink(missing_ok=True)

    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        path = Path(file_path)
        return path.stat().st_size if path.exists() else 0


# Global singleton instance
analysis_service = AnalysisService()


def get_analysis_service() -> AnalysisService:
    """Dependency for getting analysis service."""
    return analysis_service
