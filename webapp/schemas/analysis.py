"""Analysis schemas for request/response validation."""

from datetime import datetime
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field


class DetectionSchema(BaseModel):
    """Schema for a single detection."""
    bbox: tuple
    confidence: float
    class_id: int
    class_name: str
    center: Optional[tuple] = None


class AnalysisCreate(BaseModel):
    """Schema for creating analysis (internal use)."""
    session_id: Optional[int] = None
    notes: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Schema for analysis list response."""
    id: int
    original_filename: str
    has_aneurysm: bool
    max_confidence: float
    num_detections: int
    total_time_ms: Optional[float]
    created_at: datetime
    session_id: Optional[int]

    model_config = {"from_attributes": True}


class AnalysisDetailResponse(BaseModel):
    """Schema for detailed analysis response."""
    id: int
    user_id: int
    session_id: Optional[int]

    # Image info
    original_filename: str
    stored_filename: str
    image_path: str
    file_size_bytes: Optional[int]

    # Detection results
    has_aneurysm: bool
    max_confidence: float
    num_detections: int
    detections_json: Optional[List[Dict[str, Any]]]

    # Timing info
    preprocess_time_ms: Optional[float]
    inference_time_ms: Optional[float]
    postprocess_time_ms: Optional[float]
    total_time_ms: Optional[float]

    # Visualization
    visualization_path: Optional[str]

    # Metadata
    created_at: datetime
    notes: Optional[str]

    model_config = {"from_attributes": True}


class NotesUpdate(BaseModel):
    """Schema for updating analysis notes."""
    notes: str = Field(..., max_length=2000)
