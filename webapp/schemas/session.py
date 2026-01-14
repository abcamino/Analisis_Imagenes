"""Session schemas for request/response validation."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

from .analysis import AnalysisResponse


class SessionCreate(BaseModel):
    """Schema for creating analysis session."""
    name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class SessionUpdate(BaseModel):
    """Schema for updating session."""
    name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class SessionResponse(BaseModel):
    """Schema for session list response."""
    id: int
    name: Optional[str]
    description: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    is_active: bool
    total_images: int
    aneurysm_detected_count: int
    average_confidence: float

    model_config = {"from_attributes": True}


class SessionDetailResponse(SessionResponse):
    """Schema for detailed session response with analyses."""
    user_id: int
    total_processing_time_ms: float
    analyses: List[AnalysisResponse] = []

    model_config = {"from_attributes": True}


class DashboardStats(BaseModel):
    """Schema for dashboard statistics."""
    total_analyses: int
    aneurysm_count: int
    detection_rate: float
    avg_processing_time_ms: float
    total_sessions: int
    active_sessions: int


class TrendData(BaseModel):
    """Schema for trend analysis."""
    date: str
    count: int
    aneurysm_count: int
