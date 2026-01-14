"""SQLAlchemy ORM models."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, Text, JSON
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class User(Base):
    """User account for authentication and analysis tracking."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    sessions = relationship(
        "AnalysisSession",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    analyses = relationship(
        "Analysis",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    user_sessions = relationship(
        "UserSession",
        back_populates="user",
        cascade="all, delete-orphan"
    )


class AnalysisSession(Base):
    """Groups multiple analyses into a logical session."""

    __tablename__ = "analysis_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    name = Column(String(100))
    description = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    is_active = Column(Boolean, default=True)

    # Aggregated stats (updated after each analysis)
    total_images = Column(Integer, default=0)
    aneurysm_detected_count = Column(Integer, default=0)
    average_confidence = Column(Float, default=0.0)
    total_processing_time_ms = Column(Float, default=0.0)

    # Relationships
    user = relationship("User", back_populates="sessions")
    analyses = relationship(
        "Analysis",
        back_populates="session",
        cascade="all, delete-orphan"
    )


class Analysis(Base):
    """Individual image analysis result."""

    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    session_id = Column(
        Integer,
        ForeignKey("analysis_sessions.id", ondelete="SET NULL"),
        nullable=True
    )

    # Image info
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer)

    # Detection results (from pipeline output)
    has_aneurysm = Column(Boolean, nullable=False)
    max_confidence = Column(Float, nullable=False)
    num_detections = Column(Integer, default=0)
    detections_json = Column(JSON)

    # Timing info
    preprocess_time_ms = Column(Float)
    inference_time_ms = Column(Float)
    postprocess_time_ms = Column(Float)
    total_time_ms = Column(Float)

    # Visualization
    visualization_path = Column(String(500))

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)

    # Relationships
    user = relationship("User", back_populates="analyses")
    session = relationship("AnalysisSession", back_populates="analyses")


class UserSession(Base):
    """HTTP session for authentication (cookie-based)."""

    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))

    # Relationship
    user = relationship("User", back_populates="user_sessions")
