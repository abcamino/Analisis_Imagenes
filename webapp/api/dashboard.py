"""Dashboard API routes."""

from typing import List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from webapp.database.connection import get_db
from webapp.database.models import User, Analysis, AnalysisSession
from webapp.schemas.session import DashboardStats
from webapp.schemas.analysis import AnalysisResponse
from webapp.auth.dependencies import get_current_user

router = APIRouter()


@router.get("/stats", response_model=DashboardStats)
async def get_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get dashboard statistics for current user.
    """
    # Total analyses
    total_analyses = db.query(func.count(Analysis.id)).filter(
        Analysis.user_id == current_user.id
    ).scalar() or 0

    # Aneurysm count
    aneurysm_count = db.query(func.count(Analysis.id)).filter(
        Analysis.user_id == current_user.id,
        Analysis.has_aneurysm == True
    ).scalar() or 0

    # Detection rate
    detection_rate = aneurysm_count / total_analyses if total_analyses > 0 else 0.0

    # Average processing time
    avg_time = db.query(func.avg(Analysis.total_time_ms)).filter(
        Analysis.user_id == current_user.id
    ).scalar() or 0.0

    # Total sessions
    total_sessions = db.query(func.count(AnalysisSession.id)).filter(
        AnalysisSession.user_id == current_user.id
    ).scalar() or 0

    # Active sessions
    active_sessions = db.query(func.count(AnalysisSession.id)).filter(
        AnalysisSession.user_id == current_user.id,
        AnalysisSession.is_active == True
    ).scalar() or 0

    return DashboardStats(
        total_analyses=total_analyses,
        aneurysm_count=aneurysm_count,
        detection_rate=detection_rate,
        avg_processing_time_ms=avg_time,
        total_sessions=total_sessions,
        active_sessions=active_sessions
    )


@router.get("/recent", response_model=List[AnalysisResponse])
async def get_recent_analyses(
    limit: int = Query(5, ge=1, le=20),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get most recent analyses for dashboard.
    """
    analyses = db.query(Analysis).filter(
        Analysis.user_id == current_user.id
    ).order_by(Analysis.created_at.desc()).limit(limit).all()

    return analyses
