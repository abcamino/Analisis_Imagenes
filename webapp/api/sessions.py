"""Analysis session API routes."""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from webapp.database.connection import get_db
from webapp.database.models import User, AnalysisSession
from webapp.schemas.session import SessionCreate, SessionResponse, SessionDetailResponse, SessionUpdate
from webapp.auth.dependencies import get_current_user

router = APIRouter()


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_data: SessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new analysis session.
    """
    session = AnalysisSession(
        user_id=current_user.id,
        name=session_data.name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        description=session_data.description
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return session


@router.get("", response_model=List[SessionResponse])
async def list_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    is_active: Optional[bool] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List user's analysis sessions.
    """
    query = db.query(AnalysisSession).filter(
        AnalysisSession.user_id == current_user.id
    )

    if is_active is not None:
        query = query.filter(AnalysisSession.is_active == is_active)

    sessions = query.order_by(AnalysisSession.started_at.desc()).offset(skip).limit(limit).all()
    return sessions


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get session details with analyses.
    """
    session = db.query(AnalysisSession).filter(
        AnalysisSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if session.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    return session


@router.put("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: int,
    session_data: SessionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update session details.
    """
    session = db.query(AnalysisSession).filter(
        AnalysisSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    if session_data.name is not None:
        session.name = session_data.name
    if session_data.description is not None:
        session.description = session_data.description

    db.commit()
    db.refresh(session)

    return session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a session and all its analyses.
    """
    session = db.query(AnalysisSession).filter(
        AnalysisSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if session.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    db.delete(session)
    db.commit()


@router.post("/{session_id}/end", response_model=SessionResponse)
async def end_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    End an active session.
    """
    session = db.query(AnalysisSession).filter(
        AnalysisSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    session.is_active = False
    session.ended_at = datetime.utcnow()
    db.commit()
    db.refresh(session)

    return session
