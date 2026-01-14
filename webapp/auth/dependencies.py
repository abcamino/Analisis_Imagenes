"""Authentication dependencies for FastAPI."""

from datetime import datetime
from typing import Optional
from fastapi import Depends, HTTPException, status, Cookie
from sqlalchemy.orm import Session

from webapp.database.connection import get_db
from webapp.database.models import User, UserSession
from webapp.config import settings


async def get_current_user(
    session_token: Optional[str] = Cookie(None, alias=settings.SESSION_COOKIE_NAME),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current authenticated user from session cookie.

    Args:
        session_token: Session token from cookie
        db: Database session

    Returns:
        Authenticated user

    Raises:
        HTTPException: If not authenticated or session expired
    """
    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    # Find session
    user_session = db.query(UserSession).filter(
        UserSession.session_token == session_token
    ).first()

    if not user_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session"
        )

    # Check expiration
    if user_session.expires_at < datetime.utcnow():
        db.delete(user_session)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired"
        )

    # Get user
    user = db.query(User).filter(User.id == user_session.user_id).first()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )

    return user


async def get_current_user_optional(
    session_token: Optional[str] = Cookie(None, alias=settings.SESSION_COOKIE_NAME),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.

    Used for pages that work both authenticated and not.
    """
    if not session_token:
        return None

    try:
        return await get_current_user(session_token, db)
    except HTTPException:
        return None


async def get_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user and verify they are admin.

    Raises:
        HTTPException: If user is not admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user
