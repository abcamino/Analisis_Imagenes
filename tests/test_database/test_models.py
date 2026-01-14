"""Tests for database models."""

import pytest
from sqlalchemy.exc import IntegrityError
from webapp.database.models import User, Analysis, AnalysisSession
from webapp.auth.security import hash_password


class TestUserModel:
    """Tests for User model."""

    def test_create_user(self, test_db):
        """Test creating a user."""
        user = User(
            username="newuser",
            email="new@example.com",
            hashed_password=hash_password("password123")
        )
        test_db.add(user)
        test_db.commit()

        assert user.id is not None
        assert user.is_active is True
        assert user.is_admin is False
        assert user.created_at is not None

    def test_user_unique_username(self, test_db, test_user):
        """Test unique constraint on username."""
        duplicate = User(
            username="testuser",
            email="different@example.com",
            hashed_password="hashed"
        )
        test_db.add(duplicate)

        with pytest.raises(IntegrityError):
            test_db.commit()

    def test_user_unique_email(self, test_db, test_user):
        """Test unique constraint on email."""
        test_db.rollback()  # Clear any pending state
        duplicate = User(
            username="different",
            email="test@example.com",
            hashed_password="hashed"
        )
        test_db.add(duplicate)

        with pytest.raises(IntegrityError):
            test_db.commit()


class TestAnalysisModel:
    """Tests for Analysis model."""

    def test_create_analysis(self, test_db, test_user):
        """Test creating an analysis."""
        analysis = Analysis(
            user_id=test_user.id,
            original_filename="test.jpg",
            stored_filename="uuid.jpg",
            image_path="/path/test.jpg",
            has_aneurysm=True,
            max_confidence=0.85
        )
        test_db.add(analysis)
        test_db.commit()

        assert analysis.id is not None
        assert analysis.created_at is not None

    def test_analysis_user_relationship(self, test_db, test_user):
        """Test analysis-user relationship."""
        analysis = Analysis(
            user_id=test_user.id,
            original_filename="test.jpg",
            stored_filename="uuid.jpg",
            image_path="/path/test.jpg",
            has_aneurysm=False,
            max_confidence=0.1
        )
        test_db.add(analysis)
        test_db.commit()
        test_db.refresh(analysis)

        assert analysis.user == test_user
        assert analysis in test_user.analyses

    def test_analysis_cascade_delete(self, test_db, test_user):
        """Test analyses are deleted when user is deleted."""
        analysis = Analysis(
            user_id=test_user.id,
            original_filename="test.jpg",
            stored_filename="uuid.jpg",
            image_path="/path/test.jpg",
            has_aneurysm=False,
            max_confidence=0.1
        )
        test_db.add(analysis)
        test_db.commit()
        analysis_id = analysis.id

        test_db.delete(test_user)
        test_db.commit()

        result = test_db.query(Analysis).filter_by(id=analysis_id).first()
        assert result is None


class TestSessionModel:
    """Tests for AnalysisSession model."""

    def test_create_session(self, test_db, test_user):
        """Test creating a session."""
        session = AnalysisSession(
            user_id=test_user.id,
            name="Test Session"
        )
        test_db.add(session)
        test_db.commit()

        assert session.id is not None
        assert session.is_active is True
        assert session.total_images == 0

    def test_session_analyses_relationship(self, test_db, test_user):
        """Test session-analyses relationship."""
        session = AnalysisSession(user_id=test_user.id, name="Test")
        test_db.add(session)
        test_db.commit()

        for i in range(3):
            analysis = Analysis(
                user_id=test_user.id,
                session_id=session.id,
                original_filename=f"test{i}.jpg",
                stored_filename=f"uuid{i}.jpg",
                image_path=f"/path/uuid{i}.jpg",
                has_aneurysm=False,
                max_confidence=0.1
            )
            test_db.add(analysis)
        test_db.commit()

        test_db.refresh(session)
        assert len(session.analyses) == 3
