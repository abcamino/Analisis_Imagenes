"""Pytest fixtures for web application tests."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from webapp.main import app
from webapp.database.models import Base, User, Analysis, AnalysisSession
from webapp.database.connection import get_db
from webapp.auth.security import hash_password


# Test database (in-memory SQLite)
TEST_DATABASE_URL = "sqlite://"


@pytest.fixture(scope="function")
def test_db():
    """Create fresh test database for each test."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(test_db):
    """Test client with overridden database."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(test_db):
    """Create a test user in database."""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=hash_password("TestPass123"),
        full_name="Test User"
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def authenticated_client(client, test_user):
    """Client with authenticated session."""
    response = client.post("/auth/login", data={
        "username": "testuser",
        "password": "TestPass123"
    }, follow_redirects=False)
    # Session cookie should be set
    return client


@pytest.fixture
def test_session(test_db, test_user):
    """Create a test analysis session."""
    session = AnalysisSession(
        user_id=test_user.id,
        name="Test Session",
        description="A test session"
    )
    test_db.add(session)
    test_db.commit()
    test_db.refresh(session)
    return session


@pytest.fixture
def test_analysis(test_db, test_user):
    """Create a test analysis."""
    analysis = Analysis(
        user_id=test_user.id,
        original_filename="test.jpg",
        stored_filename="uuid-test.jpg",
        image_path="/path/to/test.jpg",
        has_aneurysm=False,
        max_confidence=0.15,
        num_detections=0
    )
    test_db.add(analysis)
    test_db.commit()
    test_db.refresh(analysis)
    return analysis


@pytest.fixture
def mock_pipeline_result():
    """Mock pipeline result for testing without model."""
    return {
        "image_path": "test.jpg",
        "has_aneurysm": True,
        "max_confidence": 0.85,
        "num_detections": 1,
        "inference_time_ms": 25.5,
        "timings": {
            "preprocess_ms": 4.2,
            "inference_ms": 18.3,
            "postprocess_ms": 3.0,
            "total_ms": 25.5
        },
        "detections": [
            {
                "bbox": (100, 100, 50, 50),
                "confidence": 0.85,
                "class_id": 1,
                "class_name": "aneurysm",
                "center": (125.0, 125.0)
            }
        ]
    }
