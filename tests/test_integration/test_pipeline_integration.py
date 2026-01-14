"""Tests for pipeline integration with web app."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestAnalysisService:
    """Tests for AnalysisService integration."""

    def test_save_upload(self, tmp_path):
        """Test file upload saving."""
        # Patch settings to use temp directory
        with patch('webapp.services.analysis_service.settings') as mock_settings:
            mock_settings.UPLOAD_DIR = tmp_path / "uploads"
            mock_settings.VISUALIZATION_DIR = tmp_path / "viz"
            mock_settings.PIPELINE_CONFIG_PATH = "config.yaml"

            from webapp.services.analysis_service import AnalysisService

            # Reset singleton for testing
            AnalysisService._instance = None
            AnalysisService._pipeline = None

            with patch.object(AnalysisService, '__init__', lambda x: None):
                service = AnalysisService()
                service._pipeline = MagicMock()
                service._overlay = MagicMock()

                # Create upload dir
                (tmp_path / "uploads").mkdir(exist_ok=True)

                # Test save
                content = b"fake image content"
                stored_name, full_path = service.save_upload(content, "original.jpg")

                assert Path(full_path).exists()
                assert stored_name.endswith(".jpg")
                assert stored_name != "original.jpg"


class TestPipelineResultMapping:
    """Tests for mapping pipeline results to database models."""

    def test_result_mapping(self, test_db, test_user, mock_pipeline_result):
        """Test pipeline result maps correctly to Analysis model."""
        from webapp.database.models import Analysis

        analysis = Analysis(
            user_id=test_user.id,
            original_filename="test.jpg",
            stored_filename="uuid.jpg",
            image_path="/path/uuid.jpg",
            has_aneurysm=mock_pipeline_result["has_aneurysm"],
            max_confidence=mock_pipeline_result["max_confidence"],
            num_detections=mock_pipeline_result["num_detections"],
            preprocess_time_ms=mock_pipeline_result["timings"]["preprocess_ms"],
            inference_time_ms=mock_pipeline_result["timings"]["inference_ms"],
            total_time_ms=mock_pipeline_result["timings"]["total_ms"]
        )
        test_db.add(analysis)
        test_db.commit()

        assert analysis.has_aneurysm == mock_pipeline_result["has_aneurysm"]
        assert analysis.max_confidence == mock_pipeline_result["max_confidence"]
        assert analysis.num_detections == mock_pipeline_result["num_detections"]
        assert analysis.preprocess_time_ms == mock_pipeline_result["timings"]["preprocess_ms"]


class TestFullWorkflow:
    """Tests for complete user workflows."""

    def test_register_login_workflow(self, client):
        """Test complete user workflow from registration to login."""
        # 1. Register
        response = client.post("/auth/register", json={
            "username": "workflowuser",
            "email": "workflow@example.com",
            "password": "SecurePass123",
            "full_name": "Workflow User"
        })
        assert response.status_code == 201

        # 2. Login
        response = client.post("/auth/login", data={
            "username": "workflowuser",
            "password": "SecurePass123"
        }, follow_redirects=False)
        assert response.status_code == 302
        assert "session_token" in response.cookies

        # 3. Access protected endpoint
        response = client.get("/auth/me")
        assert response.status_code == 200
        assert response.json()["username"] == "workflowuser"

    def test_session_workflow(self, authenticated_client):
        """Test creating session and viewing it."""
        # 1. Create session
        response = authenticated_client.post("/api/sessions", json={
            "name": "Batch Analysis",
            "description": "Testing multiple images"
        })
        assert response.status_code == 201
        session_id = response.json()["id"]

        # 2. Get session details
        response = authenticated_client.get(f"/api/sessions/{session_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "Batch Analysis"

        # 3. End session
        response = authenticated_client.post(f"/api/sessions/{session_id}/end")
        assert response.status_code == 200
        assert response.json()["is_active"] is False

        # 4. List sessions
        response = authenticated_client.get("/api/sessions")
        assert response.status_code == 200
        assert any(s["id"] == session_id for s in response.json())
