"""Analysis API routes."""

from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from webapp.database.connection import get_db
from webapp.database.models import User, Analysis, AnalysisSession
from webapp.schemas.analysis import AnalysisResponse, AnalysisDetailResponse, NotesUpdate
from webapp.auth.dependencies import get_current_user
from webapp.services.analysis_service import get_analysis_service, AnalysisService
from webapp.config import settings

router = APIRouter()


@router.post("", response_model=AnalysisDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis(
    file: UploadFile = File(...),
    session_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    service: AnalysisService = Depends(get_analysis_service)
):
    """
    Upload image and run analysis.

    Args:
        file: Image file to analyze
        session_id: Optional session to add analysis to

    Returns:
        Analysis result with detection details
    """
    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {ext} not allowed. Use: {settings.ALLOWED_EXTENSIONS}"
        )

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB"
        )

    # Validate session if provided
    if session_id:
        session = db.query(AnalysisSession).filter(
            AnalysisSession.id == session_id,
            AnalysisSession.user_id == current_user.id
        ).first()
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

    # Save file
    stored_filename, image_path = service.save_upload(content, file.filename)

    # Run analysis
    try:
        result = service.run_analysis(image_path)
    except Exception as e:
        # Clean up on failure
        service.delete_analysis_files(image_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

    # Convert detections to JSON-serializable format
    detections_json = []
    for det in result.get("detections", []):
        detections_json.append({
            "bbox": list(det.get("bbox", ())),
            "confidence": det.get("confidence", 0),
            "class_id": det.get("class_id", 0),
            "class_name": det.get("class_name", "unknown"),
            "center": list(det.get("center", ())) if det.get("center") else None
        })

    # Create database record
    analysis = Analysis(
        user_id=current_user.id,
        session_id=session_id,
        original_filename=file.filename,
        stored_filename=stored_filename,
        image_path=image_path,
        file_size_bytes=len(content),
        has_aneurysm=result.get("has_aneurysm", False),
        max_confidence=result.get("max_confidence", 0.0),
        num_detections=result.get("num_detections", 0),
        detections_json=detections_json,
        preprocess_time_ms=result.get("timings", {}).get("preprocess_ms"),
        inference_time_ms=result.get("timings", {}).get("inference_ms"),
        postprocess_time_ms=result.get("timings", {}).get("postprocess_ms"),
        total_time_ms=result.get("timings", {}).get("total_ms"),
        visualization_path=result.get("visualization_path")
    )
    db.add(analysis)

    # Update session stats if applicable
    if session_id:
        session.total_images += 1
        if result.get("has_aneurysm"):
            session.aneurysm_detected_count += 1
        session.total_processing_time_ms += result.get("timings", {}).get("total_ms", 0)
        # Recalculate average confidence
        total_conf = session.average_confidence * (session.total_images - 1)
        session.average_confidence = (total_conf + result.get("max_confidence", 0)) / session.total_images

    db.commit()
    db.refresh(analysis)

    return analysis


@router.get("", response_model=List[AnalysisResponse])
async def list_analyses(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    has_aneurysm: Optional[bool] = None,
    session_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List user's analyses with optional filters.
    """
    query = db.query(Analysis).filter(Analysis.user_id == current_user.id)

    if has_aneurysm is not None:
        query = query.filter(Analysis.has_aneurysm == has_aneurysm)

    if session_id is not None:
        query = query.filter(Analysis.session_id == session_id)

    analyses = query.order_by(Analysis.created_at.desc()).offset(skip).limit(limit).all()
    return analyses


@router.get("/{analysis_id}", response_model=AnalysisDetailResponse)
async def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get analysis details by ID.
    """
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()

    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )

    if analysis.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    return analysis


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    service: AnalysisService = Depends(get_analysis_service)
):
    """
    Delete an analysis and its files.
    """
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()

    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )

    if analysis.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Delete files
    service.delete_analysis_files(analysis.image_path, analysis.visualization_path)

    # Delete from database
    db.delete(analysis)
    db.commit()


@router.get("/{analysis_id}/image")
async def get_analysis_image(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get original image file.
    """
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()

    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    if analysis.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    if not Path(analysis.image_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found")

    return FileResponse(analysis.image_path, filename=analysis.original_filename)


@router.get("/{analysis_id}/visualization")
async def get_analysis_visualization(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get visualization image.
    """
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()

    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    if analysis.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    if not analysis.visualization_path or not Path(analysis.visualization_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Visualization not found")

    return FileResponse(analysis.visualization_path)


@router.put("/{analysis_id}/notes", response_model=AnalysisDetailResponse)
async def update_analysis_notes(
    analysis_id: int,
    notes_data: NotesUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update analysis notes.
    """
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()

    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    if analysis.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    analysis.notes = notes_data.notes
    db.commit()
    db.refresh(analysis)

    return analysis
