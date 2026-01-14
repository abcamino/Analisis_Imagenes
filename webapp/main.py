"""FastAPI application entry point."""

from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from webapp.config import settings
from webapp.database.connection import init_db, get_db
from webapp.database.models import User, UserSession

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Educational aneurysm detection system for brain CT images.",
    version="1.0.0",
)

# Get base directory
BASE_DIR = Path(__file__).resolve().parent

# Static files
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Upload directory
uploads_dir = settings.UPLOAD_DIR
uploads_dir.mkdir(parents=True, exist_ok=True)
settings.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# Templates
templates_dir = BASE_DIR / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_db()


# Import and include routers
from webapp.auth.routes import router as auth_router
from webapp.api.analyses import router as analyses_router
from webapp.api.sessions import router as sessions_router
from webapp.api.dashboard import router as dashboard_router
from webapp.api.admin import router as admin_router

app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(analyses_router, prefix="/api/analyses", tags=["Analyses"])
app.include_router(sessions_router, prefix="/api/sessions", tags=["Sessions"])
app.include_router(dashboard_router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])


# Helper to get current user from cookie for templates
from datetime import datetime

async def get_current_user_for_template(request: Request, db: Session) -> Optional[User]:
    """Get current user from session cookie for template context."""
    session_token = request.cookies.get(settings.SESSION_COOKIE_NAME)
    if not session_token:
        return None

    user_session = db.query(UserSession).filter(
        UserSession.session_token == session_token
    ).first()

    if not user_session or user_session.expires_at < datetime.utcnow():
        return None

    user = db.query(User).filter(User.id == user_session.user_id).first()
    if user and user.is_active:
        return user
    return None


# Page routes
@app.get("/")
async def root():
    """Redirect to dashboard or login."""
    return RedirectResponse(url="/login")


@app.get("/login")
async def login_page(request: Request, db: Session = Depends(get_db)):
    """Login page."""
    current_user = await get_current_user_for_template(request, db)
    return templates.TemplateResponse("auth/login.html", {"request": request, "current_user": current_user})


@app.get("/register")
async def register_page(request: Request, db: Session = Depends(get_db)):
    """Registration page."""
    current_user = await get_current_user_for_template(request, db)
    return templates.TemplateResponse("auth/register.html", {"request": request, "current_user": current_user})


@app.get("/dashboard")
async def dashboard_page(request: Request, db: Session = Depends(get_db)):
    """Main dashboard page."""
    current_user = await get_current_user_for_template(request, db)
    return templates.TemplateResponse("dashboard/index.html", {"request": request, "current_user": current_user})


@app.get("/upload")
async def upload_page(request: Request, db: Session = Depends(get_db)):
    """Image upload page."""
    current_user = await get_current_user_for_template(request, db)
    return templates.TemplateResponse("analysis/upload.html", {"request": request, "current_user": current_user})


@app.get("/analyses")
async def history_page(request: Request, db: Session = Depends(get_db)):
    """Analysis history page."""
    current_user = await get_current_user_for_template(request, db)
    return templates.TemplateResponse("analysis/history.html", {"request": request, "current_user": current_user})


@app.get("/analyses/{analysis_id}")
async def result_page(request: Request, analysis_id: int, db: Session = Depends(get_db)):
    """Single analysis result page."""
    current_user = await get_current_user_for_template(request, db)
    return templates.TemplateResponse(
        "analysis/result.html",
        {"request": request, "analysis_id": analysis_id, "current_user": current_user}
    )


@app.get("/admin/tests")
async def admin_tests_page(request: Request, db: Session = Depends(get_db)):
    """Admin tests viewer page."""
    current_user = await get_current_user_for_template(request, db)
    return templates.TemplateResponse("admin/tests.html", {"request": request, "current_user": current_user})


@app.get("/admin/database")
async def admin_database_page(request: Request, db: Session = Depends(get_db)):
    """Admin database explorer page."""
    current_user = await get_current_user_for_template(request, db)
    return templates.TemplateResponse("admin/database.html", {"request": request, "current_user": current_user})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp.main:app", host="0.0.0.0", port=8000, reload=True)
