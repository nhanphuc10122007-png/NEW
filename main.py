from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .models.schemas import HealthResponse
from .services.history_service import init_db
from .api import routes_image, routes_face, routes_text, routes_history

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database on startup
    init_db()
    print(f"[AURA Backend] Database initialized at: {settings.DATABASE_PATH}")
    print(f"[AURA Backend] Gemini API Key configured: {bool(settings.GEMINI_API_KEY)}")
    yield
    print("[AURA Backend] Server shutting down.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API phân tích hình ảnh, thị giác máy tính và trắc địa sinh trắc học chuẩn Light Studio.",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(routes_image.router, prefix=settings.API_PREFIX)
app.include_router(routes_face.router, prefix=settings.API_PREFIX)
app.include_router(routes_text.router, prefix=settings.API_PREFIX)
app.include_router(routes_history.router, prefix=settings.API_PREFIX)

@app.get(f"{settings.API_PREFIX}/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="operational",
        version=settings.VERSION,
        server_time=datetime.now().isoformat(),
        features={
            "yunet_face_detector": True,
            "sface_deep_biometrics": True,
            "sqlite_history_store": True,
            "gemini_multimodal_vision": bool(settings.GEMINI_API_KEY),
            "gemini_model": settings.GEMINI_MODEL if settings.GEMINI_API_KEY else "Not Configured"
        }
    )

@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
