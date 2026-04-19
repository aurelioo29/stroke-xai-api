from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.health import router as health_router
from app.api.routes.predict import router as predict_router
from app.core.config import get_model_io_details, mri_session, eeg_session
from app.db.database import Base, engine
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Stroke XAI API",
    description="API untuk diagnosis stroke multimodal berbasis MRI + EEG + XAI",
    version="1.0.0"
)

app.include_router(health_router)
app.include_router(predict_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)

    print("=== MRI MODEL INFO ===")
    print(get_model_io_details(mri_session))

    print("=== EEG MODEL INFO ===")
    print(get_model_io_details(eeg_session))

    print("=== DATABASE TABLES CREATED ===")