from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from app.db.database import Base


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True, index=True)

    mri_filename = Column(String, nullable=True)

    mri_prediction_index = Column(Integer, nullable=True)
    mri_prediction_label = Column(String, nullable=True)
    mri_confidence = Column(Float, nullable=True)
    mri_probabilities = Column(Text, nullable=True)

    eeg_prediction_index = Column(Integer, nullable=True)
    eeg_prediction_label = Column(String, nullable=True)
    eeg_confidence = Column(Float, nullable=True)
    eeg_probabilities = Column(Text, nullable=True)

    fusion_prediction_index = Column(Integer, nullable=True)
    fusion_prediction_label = Column(String, nullable=True)
    fusion_confidence = Column(Float, nullable=True)
    fusion_probabilities = Column(Text, nullable=True)

    heatmap_url = Column(String, nullable=True)
    overlay_url = Column(String, nullable=True)
    xai_method = Column(String, nullable=True)
    explanation_text = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())