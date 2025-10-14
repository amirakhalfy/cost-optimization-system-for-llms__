from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.db_setup import Base

class ModelHistory(Base):
    __tablename__ = "model_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('models.id'), nullable=False)
    input_cost = Column(Float, nullable=False)
    output_cost = Column(Float, nullable=False)
    training_cost = Column(Float)
    currency = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    
    model = relationship("Model", back_populates="model_history")