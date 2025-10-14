from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.db_setup import Base

class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    provider_id = Column(Integer, ForeignKey('providers.id'), nullable=False)
    license = Column(String(255))
    description = Column(Text)
    context_window = Column(Integer)
    max_tokens = Column(Integer)
    parameters = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    provider = relationship("Provider", back_populates="models")
    pricings = relationship("Pricing", back_populates="model", cascade="all, delete-orphan")
    model_history = relationship("ModelHistory", back_populates="model", cascade="all, delete-orphan")
    benchmark_results = relationship("BenchmarkResult", back_populates="model", cascade="all, delete-orphan")
    tasks = relationship("Task", secondary="model_tasks", back_populates="models")
    alerts = relationship("Alert", back_populates="model", cascade="all, delete-orphan")