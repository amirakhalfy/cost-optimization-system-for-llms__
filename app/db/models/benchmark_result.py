from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.db_setup import Base

class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"
    
    model_id = Column(Integer, ForeignKey('models.id'), primary_key=True)
    benchmark_id = Column(Integer, ForeignKey('benchmarks.id'), primary_key=True)
    score = Column(Float)
    evaluation_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)
    
    model = relationship("Model", back_populates="benchmark_results")
    benchmark = relationship("Benchmark", back_populates="benchmark_results")