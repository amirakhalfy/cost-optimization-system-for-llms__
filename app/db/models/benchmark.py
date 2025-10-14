from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.db_setup import Base

class Benchmark(Base):
    __tablename__ = "benchmarks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    score_benchmark = Column(Float)
    description = Column(Text)
    votes = Column(Integer, default=0)
    rank = Column(Integer)
    arena_score = Column(Float)
    confidence_interval = Column(String(255))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    benchmark_results = relationship("BenchmarkResult", back_populates="benchmark", cascade="all, delete-orphan")