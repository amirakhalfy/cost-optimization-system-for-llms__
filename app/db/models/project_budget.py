from sqlalchemy import Column, Integer, Float, String, Text, Boolean, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.db_setup import Base

class ProjectBudget(Base):
    __tablename__ = "project_budgets"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    amount = Column(Float, nullable=False)
    currency = Column(String(255), nullable=False)
    period = Column(String(255), nullable=False)
    alert_threshold = Column(Float, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    alerts = relationship("Alert", back_populates="project_budget", cascade="all, delete-orphan")
    users = relationship("User", secondary="user_project_budgets", back_populates="project_budgets")