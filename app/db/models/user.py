from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from ..db_setup import Base

class User(Base):
    __tablename__ = "users"
    
    user_mail = Column(String(255), primary_key=True)
    role = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    project_budgets = relationship("ProjectBudget", secondary="user_project_budgets", back_populates="users")
    usage_logs = relationship("ModelUsageLog", back_populates="user", cascade="all, delete-orphan")


