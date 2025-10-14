from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.db_setup import Base

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String(255), nullable=False)
    severity = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    user_mail = Column(String(255), ForeignKey('users.user_mail'))
    project_budget_id = Column(Integer, ForeignKey('project_budgets.id'))
    model_id = Column(Integer, ForeignKey('models.id'))
    resolved = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.now)
    
    user = relationship("User", back_populates="alerts")
    project_budget = relationship("ProjectBudget", back_populates="alerts")
    model = relationship("Model", back_populates="alerts")