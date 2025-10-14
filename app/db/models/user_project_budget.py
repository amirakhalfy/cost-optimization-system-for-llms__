from sqlalchemy import Column, Integer, String, ForeignKey
from app.db.db_setup import Base

class UserProjectBudget(Base):
    __tablename__ = "user_project_budgets"
    
    user_mail = Column(String(255), ForeignKey('users.user_mail'), primary_key=True)
    project_budget_id = Column(Integer, ForeignKey('project_budgets.id'), primary_key=True)