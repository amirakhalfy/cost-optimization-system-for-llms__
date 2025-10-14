from sqlalchemy import Column, Integer, ForeignKey
from app.db.db_setup import Base

class ModelTask(Base):
    __tablename__ = "model_tasks"
    
    model_id = Column(Integer, ForeignKey('models.id'), primary_key=True)
    task_id = Column(Integer, ForeignKey('tasks.id'), primary_key=True)