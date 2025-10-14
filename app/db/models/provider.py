from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.db_setup import Base

class Provider(Base):
    __tablename__ = "providers"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    website = Column(String(255))
    created_at = Column(DateTime, default=datetime.now)
    
    models = relationship("Model", back_populates="provider", cascade="all, delete-orphan")