from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.db_setup import Base

class Pricing(Base):
    __tablename__ = "pricing"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('models.id'), nullable=False)
    input_cost = Column(Float)
    output_cost = Column(Float)
    cached_input = Column(Float)
    training_cost = Column(Float)
    token_unit = Column(String(255), nullable=False)  
    currency = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    model = relationship("Model", back_populates="pricings")