from sqlalchemy import Column, Integer, String, DateTime, BigInteger, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.db_setup import Base

class ModelUsageLog(Base):
    __tablename__ = 'model_usage_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_mail = Column(String(256), ForeignKey('users.user_mail'), nullable=False, index=True)
    model_name = Column(String(256), nullable=False, index=True)
    input_tokens = Column(BigInteger, default=0, nullable=False)
    output_tokens = Column(BigInteger, default=0, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, nullable=False, index=True)
    cached_input = Column(BigInteger, default=0, nullable=False)
    cached_output = Column(BigInteger, default=0, nullable=False)
    cost = Column(Numeric(10, 4), default=0.0, nullable=False)
    user = relationship("User", back_populates="usage_logs")
