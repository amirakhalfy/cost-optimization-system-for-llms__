from sqlalchemy import Column, String, Integer, DateTime, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

from sqlalchemy import Column, String, Integer, DateTime, Text

class PromptCache(Base):
    __tablename__ = "prompt_cache"

    prompt_key = Column(String(64), primary_key=True)
    raw_prompt = Column(Text, nullable=False)
    model_name = Column(String(100), nullable=False)
    max_tokens = Column(Integer, nullable=False)
    response = Column(Text, nullable=False)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
