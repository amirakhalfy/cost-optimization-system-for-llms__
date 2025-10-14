from sqlalchemy import Column, String, Integer, DateTime, Text, Numeric
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class PromptEmbedding(Base):
    __tablename__ = "prompt_embeddings"
    id = Column(String(36), primary_key=True)
    raw_prompt = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    cost = Column(Numeric(10, 4), default=0.0, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
