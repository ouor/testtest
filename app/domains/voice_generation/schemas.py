from pydantic import BaseModel, Field

class GenerateVoiceRequest(BaseModel):
    ref_text: str = Field(..., min_length=1, max_length=2000)
    text: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(..., min_length=2, max_length=10)
