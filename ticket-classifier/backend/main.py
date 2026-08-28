from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from model import predict_category

app = FastAPI(title="Ticket Classifier API")


class TicketRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        return v


class TicketResponse(BaseModel):
    category: str
    confidence: float


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict", response_model=TicketResponse)
def predict(request: TicketRequest):
    try:
        category, confidence = predict_category(request.text)
        return TicketResponse(category=category, confidence=round(confidence, 3))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
