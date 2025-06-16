from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.model_classification import classify_profiles

app = FastAPI(title="Bio-Classifier")

class ClassifyRequest(BaseModel):
    bios: List[str]

class ClassifyResponse(BaseModel):
    results: List[str]

@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    payload = [{"username": str(i), "bio": b} for i, b in enumerate(req.bios)]
    classified = classify_profiles(payload)
    print(classified)
    flags = [row["is_christian"] for row in classified]
    return {"results": flags}