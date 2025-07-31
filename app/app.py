from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.model_classification import classify_profiles, get_classification_prompt, update_classification_prompt, reset_to_default_prompt

app = FastAPI(title="Bio-Classifier")

class ClassifyRequest(BaseModel):
    bios: List[str]

class ClassifyResponse(BaseModel):
    results: List[str]

class PromptUpdateRequest(BaseModel):
    prompt: str

class PromptResponse(BaseModel):
    prompt: str

@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    payload = [{"username": str(i), "bio": b} for i, b in enumerate(req.bios)]
    flags = classify_profiles(payload)
    return {"results": flags}

@app.get("/prompt", response_model=PromptResponse)
async def get_prompt():
    """Get the current classification prompt"""
    prompt = get_classification_prompt()
    return {"prompt": prompt}

@app.put("/prompt", response_model=PromptResponse)
async def update_prompt(req: PromptUpdateRequest):
    """Update the classification prompt"""
    updated_prompt = update_classification_prompt(req.prompt)
    return {"prompt": updated_prompt}

@app.post("/prompt/reset", response_model=PromptResponse)
async def reset_prompt():
    """Reset the classification prompt to default"""
    reset_prompt = reset_to_default_prompt()
    return {"prompt": reset_prompt}