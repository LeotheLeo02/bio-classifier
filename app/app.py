from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.model_classification import (
    classify_profiles,
    get_classification_prompt,
    update_classification_prompt,
    reset_to_default_prompt,
    get_editable_criteria,
)

app = FastAPI(title="Bio-Classifier")

class ClassifyRequest(BaseModel):
    bios: List[str]
    criteria: str | None = None

class ClassifyResponse(BaseModel):
    results: List[str]

class PromptUpdateRequest(BaseModel):
    criteria: str

class PromptResponse(BaseModel):
    prompt: str

@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    payload = [{"username": str(i), "bio": b} for i, b in enumerate(req.bios)]
    # If per-request criteria is provided, temporarily override for this call only
    if req.criteria and isinstance(req.criteria, str) and req.criteria.strip():
        print(f"🧪 [DEBUG] Using per-request criteria for /classify (chars={len(req.criteria)})")
        from app import model_classification as mc
        original_criteria = mc._current_criteria
        original_prompt = mc._current_prompt
        try:
            mc._current_criteria = req.criteria
            mc._current_prompt = mc._build_full_prompt(req.criteria)
            flags = classify_profiles(payload)
        finally:
            mc._current_criteria = original_criteria
            mc._current_prompt = original_prompt
    else:
        print("ℹ️ [DEBUG] /classify using default server criteria")
        flags = classify_profiles(payload)
    return {"results": flags}

@app.get("/prompt", response_model=PromptResponse)
async def get_prompt():
    """Get the current classification prompt"""
    prompt = get_classification_prompt()
    return {"prompt": prompt}

@app.get("/criteria")
async def get_criteria():
    return {"criteria": get_editable_criteria()}

@app.put("/prompt", response_model=PromptResponse)
async def update_prompt(req: PromptUpdateRequest):
    """Update only the editable criteria; boilerplate is fixed server-side."""
    updated_prompt = update_classification_prompt(req.criteria)
    return {"prompt": updated_prompt}

@app.post("/prompt/reset", response_model=PromptResponse)
async def reset_prompt():
    """Reset the classification prompt to default"""
    reset_prompt = reset_to_default_prompt()
    return {"prompt": reset_prompt}