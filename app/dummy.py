from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
router = APIRouter()
client = OpenAI()

# Request body schema
class PromptRequest(BaseModel):
    input: str

@router.post("/")
async def generate_response(request: PromptRequest):
    try:
        response = client.responses.create(
            model="gpt-4o",
            input=request.input,
        )
        return {"output": response.output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))