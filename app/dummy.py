from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from openai import OpenAI
import aiofiles
import os
import tempfile
from PyPDF2 import PdfReader
from docx import Document

router = APIRouter()
client = OpenAI()

conversation_history = []

def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

@router.post("/")
async def generate_response(
    input: str = Form(...),
    file: UploadFile = File(default=None)
):
    try:
        file_text = ""

        if file:
            filename = file.filename
            if not filename.endswith((".pdf", ".docx")):
                raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed.")

            suffix = os.path.splitext(filename)[-1]
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            async with aiofiles.open(temp_file.name, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)

            # Extract text from file
            if suffix == ".pdf":
                file_text = read_pdf(temp_file.name)
            elif suffix == ".docx":
                file_text = read_docx(temp_file.name)

            os.unlink(temp_file.name)

        full_prompt = input + ("\n\n" + file_text if file_text else "")

        # Add user message to conversation history
        conversation_history.append({"role": "user", "content": full_prompt})

        # Generate response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history
        )

        # Add assistant reply to history
        reply = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": reply})

        return {"output": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))