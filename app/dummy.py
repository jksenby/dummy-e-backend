from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from openai import OpenAI
import aiofiles
import os
import tempfile
from PyPDF2 import PdfReader
from docx import Document
import boto3
from .config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

s3 = boto3.resource(
    service_name="s3",
    region_name="eu-north-1",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)
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
            temp_file_path = None
            filename = file.filename
            if not filename.endswith((".pdf", ".docx")):
                raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed.")

            suffix = os.path.splitext(filename)[-1]
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file_path = temp_file.name
            async with aiofiles.open(temp_file.name, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)

            # Extract text from file
            if suffix == ".pdf":
                file_text = read_pdf(temp_file.name)
            elif suffix == ".docx":
                file_text = read_docx(temp_file.name)

            file_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": "Generate a four-word filename in word1_word2_word3_word4 format based on the following text. Only respond with the filename, no additional text or explanation."
                }, {
                    "role": "user",
                    "content": file_text[:4000]  # Limit to first 4000 chars to avoid token limits
                }]
            )

            generated_filename = file_response.choices[0].message.content.strip()

            s3.Bucket("dummy-e").upload_file(Key=f"{generated_filename}{suffix}", Filename=temp_file_path)

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


    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file to S3: {str(e)}")