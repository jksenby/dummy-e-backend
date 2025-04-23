from typing import Literal

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from openai import OpenAI
import aiofiles
import os

from sqlalchemy.orm import Session
import tempfile
import boto3
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

from .config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
from .database import get_db
from .models import UploadedFile, Request
from .read_files import read_pdf, read_docx

s3 = boto3.resource(
    service_name="s3",
    region_name="eu-north-1",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

model = SentenceTransformer("all-MiniLM-L6-v2")
router = APIRouter()
client = OpenAI()

@router.post("/")
async def generate_response(
    user_input: str = Form(...),
    file: UploadFile = File(default=None),
    db: Session = Depends(get_db)
):
    try:
        file_text = ""
        conversation_history = db.query(Request).order_by(Request.created_at.desc()).all()

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
                    "content": "Generate a eight-word filename in word1_word2_word3_word4 format based on the following text. Only respond with the filename, no additional text or explanation."
                }, {
                    "role": "user",
                    "content": file_text[:4000]  # Limit to first 4000 chars to avoid token limits
                }]
            )

            generated_filename = file_response.choices[0].message.content.strip()

            s3.Bucket("dummy-e").upload_file(Key=f"{generated_filename}{suffix}", Filename=temp_file_path)

            embedding = model.encode(file_text).astype(np.float32)

            # Store embedding as binary blob
            embedding_blob = pickle.dumps(embedding)

            # Save metadata to DB
            db_file = UploadedFile(
                filename=generated_filename,
                embedding_vector=embedding_blob,
            )

            db.add(db_file)
            db.commit()
            db.refresh(db_file)

            os.unlink(temp_file.name)

        else:
            query_embedding = model.encode(user_input).astype(np.float32)

            # Load all saved embeddings from DB
            files = db.query(UploadedFile).all()
            similarities = []

            for f in files:
                if f.embedding_vector:
                    stored_vector = pickle.loads(f.embedding_vector)

                    stored_vector = stored_vector / np.linalg.norm(stored_vector)
                    query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
                    score = np.dot(stored_vector, query_embedding_norm)
                    similarities.append((score, f.filename))

            # Get top matching file
            if similarities:
                best_match = max(similarities, key=lambda x: x[0])
                matched_filename = best_match[1]
            else:
                matched_filename = None

            if matched_filename:
                for ext in ['.pdf', '.docx']:
                    try:
                        obj = s3.Object("dummy-e", f"{matched_filename}{ext}")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                            obj.download_fileobj(f)
                            f.flush()  # Ensure all data is written
                            if ext == '.pdf':
                                file_text = read_pdf(f.name)
                            else:
                                file_text = read_docx(f.name)
                            break
                    except ClientError:
                        continue

        full_prompt = user_input + ("\n\n" + file_text if file_text else "")

        user_message = Request(
            role="user",
            content=full_prompt,
        )
        db.add(user_message)
        db.commit()

        messages = [{
            "role": "user",
            "content": full_prompt
        }]

        for msg in conversation_history:
            messages.insert(0, {
                "role": msg.role,
                "content": msg.content
            })

        # Generate response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

        # Add assistant reply to history
        reply = response.choices[0].message.content
        assistant_message = Request(
            role="assistant",
            content=reply,
        )

        db.add(assistant_message)
        db.commit()

        return {"output": reply}


    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file to S3: {str(e)}")

@router.get("/")
def get_filenames(db: Session = Depends(get_db)):
    uploaded_files = db.query(UploadedFile).all()

    return uploaded_files