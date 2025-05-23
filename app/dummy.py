from typing import Literal

import openai
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from openai import OpenAI
import os

from sqlalchemy import func
from sqlalchemy.orm import Session
import tempfile
import boto3
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import tiktoken
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import math

from .auth import get_current_user
from .config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
from .database import get_db
from .models import UploadedFile, User, Conversation, Message
from .read_files import read_pdf, read_docx
from .schemas import FileResponse

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

s3 = boto3.resource(
    service_name="s3",
    region_name="eu-north-1",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

tokenizer = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def truncate_text(text: str, max_tokens: int) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(lambda e: isinstance(e, (openai.RateLimitError, openai.APITimeoutError)))
)
def make_openai_request(messages, max_retries=3):
    try:
        # Calculate total tokens
        total_tokens = sum(count_tokens(msg["content"]) for msg in messages)

        # If over limit, truncate the oldest messages (keeping system messages)
        while total_tokens > 30000 and len(messages) > 1:
            removed = messages.pop(1)  # Keep system message at index 0 if exists
            total_tokens -= count_tokens(removed["content"])

        # If still over limit, truncate content
        if total_tokens > 30000:
            for msg in messages:
                if count_tokens(msg["content"]) > 1000:  # Only truncate long messages
                    msg["content"] = truncate_text(msg["content"], 12000)
        return client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=12000  # Limit response size
        )
    except openai.RateLimitError as e:
        if max_retries > 0:
            time.sleep(math.pow(2, 4 - max_retries))  # Exponential backoff
            return make_openai_request(messages, max_retries - 1)
        raise

model = SentenceTransformer("all-MiniLM-L6-v2")
router = APIRouter()
client = OpenAI()

def generate_conversation_title(content: str) -> str:
    """Generate a concise title based on the conversation content"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Generate a 3-5 word title summarizing this content. Only return the title, no other text."
            }, {
                "role": "user",
                "content": content[:2000]  # Limit to first 2000 chars
            }],
            max_tokens=20
        )
        return response.choices[0].message.content.strip()
    except:
        # Fallback if title generation fails
        return content[:30] + "..." if len(content) > 30 else content

@router.post("/conversations/{conversation_id}/messages")
async def generate_response(
        conversation_id: int,
        user_input: str = Form(...),
        file: UploadFile = File(default=None),
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    try:
        logger.info(f"Starting message processing for conversation {conversation_id}")
        # Verify conversation belongs to user
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        ).first()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if not conversation.s3_bucket:
            raise HTTPException(status_code=400, detail="Conversation bucket not initialized")

        file_text = ""
        file_metadata = None

        if not conversation.title or conversation.title == "New Conversation":
            title_content = user_input
            file_content = file_text if file_text else ""
            conversation.title = generate_conversation_title(f"{title_content} {file_content}"[:500])

        # File processing
        if file:
            filename = file.filename
            if not filename.endswith((".pdf", ".docx")):
                raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")

            suffix = os.path.splitext(filename)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name

            try:
                # Read file content
                if suffix == ".pdf":
                    file_text = read_pdf(temp_file_path)
                elif suffix == ".docx":
                    file_text = read_docx(temp_file_path)

                file_text = truncate_text(file_text, 8000)

                # Generate filename using AI
                file_response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{
                        "role": "system",
                        "content": "Generate an eight-word filename in word1_word2_word3 format based on the text. Only respond with the filename."
                    }, {
                        "role": "user",
                        "content": file_text[:4000]
                    }]
                )
                generated_filename = file_response.choices[0].message.content.strip()
                generated_filename = ''.join(c if c.isalnum() or c == '_' else '_' for c in generated_filename)

                # Upload to S3
                s3_key = f"{generated_filename}{suffix}"
                s3.Bucket("dummy-e").upload_file(Key=s3_key, Filename=temp_file_path)

                # Create embedding
                embedding = model.encode(file_text).astype(np.float32)
                embedding_blob = pickle.dumps(embedding)

                # Save metadata
                file_metadata = UploadedFile(
                    filename=generated_filename,
                    original_filename=filename,
                    file_type=suffix[1:],  # Remove dot
                    s3_key=s3_key,
                    embedding_vector=embedding_blob,
                    conversation_id=conversation_id
                )
                db.add(file_metadata)
                db.commit()

            finally:
                os.unlink(temp_file_path)

        else:
            if not user_input:
                raise HTTPException(status_code=400, detail="No file or query text provided")

            # Generate embedding from the user's input text
            input_embedding = model.encode(user_input).astype(np.float32)

            # Retrieve all stored files and their embeddings
            all_files = db.query(UploadedFile).all()
            best_match = None
            best_similarity = -1

            for file in all_files:
                stored_embedding = pickle.loads(file.embedding_vector)
                similarity = cosine_similarity(input_embedding, stored_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = file
            print(f"Total files: {len(all_files)}")
            print(f"Files with embeddings: {sum(1 for f in all_files if f.embedding_vector is not None)}")
            print(f"Best similarity: {best_similarity:.4f}")

            if not best_match:
                raise HTTPException(status_code=404, detail="No similar document found")

            ext = "." + best_match.file_type
            s3_key = f"{best_match.filename}{ext}"
            obj = s3.Object("dummy-e", f"{s3_key}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                obj.download_fileobj(f)
                f.flush()  # Ensure all data is written
                if ext == '.pdf':
                    file_text = read_pdf(f.name)
                else:
                    file_text = read_docx(f.name)
        # Create user message
        full_content = user_input
        if file_text:
            full_content += f"\n\n<----->\n{file_text}\n<----->"

        if user_input:
            user_message = Message(  # Changed from Request to Message
                role="user",
                content=full_content,
                conversation_id=conversation_id
            )
            db.add(user_message)

            # Get conversation history (last 10 messages)
            messages_history = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at.desc()).limit(10).all()

            # Prepare messages for AI
            ai_messages = []

            # System message
            ai_messages.append({
                "role": "system",
                "content": "You are a helpful and honest assistant. Respond concisely and strictly based on the information provided in the current conversation and any attached or referenced content. "
                            "If you do not have enough information to accurately answer a question, respond with 'I'm not sure' or 'I don't know'. "
                            "Do not fabricate facts, do not speculate, and do not generalize beyond what the user explicitly asked. "
                            "Only elaborate or infer when the user clearly requests it. "
                            "If a user asks a specific question but no relevant information is available, reply with 'I don't know'."
            })

            # Conversation history (oldest first)
            for msg in reversed(messages_history):
                ai_messages.append({
                    "role": msg.role,
                    "content": truncate_text(msg.content, 2000)
                })

            # Current user message
            ai_messages.append({
                "role": "user",
                "content": truncate_text(full_content, 4000)
            })

            # Get AI response
            response = make_openai_request(ai_messages)
            ai_response = response.choices[0].message.content

            # Save AI response
            assistant_message = Message(
                role="assistant",
                content=ai_response,
                conversation_id=conversation_id
            )
            db.add(assistant_message)

            # Update conversation timestamp
            conversation.updated_at = func.now()
            db.commit()

            return {
                "output": ai_response,
                "file_id": file_metadata.id if file_metadata else None
            }

    except openai.APIError as e:
        db.rollback()
        raise HTTPException(
            status_code=429,
            detail=f"OpenAI API error: {str(e)}"
        )
    except (BotoCoreError, ClientError) as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Storage error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations/{conversation_id}/files")
async def get_conversation_files(
        conversation_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Get all files for a specific conversation"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation.files


@router.post("/conversations")
async def create_conversation(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    try:
        # Initialize S3 client with proper error handling
        s3_client = boto3.client(
            's3',
            region_name='eu-north-1',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        # Create conversation record
        conversation = Conversation(
            user_id=current_user.id,
            title="New Conversation",
            s3_bucket=None  # Will be set after bucket creation
        )
        db.add(conversation)
        db.flush()  # Get the ID without committing

        # Generate bucket name
        bucket_name = f"conv-{conversation.id}-{current_user.id}".lower()

        try:
            try:
                s3_client.head_bucket(Bucket=bucket_name)
                # If we get here, the bucket exists
                logger.warning(f"Bucket {bucket_name} already exists")

                # Since bucket exists, we'll use it but verify ownership
                conversation.s3_bucket = bucket_name
                db.commit()
                logger.info(f"Reusing existing bucket {bucket_name} for conversation {conversation.id}")
                return conversation

            except ClientError as e:
                # If we get a 404, bucket doesn't exist - which is what we want
                if e.response['Error']['Code'] != '404':
                    raise

            # Create S3 bucket
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': 'eu-north-1'}
            )

            # Wait for bucket to be ready
            waiter = s3_client.get_waiter('bucket_exists')
            waiter.wait(Bucket=bucket_name)

            # Update conversation with bucket name
            conversation.s3_bucket = bucket_name
            db.commit()

            logger.info(f"Created conversation {conversation.id} with bucket {bucket_name}")
            return conversation

        except (BotoCoreError, ClientError) as e:
            db.rollback()
            logger.error(f"S3 bucket creation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to create storage bucket. Please try again."
            )

    except Exception as e:
        db.rollback()
        logger.error(f"Conversation creation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to create conversation. Please try again."
        )

@router.get("/conversations")
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all conversations for current user"""
    return db.query(Conversation).filter(Conversation.user_id == current_user.id).all()

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
        conversation_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    try:
        # Get the conversation
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        ).first()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            region_name='eu-north-1',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        bucket_name = conversation.s3_bucket

        try:
            # First try to empty the bucket
            try:
                # List and delete all objects in the bucket
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket_name)

                delete_requests = []
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            delete_requests.append({'Key': obj['Key']})

                # Delete objects in batches of 1000 (S3 limit)
                for i in range(0, len(delete_requests), 1000):
                    s3_client.delete_objects(
                        Bucket=bucket_name,
                        Delete={'Objects': delete_requests[i:i + 1000]}
                    )

                logger.info(f"Deleted all objects from bucket {bucket_name}")

            except s3_client.exceptions.NoSuchBucket:
                logger.warning(f"Bucket {bucket_name} not found - may have been deleted already")
                pass

            # Then delete the bucket
            try:
                s3_client.delete_bucket(Bucket=bucket_name)
                logger.info(f"Deleted bucket {bucket_name}")
            except s3_client.exceptions.NoSuchBucket:
                logger.warning(f"Bucket {bucket_name} not found during bucket deletion")
                pass

        except Exception as s3_error:
            logger.error(f"Error cleaning up S3 bucket {bucket_name}: {str(s3_error)}")
            # We'll still proceed with DB deletion even if S3 cleanup fails

        # Delete the conversation from DB
        try:
            db.delete(conversation)
            db.commit()
            logger.info(f"Deleted conversation {conversation_id} from database")
            return {"message": "Conversation deleted successfully"}

        except Exception as db_error:
            db.rollback()
            logger.error(f"Database deletion failed for conversation {conversation_id}: {str(db_error)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to delete conversation from database"
            )

    except HTTPException:
        # Re-raise HTTP exceptions (like the 404 we raise above)
        raise

    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error deleting conversation {conversation_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to delete conversation due to an unexpected error"
        )
@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify conversation belongs to user
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get messages ordered by creation time (oldest first)
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.asc()).all()

    return [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat()
        }
        for msg in messages
    ]

@router.get("/conversations/{conversation_id}/files/{file_id}/download")
async def download_file(
        conversation_id: int,
        file_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    file = db.query(UploadedFile).filter(
        UploadedFile.id == file_id,
        UploadedFile.conversation_id == conversation_id
    ).first()

    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Get the file from the conversation's bucket
        obj = s3.Object(conversation.s3_bucket, file.s3_key)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.file_type}") as temp_file:
            obj.download_fileobj(temp_file)
            temp_file.flush()

            # Return the file
            return FileResponse(
                temp_file.name,
                filename=file.original_filename,
                media_type=f"application/{file.file_type}"
            )

    except ClientError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download file: {str(e)}"
        )
    finally:
        if 'temp_file' in locals():
            os.unlink(temp_file.name)

@router.patch("/conversations/{conversation_id}/title")
async def update_conversation_title(
    conversation_id: int,
    new_title: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation.title = new_title[:100]  # Limit title length
    conversation.updated_at = func.now()
    db.commit()

    return {"message": "Title updated successfully", "new_title": conversation.title}