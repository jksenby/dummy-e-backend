from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any

class Token(BaseModel):
    access_token: str
    token_type: str

class UserBase(BaseModel):
    email: str = Field(..., example="user@example.com")

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, example="securepassword")

class UserResponse(UserBase):
    id: int
    is_active: bool
    
    class Config:
        orm_mode = True

class LoginRequest(BaseModel):
    email: str
    password: str

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)

class FileResponse(BaseModel):
    temp_file: str
    filename: str
    media_type: str