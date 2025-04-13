import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/server_monitoring")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
JWT_SECRET = os.getenv("JWT_SECRET", "supersecret")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
SSH_DEFAULT_PORT = 22