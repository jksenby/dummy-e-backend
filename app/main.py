from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.auth import router as auth_router
from app.dummy import router as dummy_router
from app.database import Base, engine

app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создание таблиц при запуске приложения
Base.metadata.create_all(bind=engine)

app.include_router(auth_router, prefix="/auth")
app.include_router(dummy_router, prefix="/requests")

@app.get("/")
def root():
    return {"message": "Server Monitoring API"}