from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

def getFileEmbeddings(file_keywords: str):
    return model.encode(file_keywords, convert_to_numpy=True)
