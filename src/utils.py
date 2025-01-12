"""
Utility functions for the project.
"""
import json
from chromadb.utils import embedding_functions


def get_api_key():
    credentials_path = "../misc/credentials.json"
    with open(credentials_path, "r") as f:
        credentials = json.load(f)
    return credentials["groq_api"]["api_key"]


def get_embedding_function():
    ef = embedding_functions.DefaultEmbeddingFunction()
    return ef


def embed_text(text, ef):
    return ef(text)

