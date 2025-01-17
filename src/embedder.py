"""
Embeds documents to a VectorDB with OpenAI API
"""
import chromadb
import nltk
from nltk.corpus import reuters
from utils import get_embedding_function


def main(nltk_download=True):
    if nltk_download:
        nltk.download('reuters')
        nltk.download('punkt')
   
    reuters_subset = reuters.fileids()[0:100]
    reuters_subset = [id for id in reuters_subset if len(reuters.words(id)) < 500]
    
    client = chromadb.PersistentClient(path="../chromadb/test_db2")
    collection = client.create_collection(name="reuters_collection", embedding_function=get_embedding_function())

    for i, file_id in enumerate(reuters_subset):
        collection.add(
            documents=[reuters.raw(file_id)],
            metadatas=[{"nltk_file_id": file_id}],
            ids=[str(i)]
        )
    
if __name__ == "__main__":
    main()

