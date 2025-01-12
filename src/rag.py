"""
Build and querying the RAG model. 
"""
from langchain_groq import ChatGroq
import chromadb
from utils import get_embedding_function, get_api_key

def response(model, query):
    response = model.invoke([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{query}"},
        ])
    return response.content


def rag_response(model, query, context):
    response = model.invoke([
            {"role": "system", "content": "You are a helpful assistant. You can use the context provided to answer."},
            {"role": "user", "content": f"query: {query}. context: {context}"},
        ])
    return response.content


def get_rag_context(query, client, num_docs=3):
    collection = client.get_collection(name="reuters_collection", embedding_function=get_embedding_function())
    results = collection.query(
        query_texts=[query],
        n_results=num_docs
    )
    contexts = [doc.replace("\n", " ") for doc in results['documents'][0]]
    return contexts


def main():
    client = chromadb.PersistentClient(path="../chromadb/test_db")

    model = ChatGroq(model="llama3-8b-8192", api_key=get_api_key())

    query = "When do farmers sow sugar beet in Holland?"
    contexts = get_rag_context(query, client)
    default_response = response(model, query)
    ragged_response = rag_response(model, query, ";".join(contexts))
    print(f"Query: {query}")
    print(f"Default response: {default_response}")
    print(f"RAG response: {ragged_response}")
    print("\n")

    query = "Name a finance minister of West Germany."
    contexts = get_rag_context(query, client)
    default_response = response(model, query)
    ragged_response = rag_response(model, query, ";".join(contexts))
    print(f"Query: {query}")
    print(f"Default response: {default_response}")
    print(f"RAG response: {ragged_response}")
    print("\n")

    query = "What was the inflation rate in Indonesia in 1986?"
    contexts = get_rag_context(query, client)
    default_response = response(model, query)
    ragged_response = rag_response(model, query, ";".join(contexts))
    print(f"Query: {query}")
    print(f"Default response: {default_response}")
    print(f"RAG response: {ragged_response}")
    print("\n")


if __name__ == "__main__":
    main()