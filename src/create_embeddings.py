from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_vectorstore(texts, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore
