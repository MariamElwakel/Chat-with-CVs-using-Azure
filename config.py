import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from qdrant_client import QdrantClient

import cohere

# ENV 
load_dotenv()

# Qdrant for vector store
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "hr_cv_collection"
EMBEDDING_DIM = 1536  # text-embedding-3-small

# Azure for LLM and Embedding models
AZURE_KEY = os.getenv("AZURE_KEY")

# Cohere for reranking
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  


# Embedding
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint="https://gbgacademy-genai-4.openai.azure.com/",
    azure_deployment="text-embedding-3-small",
    api_version="2024-12-01-preview",
    api_key=AZURE_KEY,
)


# LLM (gpt-4.1-nano)
llm = AzureChatOpenAI(
    azure_endpoint="https://gbgacademy-genai-4.openai.azure.com/",
    azure_deployment="gpt-4.1-nano",
    api_version="2024-12-01-preview",
    api_key=AZURE_KEY,
    temperature=0,
)


# Qdrant Cloud Setup
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=300
)


# Cohere client
cohere_client = cohere.Client(COHERE_API_KEY)