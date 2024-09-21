from langchain_community.embeddings import GooglePalmEmbeddings
import os

# Load API key from environment
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize embeddings
embeddings = GooglePalmEmbeddings()

# Sample text embedding
result = embeddings.embed_documents(["Test document for embedding"])
print(result)
