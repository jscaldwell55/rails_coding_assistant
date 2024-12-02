import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import os
path_to_index = os.path.join(os.getcwd(), 'rails_index.pkl')
with open(path_to_index, 'rb') as f:
    index, doc_chunks = pickle.load(f)

import openai  # Corrected import for OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
    raise ValueError("OpenAI API key not found.")

openai.api_key = api_key  # Set the API key directly for OpenAI

# Initialize SentenceTransformer for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and document metadata
def load_faiss_index():
    """Load FAISS index and document metadata."""
    try:
        with open('rails_index.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.error("FAISS index file 'rails_index.pkl' not found")
        raise FileNotFoundError("The FAISS index file 'rails_index.pkl' was not found.")

index, doc_chunks = load_faiss_index()

# Function to search the RAG system
def search_rag(query: str) -> str:
    """
    Search the RAG system for relevant snippets.

    Args:
        query: The search query string.

    Returns:
        Relevant snippets from the data files.
    """
    try:
        # Create the query embedding
        query_embedding = model.encode([query])
        k = 3  # Number of results to retrieve
        distances, indices = index.search(query_embedding, k)

        # Retrieve the document chunks based on indices
        results = []
        for idx in indices[0]:
            if idx == -1:  # Skip invalid indices
                continue
            try:
                chunk = doc_chunks[idx]
                results.append(f"Source: {chunk['source']}\n{chunk['content']}")
            except IndexError:
                logger.error(f"Index {idx} is out of bounds for document chunks.")
                continue

        # Return concatenated results
        return "\n\n".join(results) if results else "No relevant documentation found."
    except Exception as e:
        logger.error(f"Error in RAG search: {str(e)}")
        return f"An error occurred during the RAG search: {str(e)}"


# Function to generate GPT response
def load_faiss_index():
    """Load FAISS index and document metadata."""
    try:
        with open('rails_index.pkl', 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        st.error("""
            FAISS index file 'rails_index.pkl' not found or corrupted. 
            Please ensure you have:
            1. Generated the index file
            2. Placed it in the correct directory
            3. Have proper read permissions
        """)
        logger.error(f"FAISS index error: {str(e)}")
        # Return empty/default values
        dimension = 384  # dimension for 'all-MiniLM-L6-v2' model
        empty_index = faiss.IndexFlatL2(dimension)
        return empty_index, []

# Modified main code to handle missing index
try:
    index, doc_chunks = load_faiss_index()
    if len(doc_chunks) == 0:
        st.warning("Running in limited mode: RAG features are disabled due to missing index file.")
except Exception as e:
    logger.error(f"Error initializing FAISS index: {str(e)}")
    st.error("Error initializing search index. Running in GPT-only mode.")
    index = None
    doc_chunks = []

# Modified search_rag function to handle missing index
def search_rag(query: str) -> str:
    """
    Search the RAG system for relevant snippets.

    Args:
        query: The search query string.

    Returns:
        Relevant snippets from the data files.
    """
    if index is None or len(doc_chunks) == 0:
        return "RAG search is currently unavailable. Running in GPT-only mode."
        
    try:
        query_embedding = model.encode([query])
        k = 3  # Number of results to retrieve
        D, I = index.search(query_embedding, k)

        results = []
        for idx in I[0]:
            if idx == -1 or idx >= len(doc_chunks):
                continue
            chunk = doc_chunks[idx]
            results.append(f"Source: {chunk['source']}\n{chunk['content']}")

        return "\n\n".join(results) if results else "No relevant documentation found."
    except Exception as e:
        logger.error(f"Error in RAG search: {str(e)}")
        return "An error occurred during the RAG search."
        
# Function to combine GPT and RAG results
def augment_with_rag(gpt_response: str, rag_content: str) -> str:
    """
    Combine GPT response with RAG content.

    Args:
        gpt_response: The original GPT response.
        rag_content: The content retrieved from the RAG system.

    Returns:
        A single augmented response.
    """
    if rag_content and rag_content != "No relevant documentation found.":
        return f"{gpt_response}\n\n---\n\nRelevant Documentation:\n{rag_content}"
    return gpt_response

# Streamlit App UI
st.title("Rails Coding Assistant")

st.write("Ask a coding question and get GPT responses augmented with relevant Rails documentation.")

query = st.text_area("Your Query:", placeholder="Enter your coding question here...")

categories = [
    "None (Use GPT only)",
    "ActiveRecord & Database Interactions",
    "Routing & RESTful APIs",
    "Controller Logic & Actions",
    "View and Template Helpers",
    "Testing (RSpec, Minitest)",
    "Gems & Integrations",
    "Debugging & Error Handling",
    "Performance Optimization",
    "Rails Environments & Configurations",
    "Frontend Integration",
]

selected_category = st.selectbox("Choose a RAG Category (Optional):", categories)

if st.button("Submit"):
    if not query.strip():
        st.error("Please enter a query.")
    else:
        # Base GPT response
        gpt_response = fallback_gpt(query)

        # RAG augmentation if a category is selected
        rag_content = None
        if selected_category != "None (Use GPT only)":
            rag_content = search_rag(query)

        # Combine GPT and RAG
        final_response = augment_with_rag(gpt_response, rag_content)
        st.subheader("Response:")
        st.text_area("Output:", value=final_response, height=300)
