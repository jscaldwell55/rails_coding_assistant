import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import openai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
    raise ValueError("OpenAI API key not found.")

openai.api_key = api_key

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
        query_embedding = model.encode([query])
        k = 3  # Number of results to retrieve
        D, I = index.search(query_embedding, k)  # Updated FAISS search syntax

        results = []
        for idx in I[0]:
            if idx == -1:
                continue
            chunk = doc_chunks[idx]
            results.append(f"Source: {chunk['source']}\n{chunk['content']}")

        return "\n\n".join(results) if results else "No relevant documentation found."
    except Exception as e:
        logger.error(f"Error in RAG search: {str(e)}")
        return "An error occurred during the RAG search."


# Function to generate GPT response
def fallback_gpt(query: str) -> str:
    """
    Use OpenAI GPT as the base response.

    Args:
        query: The user's query string.

    Returns:
        GPT's response as a string.
    """
    try:
        client = openai.Client(api_key=api_key)  # Create client instance
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Rails coding assistant."},
                {"role": "user", "content": query}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return f"An error occurred with the GPT API: {str(e)}"

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
