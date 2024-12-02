import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Directory containing the data files
data_folder = 'data'

# Prepare lists to hold the documents and their corresponding IDs
documents = []
doc_ids = []

# Read each text file in the data directory and subdirectories
for root, _, files in os.walk(data_folder):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                content = f.read()
                documents.append(content)
                # Use the relative file path as the document ID
                doc_ids.append(os.path.relpath(file_path, data_folder))

# Generate embeddings for the documents
embeddings = model.encode(documents)

# Create a FAISS index for the embeddings
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance index
index.add(embeddings)  # Add embeddings to the index

# Save the index and document IDs to a pickle file
with open('rails_index.pkl', 'wb') as f:
    pickle.dump((index, doc_ids), f)

print(f"FAISS index built and saved to 'rails_index.pkl' with {len(documents)} documents.")
