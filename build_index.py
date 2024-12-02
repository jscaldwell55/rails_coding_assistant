import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Load documents
def load_documents(data_dir):
    doc_chunks = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    relative_path = os.path.relpath(file_path, data_dir)
                    doc_chunks.append({
                        "source": relative_path,
                        "content": content
                    })
    return doc_chunks

data_directory = "data/data"  # Update path if necessary
doc_chunks = load_documents(data_directory)

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [chunk['content'] for chunk in doc_chunks]
embeddings = model.encode(texts)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and metadata
with open('rails_index.pkl', 'wb') as f:
    pickle.dump((index, doc_chunks), f)

print(f"FAISS index and metadata successfully saved to 'rails_index.pkl'.")
