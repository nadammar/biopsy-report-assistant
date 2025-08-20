from sentence_transformers import SentenceTransformer
import numpy as np

# Load the pre-trained model for embedding generation
def get_embedding(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the SentenceTransformer model
    embedding = model.encode(text)  # Generate the embedding for the input text
    
    # Convert to numpy array if it isn't already
    embedding = np.array(embedding)

    # Ensure it's a 1D numpy array of the correct length (384 dimensions)
    if embedding.ndim != 1:
        raise ValueError(f"Embedding should be a 1D array, got {embedding.ndim}D instead.")
    
    # Return as a list instead of a numpy array for compatibility with PostgreSQL
    return embedding.tolist()
