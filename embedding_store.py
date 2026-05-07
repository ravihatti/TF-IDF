from sentence_transformers import  SentenceTransformer
# from transformers import pipeline
import faiss 
import numpy as np
model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = []
index = None
def create_vector_store(text_chunks):
    global chunks,index
    chunks = text_chunks
    embedding = model.encode(chunks)
    embedding = np.array(embedding).astype("float32")
    dimension = embedding.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding)

def semantic_search(query:str ,top_k:int = 3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distance,ids    = index.search(query_embedding,top_k)
    result = []
    for idx in ids[0]:
        if idx < len(chunks):
            result.append(chunks[idx])
    return result 
    
