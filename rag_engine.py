# from sentence_transformers import  SentenceTransformer
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from embedding_store import semantic_search
# import faiss 
# import numpy as np
# from data import documents
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# doc_embedding = embed_model.encode(documents)
# doc_embedding = np.array(doc_embedding).astype("float32")
# dem = doc_embedding.shape[1]
# index = faiss.IndexFlatL2(dem)
# index.add(doc_embedding)

model="google/flan-t5-small"
token = AutoTokenizer.from_pretrained(model)
model = AutoModelForSeq2SeqLM.from_pretrained(model)

def generate_answer(question: str):
    retrieved_chunks = semantic_search(question)
    context = "\n".join(retrieved_chunks)

    prompt = f"""
Answer only from the given context.

Context:
{context}

Question:
{question}

Answer:
"""
    inputs = token(prompt, return_tensors="pt")  # ✅ tensor
    response = model.generate(**inputs, max_length=200, num_beams=3)
    return token.decode(response[0],skip_special_tokens = True)