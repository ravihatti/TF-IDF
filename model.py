from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data import qa_pairs

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# load model 
model = SentenceTransformer("all-MiniLM-L6-v2")
# prepration of data
question  = [item["q"] for item in qa_pairs]
answers = [item["a"] for item in qa_pairs]
# vectorize
# vectorizers = TfidfVectorizer()
# X = vectorizers.fit_transform(question)
embedding  = model.encode(question)

# FAISS index
dim = embedding.shape[1]
index  = faiss.IndexFlatL2(dim)
index.add(np.array(embedding))


def get_response(user_query):
    # IT-IDF
    # query_vec = vectorizers.transform([user_query])
    # similarity = cosine_similarity(query_vec,X)
    # idx = similarity.argmax()
    # score = similarity[0][idx]
    # if score < 0.3:
    #     return "sorry i didn't understand"
    # return answers[idx]
    query_vec = model.encode([user_query])
    D,I = index.search(np.array(query_vec),k=1)
    idx  = I[0][0]
    score = D[0][0]
    if score > 1.0:
        return "Sorry , I didn't understand"
    return answers[idx]
