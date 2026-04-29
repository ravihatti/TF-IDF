from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data import qa_pairs
question  = [item["q"] for item in qa_pairs]
answers = [item["a"] for item in qa_pairs]
vectorizers = TfidfVectorizer()
X = vectorizers.fit_transform(question)

def get_response(user_query):
    query_vec = vectorizers.transform([user_query])
    similarity = cosine_similarity(query_vec,X)
    idx = similarity.argmax()
    score = similarity[0][idx]
    if score < 0.3:
        return "sorry i didn't understand"
    return answers[idx]
