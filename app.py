from fastapi import FastAPI
from model import get_response
app = FastAPI()

@app.get("/chat")
def chat(query:str):
    response = get_response(query.lower())
    return {"response":response}