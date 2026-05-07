from fastapi import FastAPI, UploadFile,File,Form,Request
import os
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from injestion import extract_text
from preprocessing import clean_text,chunk_text
from embedding_store import create_vector_store
from rag_engine import generate_answer
from model import get_response
templates = Jinja2Templates(directory="template")
app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR,exist_ok=True)
@app.get("/",response_class=HTMLResponse)
def home(request:Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    raw_text = extract_text(file_path)
    clean = clean_text(raw_text)
    chunks = chunk_text(clean)

    create_vector_store(chunks)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "message": "File uploaded, preprocessed, embedded and indexed successfully."
        }
    )

@app.post("/ask")
def ask(request: Request, question: str = Form(...)):
    answer = generate_answer(question)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "question": question,
            "answer": answer
        }
    )

# @app.get("/chat")
# def chat(query:str):
#     response = get_response(query.lower())
#     return {"response":response}
@app.get("/chat")
def chat(query):
    answer = get_response(query)
    return {"query":query,"answer":answer}
