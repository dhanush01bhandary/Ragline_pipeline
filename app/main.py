from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.rag_engine import RAGEngine
import os

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

engine = RAGEngine("app/docs/sample.pdf")

# ✅ Build index only once, reuse on next runs
if not os.path.exists(engine.index_path):
    engine.load_pdf_and_chunk()
    engine.build_faiss_index()
else:
    print("🔁 Loading existing FAISS index and metadata...")
    engine.load_pdf_and_chunk()
    engine.load_index()

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, question: str = Form(...)):
    try:
        chunks = engine.query(question)
        if "❌" in chunks[0]:
            final_answer = chunks[0]
        else:
            final_answer = engine.generate_answer(question, chunks)
    except Exception as e:
        final_answer = f"❌ Error: {str(e)}"

    return templates.TemplateResponse("form.html", {
        "request": request,
        "answer": final_answer
    })
