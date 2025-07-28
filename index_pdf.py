from app.rag_engine import RAGEngine


if __name__ == "__main__":
    engine = RAGEngine("app/docs/sample.pdf")
    engine.load_pdf_and_chunk()
    engine.build_faiss_index()
