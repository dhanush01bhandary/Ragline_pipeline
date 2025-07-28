import os
import fitz  # PyMuPDF
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline


class RAGEngine:
    def __init__(self, pdf_path, index_path="vectorstore/faiss_index.bin", metadata_path="vectorstore/metadata.pkl"):
        self.pdf_path = pdf_path
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        self.text_chunks = []
        self.index = None
        self.metadata = []
        self.generator = None

    def load_pdf_and_chunk(self):
        print("📄 Loading and chunking PDF...")
        doc = fitz.open(self.pdf_path)
        for page in doc:
            text = page.get_text().strip()
            paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 40]
            self.text_chunks.extend(paragraphs)
        print(f"✅ Extracted {len(self.text_chunks)} chunks.")

    def build_faiss_index(self):
        print("🔍 Creating FAISS index...")
        embeddings = self.model.encode(self.text_chunks)
        dim = embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))
        self.metadata = self.text_chunks

        # Save index and metadata
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print("✅ FAISS index and metadata saved.")

    def load_index(self):
        print("🔁 Loading FAISS index and metadata...")
        self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

    def query(self, question, top_k=3):
        if self.index is None:
            self.load_index()
        if not self.metadata:
            self.load_pdf_and_chunk()

        q_embedding = self.model.encode([question])
        D, I = self.index.search(np.array(q_embedding), top_k)

        if I is None or len(I[0]) == 0:
            return ["❌ No relevant chunks found."]

        results = []
        for i in I[0]:
            if i < len(self.metadata):
                results.append(self.metadata[i])
        return results or ["❌ No relevant chunks found."]

    def init_generator(self):
        self.generator = pipeline("text-generation", model="distilgpt2")

    def generate_answer(self, question, chunks):
        if self.generator is None:
            self.init_generator()

        context = " ".join(chunks[:2])  # Use only first 2 to avoid token limit
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        output = self.generator(prompt, max_length=150, num_return_sequences=1)
        return output[0]['generated_text'].split("Answer:")[-1].strip()
