import os
import docx
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

class DocumentQuerySystem:
    def __init__(self, file_paths: List[str], embedding_model_name: str = 'all-MiniLM-L6-v2'):
        os.makedirs("faiss_index", exist_ok=True)
        self.file_paths = file_paths
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.embedding_model = None
        self.llm = None
        try:
            self.llm = ChatGroq(
                groq_api_key=os.getenv('GROQ_API_KEY', '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'),
                model_name='mixtral-8x7b-32768'
            )
        except Exception as e:
            print(f"Warning: Could not initialize Groq LLM. Fallback mode enabled. {e}")
        self.index = None
        self.text_chunks = []
        self.create_faiss_index()

    def load_document(self, file_path: str) -> str:
        text = ""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.docx':
                doc = docx.Document(file_path)
                for para in doc.paragraphs:
                    if para.text:
                        text += para.text + "\n"
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
                text = df.astype(str).to_string()
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                print(f"Unsupported file type: {file_extension}")
                return ""
            return text
        except Exception as e:
            print(f"Error reading document {file_path}: {e}")
            return ""

    def create_faiss_index(self):
        if not self.embedding_model:
            print("Embedding model not available. Cannot create index.")
            return
        all_texts = []
        text_chunks = []
        for file_path in self.file_paths:
            doc_text = self.load_document(file_path)
            if doc_text.strip():
                chunks = self.split_text(doc_text)
                text_chunks.extend(chunks)
                all_texts.extend(chunks)
        if not all_texts:
            print("No text extracted from documents. Cannot create FAISS index.")
            return
        try:
            embeddings = self.generate_embeddings(all_texts)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            faiss.write_index(index, "faiss_index/document_index.faiss")
            with open("faiss_index/text_chunks.txt", "w", encoding='utf-8') as f:
                for chunk in text_chunks:
                    f.write(chunk + "\n---\n")
            self.text_chunks = text_chunks
            self.index = index
            print(f"Successfully created FAISS index with {len(text_chunks)} text chunks.")
        except Exception as e:
            print(f"Error creating FAISS index: {e}")

    def split_text(self, text: str, max_chunk_size: int = 500) -> List[str]:
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            sentence_words = sentence.split()
            if current_length + len(sentence_words) > max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence + '.')
            current_length += len(sentence_words)
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise

    def query_documents(self, query: str) -> str:
        if not self.index or not self.embedding_model:
            return "Document indexing system is not fully initialized. Please check your setup."
        try:
            query_embedding = self.generate_embeddings([query])
            k = 5
            D, I = self.index.search(query_embedding, k)
            context_chunks = [self.text_chunks[i] for i in I[0]]
            context = "\n\n".join(context_chunks)
            if not self.llm:
                return self._local_response_generation(query, context)
            prompt = (
                f"Context from documents:\n{context}\n\n"
                f"Query: {query}\n\n"
                "Carefully analyze the provided context. Generate a comprehensive, "
                "conversational answer. If the exact information isn't available, "
                "provide the most relevant information with a helpful tone."
            )
            response = self.llm.invoke(prompt)
            return response.content.strip() if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"Query processing error: {e}")
            return self._local_response_generation(query, "")

    def _local_response_generation(self, query: str, context: str) -> str:
        if context:
            tokens = query.lower().split()
            context_words = context.lower().split()
            matches = sum(1 for token in tokens if token in context_words)
            relevance_score = matches / len(tokens) if tokens else 0
            if relevance_score > 0.3:
                sentences = context.split('. ')
                relevant_sentences = [
                    sent for sent in sentences 
                    if any(token.lower() in sent.lower() for token in tokens)
                ]
                return ". ".join(relevant_sentences[:3]) + "." if relevant_sentences else context[:500]
        return "I couldn't find a specific answer to your query in the documents."

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_documents(request: QueryRequest):
    query = request.query
    try:
        answer = doc_system.query_documents(query)
        return {"query": query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

# Initialize the DocumentQuerySystem with file paths
file_paths = ["Mental health data.docx"]  # Replace with your file paths
doc_system = DocumentQuerySystem(file_paths)