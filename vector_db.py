from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pandas as pd

def build_vector_db():
    print("Loading microbiome abstracts...")
    df = pd.read_csv("microbiome_abstracts.csv")
    texts = df['abstract'].tolist()
    
    print(f"Chunking {len(texts)} abstracts...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # Create Document objects
    docs = [Document(page_content=text) for text in texts]
    chunks = splitter.split_documents(docs)
    
    print(f"Created {len(chunks)} chunks. Building embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Building FAISS index (this may take 1-2 minutes)...")
    db = FAISS.from_documents(chunks, embeddings)
    
    db.save_local("faiss_index")
    print("FAISS index saved to faiss_index/ — ready for RAG queries!")
    
    return db

if __name__ == "__main__":
    build_vector_db()