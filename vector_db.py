from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pandas as pd

def build_vector_db():
    df = pd.read_csv("microbiome_abstracts.csv")
    texts = df['abstract'].tolist()
    titles = df['title'].fillna("Untitled Study").tolist()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    docs = [Document(page_content=text, metadata={"title": titles[i]}) for i, text in enumerate(texts)]
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")
    print("FAISS index rebuilt with titles — citations will be clean!")
    
    return db

if __name__ == "__main__":
    build_vector_db()
