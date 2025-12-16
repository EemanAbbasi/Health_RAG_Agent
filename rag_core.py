from langchain.embeddings import HuggingFaceEmbeddings  # Uses PyTorch
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def build_vector_db():
    df = pd.read_csv("microbiome_abstracts.csv")
    texts = df['abstract'].tolist()
    
    # Split long texts
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_texts(texts)
    
    # Embeddings (PyTorch-based)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Or TensorFlow: use tensorflow-hub
    
    # Build FAISS index
    vector_db = FAISS.from_texts(chunks, embeddings)
    vector_db.save_local("faiss_index")
    return vector_db

# Test
if __name__ == "__main__":
    db = build_vector_db()
    results = db.similarity_search("gut inflammation remedies")
    print(results[0].page_content)



def retrieve_docs(db, query, k=10):
    # Dense retrieval (semantic)
    dense_docs = db.similarity_search(query, k=k*2)  # Get more for re-ranking
    
    # Sparse retrieval (keyword - TF-IDF)
    vectorizer = TfidfVectorizer()
    all_texts = [doc.page_content for doc in dense_docs]
    tfidf_matrix = vectorizer.fit_transform(all_texts + [query])
    sparse_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    # Re-rank with XGBoost (train a simple model on features)
    features = pd.DataFrame({
        'dense_score': [doc.metadata.get('score', 0) for doc in dense_docs],  # Add scores if needed
        'sparse_score': sparse_scores
    })
    model = xgb.XGBRanker()  # Simple ranker
    model.fit(features, range(len(features)))  # Pseudo-training; fine-tune with data later
    ranks = model.predict(features)
    
    # Sort and select top k
    sorted_indices = ranks.argsort()[::-1][:k]
    top_docs = [dense_docs[i] for i in sorted_indices]
    return top_docs


def rag_query(db, query):
    docs = retrieve_docs(db, query)
    context = "\n".join(doc.page_content for doc in docs)
    
    llm = OpenAI(temperature=0.7)
    prompt = f"Based on this literature: {context}\nQuery: {query}\nSuggest microbiome therapeutics:"
    
    chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=lambda q: docs)
    response = chain.run(prompt)
    return response