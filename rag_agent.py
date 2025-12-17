import os
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load OpenAI key — Streamlit Cloud uses secrets, local can use .env
try:
    import streamlit as st
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    # Local fallback
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in Streamlit secrets or .env")

def get_rag_response(query):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    llm = OpenAI(temperature=0.3, api_key=api_key)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    prompt = f"""
You are a microbiome therapeutics expert. Use ONLY the provided RCT abstracts as evidence.
Suggest natural, herb-based remedies (e.g., ginger, peppermint, turmeric, fennel, chamomile) and probiotics/prebiotics when relevant.
Explain WHY they may help based on the evidence.
Include brief citations (first 20 words of abstract).
If no strong evidence, say so.
Query: {query}
"""
    
    result = qa_chain.invoke({"query": prompt})
    response = result["result"]
    
    # Clean references: first sentence only
    citations = "\n\n**References:**\n"
    seen = set()
    for i, doc in enumerate(result["source_documents"][:4], 1):
        text = doc.page_content.strip()
        if text in seen:
            continue
        seen.add(text)
        first_sentence = text.split('.')[0] + '.' if '.' in text else text[:120] + "..."
        citations += f"• {first_sentence}\n"
    
    return response + citations
