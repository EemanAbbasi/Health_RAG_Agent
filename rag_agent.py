import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load key from Streamlit secrets or local .env
try:
    import streamlit as st
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0.3, api_key=api_key)

def get_rag_response(query):
    # Load FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # Prompt template
    template = """
You are a microbiome therapeutics expert. Use ONLY the provided RCT abstracts as evidence.
Suggest natural, herb-based remedies (ginger, peppermint, turmeric, fennel, chamomile) and probiotics/prebiotics when relevant.
Explain WHY they may help based on the evidence.
Include brief citations (first 20 words of abstract).
If no strong evidence, say so honestly.
Context:
{context}

Query: {query}
Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # Modern chain (replaces RetrievalQA)
    chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(query)
    
    citations = "\n\n**Key References (from RCT abstracts):**\n"
    seen = set()
    for i, doc in enumerate(result["source_documents"][:4], 1):
        text = doc.page_content.strip()
        if text in seen:
            continue
        seen.add(text)
        # Clean first sentence
        first_sentence = text.split('.')[0]
        if first_sentence:
            first_sentence += '.'  # Add period back
        else:
            first_sentence = text[:120] + "..."
        citations += f"• {first_sentence}\n"
    
    return response + citations
