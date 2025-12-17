import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load OpenAI key
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
Be concise and direct.
If no strong evidence, say so honestly.

Context:
{context}

Query: {query}
Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)

    # Modern RAG chain
    chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(query)

    # Get source documents separately for citations
    docs = retriever.invoke(query)
    citations = "\n\n(Note: Suggestions are based on RCT evidence from PubMed abstracts.)"

return response + citations
