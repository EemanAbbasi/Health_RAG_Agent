import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# Load .env file
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file! Add it with your real key.")

def get_rag_response(query):
    # Load the FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Retriever
    retriever = db.as_retriever(search_kwargs={"k": 10})
    
    # LLM with your key
    llm = OpenAI(temperature=0.3, api_key=api_key)
    
    # RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Herb-focused prompt
    prompt = f"""
You are a microbiome therapeutics expert specializing in natural remedies.
Use ONLY the provided RCT abstracts as evidence.
Prioritize herb-based remedies (ginger, peppermint, turmeric, fennel, chamomile, licorice root, aloe vera) and explain their anti-inflammatory or gut-modulating effects.
Also mention probiotics/prebiotics if strongly supported.
Structure your answer:
1. Top 2-3 recommended natural remedies
2. Why they help (based on evidence)
3. Any caveats
Be concise, avoid repetition, and cite key findings.
If evidence is weak, say so.
Query: {query}
"""
    
    result = qa_chain.invoke({"query": prompt})
    response = result["result"]
    response = response.split("Sources")[0].strip()  # Remove default sources (we add better ones)
    if response.endswith("For example, a study on"):
        response = response[:-len("For example, a study on")]
    
    # Add citations
    citations = "\n\n**Key References from RCTs:**\n"
    # Clean sources: only paper titles, no abstract text

    seen_titles = set()
    for i, doc in enumerate(result["source_documents"][:4], 1):  # Top 4 sources
        title = doc.metadata.get("title", "Untitled RCT Study").strip()
        if title in seen_titles or not title:
            continue
        seen_titles.add(title)
        citations += f"• {title}\n"
    return response + citations

# Test
if __name__ == "__main__":
    test_query = "Natural remedies for gut inflammation or dysbiosis"
    print("Query:", test_query)
    print("\nResponse:")
    print(get_rag_response(test_query))