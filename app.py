import streamlit as st
from rag_agent import get_rag_response

st.set_page_config(page_title="Microbiome RAG Agent", page_icon="🧬", layout="wide")

st.title("🧬 Microbiome RAG Agent")
st.markdown("""
Ask for **evidence-based natural remedies** from 723 microbiome-focused RCT abstracts.

Examples:
- Herbal remedies for gut dysbiosis
- Probiotics for inflammation
- Natural ways to improve gut barrier function
""")

query = st.text_input("Your query:", placeholder="e.g., natural remedies for IBS or leaky gut")
if query:
    with st.spinner("Searching RCT literature and generating evidence-based answer..."):
        response = get_rag_response(query)
    st.markdown("### Answer")
    st.markdown(response)

st.caption("Built with LangChain, OpenAI, Sentence-Transformers, FAISS • Data: PubMed 20k RCT (microbiome-filtered)")