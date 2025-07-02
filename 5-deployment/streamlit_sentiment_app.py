import streamlit as st
import torch
from transformers import pipeline

st.title("RoBERTa Question Answering")
st.markdown("Ask questions about any given context using RoBERTa")

@st.cache_resource
def load_qa_model():
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        device=0 if torch.cuda.is_available() else -1
    )
qa_pipeline = load_qa_model()

context = st.text_area("**Context**", 
                      height=200,
                      placeholder="Paste your text here...")

question = st.text_input("**Question**",
                        placeholder="What would you like to ask?")

if context and question:
    with st.spinner("Analyzing..."):
        try:
            result = qa_pipeline(question=question, context=context)
            
            st.success("**Answer:** " + result['answer'])
            st.metric("Confidence", f"{result['score']:.1%}")
            
            # Show context highlight
            st.markdown("**Relevant Context:**")
            start = max(0, result['start'] - 20)
            end = min(len(context), result['end'] + 20)
            st.markdown(f"...{context[start:end]}...")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

with st.expander("💡 Example"):
    st.markdown("""
    **Context:**  
    The Eiffel Tower is a wrought-iron lattice tower in Paris. It was designed by Gustave Eiffel for the 1889 Exposition Universelle.
    
    **Question:**  
    Who designed the Eiffel Tower?
    
    **Expected Answer:**  
    Gustave Eiffel
    """)
