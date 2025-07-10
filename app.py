import streamlit as st
from summarizer import check_model_exists, summarize
if not check_model_exists():
    st.warning("No trained model found. Using basic summarizer. Train model for better results!")
st.title("Lecture Summarizer")
st.caption("Now with trained TF-IDF model for better summaries!")

input_text = st.text_area("Paste lecture transcript:", height=300)

if st.button("Summarize"):
    # New version - uses pre-trained model automatically
    summary = summarize(input_text)  # Single function call
    st.subheader("Summary:")
    st.write(" ".join(summary))
    
    # Optional: Add evaluation if reference is provided
    if st.checkbox("Show advanced options"):
        reference = st.text_area("Paste reference summary (for evaluation):", height=100)
        if reference:
            from summarizer import evaluate_summary
            scores = evaluate_summary(summary, reference)
            st.metric("ROUGE-1 Score", f"{scores['rouge1'].fmeasure:.2f}")