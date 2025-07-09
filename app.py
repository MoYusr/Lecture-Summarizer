import streamlit as st
from summarizer import textrank, preprocess

st.title("Lecture Summarizer")
input_text = st.text_area("Paste lecture transcript:", height=300)

if st.button("Summarize"):
    sentences = preprocess(input_text)
    summary = textrank(sentences, top_n=3)
    st.subheader("Summary:")
    st.write(" ".join(summary))
