# Lecture Summarizer

An AI tool that extracts key ideas from lecture transcripts using NLP.

# First-Time Setup

## Install requirements
pip install -r requirements.txt

## Train the model (creates models/tfidf_vectorizer.pkl)
python -c "from summarizer import train_model; train_model()"

# How It Works
## Tokenization
  Breaks lectures into individual sentences using spaCy's NLP.

## TF-IDF Analysis

  Term Frequency (TF): How often a word appears in this lecture

  Inverse Document Frequency (IDF): How rare the word is across all lectures
   (Example: "backpropagation" scores high if unique to CS lectures)

## TextRank Algorithm

  Builds a "sentence network" using cosine similarity
  Scores sentences by their connections (like Google's PageRank)
  Selects top 3 most important sentences
