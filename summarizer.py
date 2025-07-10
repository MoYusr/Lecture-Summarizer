import pickle
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# Initialize NLP
nlp = spacy.load("en_core_web_sm")

# 1. Training Function (NEW)
def train_model(csv_path="dataset.csv"):
    """Train TF-IDF on your dataset.csv"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    lectures = df['text'].tolist()  # Assuming column named 'text'
    
    all_sentences = []
    for lecture in lectures:
        doc = nlp(lecture)
        all_sentences.extend([sent.text for sent in doc.sents])
    
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(all_sentences)
    
    # Save model
    pickle.dump(vectorizer, open("models/tfidf_vectorizer.pkl", "wb"))
    return vectorizer

# 2. Summarization Function (Modified)
def summarize(text, vectorizer=None, top_n=3):
    """Generate summary using pre-trained vectorizer"""
    if vectorizer is None:
        # Load if not provided
        vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    # Transform using pre-trained TF-IDF
    tfidf_matrix = vectorizer.transform(sentences)
    
    # TextRank algorithm
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)
    
    scores = np.ones(len(sentences))
    for _ in range(20):
        scores = 0.85 * np.dot(sim_matrix, scores) + 0.15
    
    top_indices = scores.argsort()[-top_n:][::-1]
    return [sentences[i] for i in sorted(top_indices)]

# 3. Evaluation (Keep existing)
def evaluate_summary(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1'])
    return scorer.score(reference, " ".join(generated))

def check_model_exists():
    """Check if trained model exists"""
    import os
    return os.path.exists("models/tfidf_vectorizer.pkl")