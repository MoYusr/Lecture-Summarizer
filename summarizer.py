import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# 1. Load lecture text
def load_lecture(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# 2. Preprocessing
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text) > 10]

# 3. TextRank Algorithm
def textrank(sentences, top_n=3):
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)  # Remove self-similarity
    
    # Sentence ranking (simplified PageRank)
    scores = np.ones(len(sentences))
    for _ in range(20):
        scores = 0.85 * np.dot(sim_matrix, scores) + 0.15
    
    # Get top indices
    top_indices = scores.argsort()[-top_n:][::-1]
    return [sentences[i] for i in sorted(top_indices)]  # Maintain original order

# 4. Evaluation
def evaluate_summary(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, generated)

# 5. Main Workflow
if __name__ == "__main__":
    # Load data
    lecture = load_lecture("test_lecture.txt")
    reference = load_lecture("reference_summary.txt")
    
    # Generate summary
    sentences = preprocess(lecture)
    summary = textrank(sentences, top_n=3)
    generated_summary = " ".join(summary)
    
    # Evaluate
    print("Generated Summary:")
    print(generated_summary)
    print("\nEvaluation Scores:")
    scores = evaluate_summary(generated_summary, reference)
    for key in scores:
        print(f"{key}: {scores[key].fmeasure:.3f}")
