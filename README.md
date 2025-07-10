Run this the first time: python -c "from summarizer import train_model; train_model()".

Text is tokenized (broken down into smaller units called tokens).
Uses TF-IDF to check how important a word is to the lecture transcript.
Term Frequency (TF):
This measures how often a term appears in a document. A higher term frequency means the word is more relevant to that specific document.
Inverse Document Frequency (IDF):
This measures how common or rare a term is across the entire corpus. Words that appear in many documents have a low IDF, while words that appear in only a few documents have a high IDF. 
