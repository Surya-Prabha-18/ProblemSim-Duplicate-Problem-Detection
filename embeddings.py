import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')  # Download tokenizer for stemming

# Load dataset
df = pd.read_excel("duplicates_details.xlsx")  # Replace with your actual file name

# Fill NaN values with empty strings
df = df.fillna("")

# Initialize Porter Stemmer
stemmer = PorterStemmer()

# Function to remove duplicate words and apply stemming
def preprocess_text(text):
    words = text.split()  # Tokenize words
    unique_words = list(dict.fromkeys(words))  # Remove duplicates while preserving order
    stemmed_words = [stemmer.stem(word) for word in unique_words]  # Apply stemming
    return " ".join(stemmed_words)  # Join words back into a sentence

# Apply preprocessing
df["leetcode title"] = df["leetcode title"].apply(preprocess_text)
df["leetcode description"] = df["leetcode description"].apply(preprocess_text)
df["gfg title"] = df["gfg title"].apply(preprocess_text)
df["gfg description "] = df["gfg description "].apply(preprocess_text)

# Load BERT-based sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings
df["lc_title_embedding"] = df["leetcode title"].apply(lambda x: model.encode(x))
df["lc_desc_embedding"] = df["leetcode description"].apply(lambda x: model.encode(x))
df["gfg_title_embedding"] = df["gfg title"].apply(lambda x: model.encode(x))
df["gfg_desc_embedding"] = df["gfg description "].apply(lambda x: model.encode(x))

# Compute cosine similarity
df["lc_title_gfg_title_sim"] = df.apply(lambda row: cosine_similarity([row["lc_title_embedding"]], [row["gfg_title_embedding"]])[0, 0], axis=1)
df["lc_title_gfg_desc_sim"] = df.apply(lambda row: cosine_similarity([row["lc_title_embedding"]], [row["gfg_desc_embedding"]])[0, 0], axis=1)
df["gfg_title_lc_desc_sim"] = df.apply(lambda row: cosine_similarity([row["gfg_title_embedding"]], [row["lc_desc_embedding"]])[0, 0], axis=1)
df["gfg_desc_lc_desc_sim"] = df.apply(lambda row: cosine_similarity([row["gfg_desc_embedding"]], [row["lc_desc_embedding"]])[0, 0], axis=1)

# Drop embedding columns to save space
df = df.drop(columns=["lc_title_embedding", "lc_desc_embedding", "gfg_title_embedding", "gfg_desc_embedding"])

# Save updated dataset
df.to_excel("embeddings_similarity.xlsx", index=False)

# Display sample output
print(df.head())
