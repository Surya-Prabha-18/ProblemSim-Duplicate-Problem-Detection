import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer, util

# Download necessary resources
nltk.download('punkt')

# Load Dataset
df = pd.read_excel("duplicates_details.xlsx")  # Replace with your filename

# Fill NaN values with empty strings
df = df.fillna("")

# Initialize Stemmer
stemmer = PorterStemmer()

# Function to Preprocess Text
def preprocess_text(text):
    words = text.split()
    unique_words = list(dict.fromkeys(words))  # Remove duplicate words
    stemmed_words = [stemmer.stem(word) for word in unique_words]  # Apply Stemming
    return " ".join(stemmed_words)

# Apply Preprocessing
df["leetcode title"] = df["leetcode title"].apply(preprocess_text)
df["leetcode description"] = df["leetcode description"].apply(preprocess_text)
df["gfg title"] = df["gfg title"].apply(preprocess_text)
df["gfg description "] = df["gfg description "].apply(preprocess_text)

# Load SBERT Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate Embeddings
df["lc_title_emb"] = df["leetcode title"].apply(lambda x: model.encode(x, convert_to_tensor=True))
df["lc_desc_emb"] = df["leetcode description"].apply(lambda x: model.encode(x, convert_to_tensor=True))
df["gfg_title_emb"] = df["gfg title"].apply(lambda x: model.encode(x, convert_to_tensor=True))
df["gfg_desc_emb"] = df["gfg description "].apply(lambda x: model.encode(x, convert_to_tensor=True))

# Compute Cosine Similarities
df["lc_title_gfg_title_sim"] = df.apply(lambda row: util.pytorch_cos_sim(row["lc_title_emb"], row["gfg_title_emb"]).item(), axis=1)
df["lc_title_gfg_desc_sim"] = df.apply(lambda row: util.pytorch_cos_sim(row["lc_title_emb"], row["gfg_desc_emb"]).item(), axis=1)
df["gfg_title_lc_desc_sim"] = df.apply(lambda row: util.pytorch_cos_sim(row["gfg_title_emb"], row["lc_desc_emb"]).item(), axis=1)
df["gfg_desc_lc_desc_sim"] = df.apply(lambda row: util.pytorch_cos_sim(row["gfg_desc_emb"], row["lc_desc_emb"]).item(), axis=1)

# Drop Embeddings Columns to Save Space
df.drop(columns=["lc_title_emb", "lc_desc_emb", "gfg_title_emb", "gfg_desc_emb"], inplace=True)

# Save to Excel
df.to_excel("sbert_similarity.xlsx", index=False)

# Display Output
print(df.head())
