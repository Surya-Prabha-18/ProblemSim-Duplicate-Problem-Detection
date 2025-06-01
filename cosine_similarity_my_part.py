# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load dataset from Excel file
# df = pd.read_excel("duplicates_details.xlsx")  # Replace with your actual filename

# # Fill NaN values with empty strings
# df = df.fillna("")

# # Combine all text columns for a consistent TF-IDF vocabulary
# all_text = pd.concat([
#     df["leetcode title"], df["leetcode description"],
#     df["gfg title"], df["gfg description "]
# ])

# # Initialize and fit a single TF-IDF Vectorizer on all text columns
# vectorizer = TfidfVectorizer()
# vectorizer.fit(all_text)

# # Transform individual columns using the same vectorizer
# tfidf_lc_title = vectorizer.transform(df["leetcode title"])
# tfidf_lc_desc = vectorizer.transform(df["leetcode description"])
# tfidf_gfg_title = vectorizer.transform(df["gfg title"])
# tfidf_gfg_desc = vectorizer.transform(df["gfg description "])

# # Compute row-wise cosine similarity
# df["lc_title_gfg_title_sim"] = [
#     cosine_similarity(tfidf_lc_title[i], tfidf_gfg_title[i])[0, 0] for i in range(len(df))
# ]
# df["lc_title_gfg_desc_sim"] = [
#     cosine_similarity(tfidf_lc_title[i], tfidf_gfg_desc[i])[0, 0] for i in range(len(df))
# ]
# df["gfg_title_lc_desc_sim"] = [
#     cosine_similarity(tfidf_gfg_title[i], tfidf_lc_desc[i])[0, 0] for i in range(len(df))
# ]
# df["gfg_desc_lc_desc_sim"] = [
#     cosine_similarity(tfidf_gfg_desc[i], tfidf_lc_desc[i])[0, 0] for i in range(len(df))
# ]

# # Save the updated dataset with new similarity columns
# df.to_excel("updated_dataset_with_similarity.xlsx", index=False)

# # Display sample output
# print(df.head())




import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt')  # Download required tokenizer for stemming

# Load dataset from Excel file
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

# Apply preprocessing to all text columns
df["leetcode title"] = df["leetcode title"].apply(preprocess_text)
df["leetcode description"] = df["leetcode description"].apply(preprocess_text)
df["gfg title"] = df["gfg title"].apply(preprocess_text)
df["gfg description "] = df["gfg description "].apply(preprocess_text)

# Combine all text for a consistent TF-IDF vocabulary
all_text = pd.concat([
    df["leetcode title"], df["leetcode description"],
    df["gfg title"], df["gfg description "]
])

# Initialize and fit TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(all_text)

# Transform individual columns using the same vectorizer
tfidf_lc_title = vectorizer.transform(df["leetcode title"])
tfidf_lc_desc = vectorizer.transform(df["leetcode description"])
tfidf_gfg_title = vectorizer.transform(df["gfg title"])
tfidf_gfg_desc = vectorizer.transform(df["gfg description "])

# Compute row-wise cosine similarity
df["lc_title_gfg_title_sim"] = [
    cosine_similarity(tfidf_lc_title[i], tfidf_gfg_title[i])[0, 0] for i in range(len(df))
]
df["lc_title_gfg_desc_sim"] = [
    cosine_similarity(tfidf_lc_title[i], tfidf_gfg_desc[i])[0, 0] for i in range(len(df))
]
df["gfg_title_lc_desc_sim"] = [
    cosine_similarity(tfidf_gfg_title[i], tfidf_lc_desc[i])[0, 0] for i in range(len(df))
]
df["gfg_desc_lc_desc_sim"] = [
    cosine_similarity(tfidf_gfg_desc[i], tfidf_lc_desc[i])[0, 0] for i in range(len(df))
]

# Save the updated dataset with new similarity columns
df.to_excel("coseine_similarity_part1.xlsx", index=False)

# Display sample output
print(df.head())
