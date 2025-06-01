import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load your Excel file
df = pd.read_excel("embeddings_similarity.xlsx")

# Positive samples (known duplicates)
pos_features = df[[
    'lc_title_gfg_title_sim',
    'lc_title_gfg_desc_sim',
    'gfg_title_lc_desc_sim',
    'gfg_desc_lc_desc_sim'
]]
pos_labels = np.ones(len(pos_features))

# Create synthetic non-duplicate (negative) samples by shuffling
shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
neg_features = pd.DataFrame({
    'lc_title_gfg_title_sim': pos_features['lc_title_gfg_title_sim'].values,
    'lc_title_gfg_desc_sim': shuffled['lc_title_gfg_desc_sim'].values,
    'gfg_title_lc_desc_sim': shuffled['gfg_title_lc_desc_sim'].values,
    'gfg_desc_lc_desc_sim': shuffled['gfg_desc_lc_desc_sim'].values
})
neg_labels = np.zeros(len(neg_features))

# Combine
X = pd.concat([pos_features, neg_features], ignore_index=True)
y = np.concatenate([pos_labels, neg_labels])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
