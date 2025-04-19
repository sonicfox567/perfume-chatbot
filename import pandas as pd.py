import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer, util
df = pd.read_excel("Perfume.xlsx")
df.fillna("Unknown", inplace=True)

# Load NLP model
model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight and good

# Prepare perfume description embeddings
if 'Embedding' not in df.columns:
    df['Combined_Description'] = df['Name'].astype(str) + " " + df['Brand'].astype(str) + " " + df['Notes'].astype(str)
    df['Embedding'] = df['Combined_Description'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    print("Embeddings computed.")
else:
    df['Embedding'] = df['Embedding'].apply(eval)  # if loading from CSV/parquet later

# Chatbot loop
print("🌸 Welcome to your AI Perfume Assistant 🌸")
print("Describe what kind of scent you're looking for (e.g., 'something warm and woody for winter' or 'smells like the beach')")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye! Hope you find your perfect scent.")
        break

    user_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = df['Embedding'].apply(lambda x: float(util.pytorch_cos_sim(user_embedding, x)))
    df['Score'] = scores
    top_matches = df.sort_values('Score', ascending=False).head(5)

    print("\n🌟 Recommended perfumes based on your description:")
    for i, row in top_matches.iterrows():
        print(f" - {row['Name']} by {row['Brand']} ({row['Gender']})")