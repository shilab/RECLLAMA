import pandas as pd
import numpy as np
import random
import ast
from itertools import product
from gensim.models import Word2Vec, KeyedVectors
from node2vec import Node2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import networkx as nx


# === Step 1: Load and Preprocess Data ===
df = pd.read_csv("symptoms_processed_medical_records_with_proteins_filtered.csv")
df["procedures"] = df["procedures"].apply(ast.literal_eval)
df["proteins"] = df["proteins"].apply(ast.literal_eval)

# === Step 2: Train Word2Vec Model for Procedures and Proteins ===
procedure_sequences = df["procedures"].tolist()
protein_sequences = df["proteins"].tolist()
w2v_model = Word2Vec(procedure_sequences + protein_sequences, vector_size=100, window=5, min_count=1, sg=1)

def get_embedding(codes):
    valid = [w2v_model.wv[code] for code in codes if code in w2v_model.wv]
    return np.mean(valid, axis=0) if valid else np.zeros(w2v_model.vector_size)

df["procedure_embedding"] = df["procedures"].apply(get_embedding)
df["protein_embedding"] = df["proteins"].apply(get_embedding)

# === Step 3: Build Procedure-Protein Relationship Graph ===
procedure_protein_pairs = []
for _, row in df.iterrows():
    procedure_protein_pairs.extend(product(row["procedures"], row["proteins"]))

relation_df = pd.DataFrame(procedure_protein_pairs, columns=["procedure", "protein"])
relation_counts = relation_df.value_counts().reset_index(name="count")

G = nx.Graph()
for _, row in relation_counts.iterrows():
    G.add_edge(row["procedure"], row["protein"], weight=row["count"])

# === Step 4: Node2Vec Embedding ===
node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=100, workers=1)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
model.wv.save_word2vec_format("node2vec_embeddings.txt")

# === Step 5: Create Positive and Negative Samples ===
positive_samples = list(G.edges())
all_procedures = list(set(df["procedures"].explode()))
all_proteins = list(set(df["proteins"].explode()))

X_pos = [np.concatenate((model.wv[proc], model.wv[prot])) for proc, prot in positive_samples]
y_pos = [1] * len(X_pos)

negative_samples = []
while len(negative_samples) < len(positive_samples):
    proc, prot = random.choice(all_procedures), random.choice(all_proteins)
    if (proc, prot) not in positive_samples:
        negative_samples.append((proc, prot))

X_neg = [np.concatenate((model.wv[proc], model.wv[prot])) for proc, prot in negative_samples]
y_neg = [0] * len(X_neg)

X = X_pos + X_neg
y = y_pos + y_neg
pairs = positive_samples + negative_samples

# === Step 6: Train/Test Split ===
X_train, X_test, y_train, y_test, pairs_train, pairs_test = train_test_split(X, y, pairs, test_size=0.2, random_state=42)

# === Step 7: Train Classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Step 8: Evaluate and Save Predictions ===
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

df_results = pd.DataFrame(pairs_test, columns=["procedure", "protein"])
df_results["true_label"] = y_test
df_results["predicted_label"] = y_pred
df_results["confidence_score"] = y_prob
df_results.to_csv("procedure_protein_predictions.csv", index=False)

# Group by procedure
predicted_links = df_results[df_results["predicted_label"] == 1]
grouped = predicted_links.groupby("procedure")["protein"].apply(list).reset_index()
grouped.to_csv("predicted_links_grouped_by_procedure.csv", index=False)

# === Step 9: Map Predicted Proteins Back to Dataset ===
proc_to_proteins = dict(zip(grouped["procedure"], grouped["protein"]))

def map_predicted_proteins(procs):
    proteins = set()
    for proc in procs:
        if proc in proc_to_proteins:
            proteins.update(proc_to_proteins[proc])
    return list(proteins)

df["predicted_proteins"] = df["procedures"].apply(map_predicted_proteins)
df.to_csv("test_with_predicted_proteins_421.csv", index=False)

print("All models and predictions saved successfully!")
