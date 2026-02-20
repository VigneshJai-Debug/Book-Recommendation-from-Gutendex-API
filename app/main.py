import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
from scipy import sparse

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "gutendex-cleaned-dataset_v3.csv")
VECTORIZER_PATH = os.path.join(PROJECT_DIR, "models", "tfidf_vectorizer.pkl")
KNN_PATH = os.path.join(PROJECT_DIR, "models", "knn_tfidf.pkl")
TFIDF_MATRIX_PATH = os.path.join(PROJECT_DIR, "models", "tfidf_matrix.npz")

@st.cache_resource
def load_resources():
    df = pd.read_csv(DATA_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    knn = joblib.load(KNN_PATH)
    tfidf_matrix = sparse.load_npz(TFIDF_MATRIX_PATH)
    return df, vectorizer, knn, tfidf_matrix

df, vectorizer, knn, tfidf_matrix = load_resources()

st.title("📚 Book Recommendation System")
st.markdown("Search by title OR enter description if book not found.")

title_input = st.text_input("Enter Book Title (optional)")
description_input = st.text_area("Enter Description (only if book not found)", height=120)

use_hybrid = st.sidebar.checkbox("Use Hybrid Ranking (Popularity + Similarity)", value=False)

def get_recommendations_from_vector(vector, exclude_index=None, k=10):
    distances, indices = knn.kneighbors(vector, n_neighbors=k+1)
    rec_indices = indices[0]
    rec_distances = distances[0]

    if exclude_index is not None:
        mask = rec_indices != exclude_index
        rec_indices = rec_indices[mask]
        rec_distances = rec_distances[mask]

    rec_indices = rec_indices[:k]
    rec_distances = rec_distances[:k]

    recommendations = df.iloc[rec_indices].copy()
    recommendations["distance"] = rec_distances
    recommendations["similarity"] = 1 - recommendations["distance"]

    return recommendations

if st.button("Recommend"):

    recommendations = None

    if title_input.strip():
        matches = df[df["title"].str.lower().str.contains(title_input.lower(), na=False)]

        if not matches.empty:
            idx = matches.index[0]
            st.success(f"Book found: {df.loc[idx, 'title']}")

            book_vector = tfidf_matrix[idx]
            recommendations = get_recommendations_from_vector(
                book_vector,
                exclude_index=idx,
                k=10
            )

        else:
            st.warning("Book not found. Using description instead.")

    if recommendations is None:
        if not description_input.strip():
            st.error("Please enter either a valid title or description.")
            st.stop()

        processed_query = clean_text(description_input)
        query_vec = vectorizer.transform([processed_query])

        recommendations = get_recommendations_from_vector(
            query_vec,
            exclude_index=None,
            k=10
        )

    if use_hybrid:
        if "download_count" in recommendations.columns:

            c_min = recommendations["download_count"].min()
            c_max = recommendations["download_count"].max()

            if c_max - c_min != 0:
                recommendations["norm_pop"] = (
                    recommendations["download_count"] - c_min
                ) / (c_max - c_min)
            else:
                recommendations["norm_pop"] = 0

            recommendations["hybrid"] = (
                0.7 * recommendations["similarity"]
                + 0.3 * recommendations["norm_pop"]
            )

            recommendations = recommendations.sort_values("hybrid", ascending=False)

    else:
        recommendations = recommendations.sort_values("similarity", ascending=False)

    st.subheader("Top Recommendations")

    for _, row in recommendations.iterrows():
        st.markdown(f"### {row['title']}")
        st.write(f"Author: {row['authors']}")
        st.write(f"Similarity: {row['similarity']:.3f}")
        st.write("---")
