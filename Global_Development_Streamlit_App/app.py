
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

st.set_page_config(page_title="Global Development Clustering", layout="wide")

st.title("🌍 Global Development Clustering App")
st.write("Cluster countries based on development indicators using multiple clustering algorithms.")

@st.cache_data
def load_data():
    return pd.read_excel("data.xlsx")


df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

numeric_df = df.select_dtypes(include=np.number)

imputer = SimpleImputer(strategy="median")
imputed_data = imputer.fit_transform(numeric_df)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(imputed_data)

st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox(
    "Select Clustering Algorithm",
    ["K-Means", "Hierarchical", "DBSCAN"]
)

if model_choice == "K-Means":
    k = st.sidebar.slider("Number of Clusters (K)", 2, 8, 3)
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(scaled_data)

elif model_choice == "Hierarchical":
    k = st.sidebar.slider("Number of Clusters", 2, 8, 3)
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(scaled_data)

else:
    eps = st.sidebar.slider("Epsilon", 0.5, 3.0, 1.2)
    model = DBSCAN(eps=eps, min_samples=5)
    labels = model.fit_predict(scaled_data)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

st.subheader("Cluster Visualization (PCA Projection)")
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(
    x=pca_data[:, 0],
    y=pca_data[:, 1],
    hue=labels,
    palette="Set2",
    ax=ax
)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
st.pyplot(fig)

st.subheader("Cluster Distribution")
st.write(pd.Series(labels).value_counts())
