# 📌 Multimodal Fashion Retrieval & Unsupervised Clustering

### *Information Retrieval Course Project – B.Tech CSE (Data Science)*

---

# 🚀 Overview

This project implements a **multimodal retrieval system** for fashion products using **CLIP (ViT-B/32)**, **FAISS**, and **K-Means clustering**.
It supports:

✔️ Text → Image retrieval
✔️ Semantic similarity search
✔️ Multimodal (text + image) embeddings
✔️ Unsupervised clustering of products
✔️ CLIP-based embeddings for both text and images
✔️ Fast vector search with FAISS

The goal is to demonstrate how modern IR systems use **embeddings**, **vector search**, and **unsupervised learning** to organize and retrieve items efficiently.

---

# 🧠 Key Features

### 🔹 1. **Text Embedding (CLIP Text Encoder)**

* Converts cleaned descriptions + display names + category
* Produces 512-dim dense semantic vectors
* Normalized for cosine similarity

### 🔹 2. **Image Embedding (CLIP Image Encoder)**

* Vision Transformer (ViT-B/32)
* Processes all product images
* Generates 512-dim vectors

### 🔹 3. **Multimodal Fusion**

Final embedding =

```
0.5 × text_embedding + 0.5 × image_embedding
```

Normalized to unit length.
This captures both visual and textual meaning.

### 🔹 4. **FAISS Similarity Search**

* Builds a fast IndexFlatIP (cosine similarity) index
* Enables real-time retrieval
* Supports top-K nearest neighbor search

### 🔹 5. **Text → Image Retrieval**

Example:

```python
search_by_text("red floral dress", k=6)
```

Returns the top 6 matching products.

### 🔹 6. **Unsupervised Clustering (K-Means)**

* Clusters all multimodal embeddings
* Used to discover semantic groups of products
* Number of clusters = **10**

### 🔹 7. **Evaluation**

Silhouette Score:

```
0.11007
```

This is typical for **high-dimensional multimodal data**, where category boundaries naturally overlap.

---

# 📂 Project Structure

```
IR_Fashion_Retrieval/
│── fashion_clustering.ipynb        # Main notebook
│── README.md                       # Project documentation
│── data/                           # Dataset (ignored in GitHub)
│── clip_outputs/                   # Saved embeddings (ignored)
│── .gitignore                      # Ignore large folders/files
```

---

# 🛠️ Technologies Used

| Component     | Technology                           |
| ------------- | ------------------------------------ |
| Embeddings    | CLIP (ViT-B/32)                      |
| Vector Search | FAISS (IndexFlatIP)                  |
| Clustering    | K-Means (scikit-learn)               |
| Evaluation    | Silhouette Score                     |
| Language      | Python                               |
| Environment   | Jupyter Notebook                     |
| Libraries     | PyTorch, NumPy, Pandas, scikit-learn |

---

# 🧬 Pipeline Architecture

```
          ┌────────────────────┐
          │   Raw Dataset      │
          │ (Images + Text)    │
          └─────────┬──────────┘
                    │
     ┌──────────────┴──────────────┐
     │                              │
┌────▼─────┐                  ┌─────▼────┐
│ CLIP Text│                  │ CLIP Image│
│ Encoder  │                  │ Encoder   │
└────▲─────┘                  └────▲──────┘
     │                              │
     └─────────┬────────────┬───────┘
               │  Fusion    │
        0.5 × text + 0.5 × image
               │
        ┌──────▼────────┐
        │ Multimodal     │
        │ Embeddings (512│
        └──────▲────────┘
               │
     ┌─────────┴───────────┐
     │                     │
┌────▼─────┐         ┌─────▼────────┐
│  FAISS   │         │  K-Means      │
│ Retriever│         │ Clustering    │
└────▲─────┘         └─────▲────────┘
     │                     │
     └────────────┬────────┘
                  │
            ┌─────▼───────┐
            │ Final Output │
            │ Search +     │
            │ Clusters     │
            └──────────────┘
```

---

# 🔎 Example: Text-Based Retrieval

```python
results = search_by_text("red floral dress", k=6)
show_results(results)
```

Returns:

* Relevant product images
* Display name
* Category
* Description
* Similarity score

---

# 📊 Evaluation

### **Silhouette Score:**

```
0.11007
```

### Interpretation:

* Typical for multimodal 512D embeddings
* Categories in fashion often overlap visually and textually
* Clustering still shows meaningful grouping
* Visual inspection confirms cluster structure

---

# 📘 How to Run the Project

### 1. Install dependencies:

```sh
pip install torch torchvision faiss-cpu clip scikit-learn pandas numpy pillow tqdm
```

### 2. Open the notebook:

```
fashion_clustering.ipynb
```

### 3. Run cells in order:

1. Load dataset
2. Text cleaning
3. CLIP text embeddings
4. CLIP image embeddings
5. Multimodal fusion
6. FAISS indexing
7. Text search function
8. K-Means clustering
9. Evaluation (Silhouette Score)

---

# 🧾 Conclusion

This project demonstrates a complete **Information Retrieval pipeline** using **modern embedding techniques**, including:

* Multimodal CLIP embeddings
* Vector-based similarity search
* Unsupervised clustering
* Real-time semantic retrieval

It showcases the power of deep learning and vector search for organizing and retrieving fashion products using both text and images.

---

# 👤 Author

CHRISTO TONIO
B.Tech CSE (Data Science)
Year: 4th Year
Subject: Information Retrieval

---
