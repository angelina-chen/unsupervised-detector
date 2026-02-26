"""Text embedding with sentence-transformers and caching"""

import os

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def embed_text(texts, model_name="all-mpnet-base-v2", cache_path=None):
    """encode texts into embeddings. Cache to .npy file if cache_path given.

    args:
        texts: list of strings
        model_name: sentence-transformers model name
        cache_path: optional path to .npy cache file
    returns:
        np.ndarray of shape (len(texts), embedding_dim)
    """
    #Check cache
    if cache_path and os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(texts):
            print(f"Loaded cached embeddings from {cache_path} ({embeddings.shape})")
            return embeddings
        else:
            print(f"Cache shape mismatch ({embeddings.shape[0]} vs {len(texts)}), re-encoding")

    print(f"Encoding {len(texts):,} texts with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=256,
        convert_to_numpy=True,
    )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"Cached embeddings to {cache_path}")

    return embeddings
