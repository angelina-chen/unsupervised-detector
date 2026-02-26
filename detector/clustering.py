"""UMAP dimensionality reduction + HDBSCAN clustering + c-TF-IDF topic keywords."""

import numpy as np
import hdbscan
import umap
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def cluster_embeddings(embeddings, texts, config):
    """reduce dimensions with UMAP, cluster with HDBSCAN, recluster noise.
    returns:
        labels: np.ndarray of cluster assignments (int, -1 = noise)
        topic_info: dict {cluster_id: {"size": int, "example_ids": list}}
    """
    n = len(embeddings)
    print(f"UMAP: {embeddings.shape[1]}d → {config['umap_n_components']}d ...")
    reducer = umap.UMAP(
        n_components=config["umap_n_components"],
        n_neighbors=config["umap_n_neighbors"],
        min_dist=config["umap_min_dist"],
        random_state=42,
        low_memory=True,
    )
    reduced = reducer.fit_transform(embeddings)

    print(f"HDBSCAN clustering (min_cluster_size={config['hdbscan_min_cluster_size']})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config["hdbscan_min_cluster_size"],
        min_samples=config["hdbscan_min_samples"],
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  {n_clusters} clusters, {n_noise} noise points ({100*n_noise/n:.1f}%)")

    #second pass: recluster noise points
    noise_mask = labels == -1
    if noise_mask.sum() >= config["noise_recluster_min_size"] * 2:
        print(f"Second-pass clustering on {noise_mask.sum()} noise points...")
        noise_reduced = reduced[noise_mask]
        sub_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config["noise_recluster_min_size"],
            min_samples=max(2, config["hdbscan_min_samples"] // 2),
            core_dist_n_jobs=-1,
        )
        sub_labels = sub_clusterer.fit_predict(noise_reduced)

        n_sub = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
        if n_sub > 0:
            max_label = labels.max()
            noise_indices = np.where(noise_mask)[0]
            for i, sl in enumerate(sub_labels):
                if sl >= 0:
                    labels[noise_indices[i]] = max_label + 1 + sl
            new_noise = (labels == -1).sum()
            print(f"  Recovered {n_sub} sub-clusters, {new_noise} noise remaining")

    #build topic_info
    topic_info = {}
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        indices = np.where(mask)[0]
        topic_info[int(cid)] = {
            "size": int(mask.sum()),
            "example_ids": indices[:5].tolist(),
        }

    total_clusters = len(topic_info)
    print(f"Total topics: {total_clusters}")
    return labels, topic_info


def compute_ctfidf(texts, labels, topic_info):
    """Compute c-TF-IDF: top keywords per cluster. Groups texts by cluster, builds a document-per-cluster, then applies TF-IDF across clusters to find discriminative terms.
    then returns: updated topic_info with 'keywords' added.
    """
    cluster_ids = sorted(topic_info.keys())
    if not cluster_ids:
        return topic_info

    #build one document per cluster by concatenating texts
    cluster_docs = []
    for cid in cluster_ids:
        mask = labels == cid
        doc = " ".join(t for i, t in enumerate(texts) if mask[i])
        cluster_docs.append(doc)

    #c-TF-IDF
    vectorizer = CountVectorizer(
        max_features=10000,
        stop_words="english",
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    counts = vectorizer.fit_transform(cluster_docs)
    tfidf = TfidfTransformer().fit_transform(counts)

    feature_names = vectorizer.get_feature_names_out()
    for idx, cid in enumerate(cluster_ids):
        row = tfidf[idx].toarray().flatten()
        top_indices = row.argsort()[-5:][::-1]
        keywords = [feature_names[i] for i in top_indices if row[i] > 0]
        topic_info[cid]["keywords"] = keywords

    return topic_info
