"""all hyperparameters in one place. Can override via CLI or direct import."""

CONFIG = {
    #embedding
    "embedding_model": "all-mpnet-base-v2",
    "embedding_batch_size": 256,

    #UMAP
    "umap_n_components": 5,
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.0,

    #HDBSCAN
    "hdbscan_min_cluster_size": 20,
    "hdbscan_min_samples": 5,

    #second-pass noise reclustering
    "noise_recluster_min_size": 10,

    #Temporal windows
    "window_freq": "ME",
    "baseline_windows": 6,
    "min_window_history": 4,

    #Alert thresholds
    "growth_threshold": 2.0,
    "surge_threshold": 3.0,
    "recency_threshold": 0.5,
    "volume_zscore_threshold": 2.5,
    "decay_threshold": 0.2,

    #report
    "top_alerts": 50,
}
