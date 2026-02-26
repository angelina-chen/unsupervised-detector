
from detector.config import CONFIG
from detector.ingest import load_data
from detector.embeddings import embed_text
from detector.clustering import cluster_embeddings, compute_ctfidf
from detector.evolution import track_topic_evolution, detect_volume_spikes
from detector.report import rank_alerts, write_report, save_intermediate


def run_pipeline(input_path, output_dir, cache_dir=None, config_overrides=None):
    """Run the full detection pipeline."""
    import os

    config = dict(CONFIG)
    if config_overrides:
        config.update(config_overrides)

    os.makedirs(output_dir, exist_ok=True)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # ingest
    df = load_data(input_path)

    #Embed
    cache_path = os.path.join(cache_dir, "embeddings.npy") if cache_dir else None
    embeddings = embed_text(df["text"].tolist(), config["embedding_model"], cache_path)

    #Cluster
    labels, topic_info = cluster_embeddings(embeddings, df["text"].tolist(), config)
    df["cluster"] = labels

    #c-TF-IDF
    topic_info = compute_ctfidf(df["text"].tolist(), labels, topic_info)

    # Evolution
    alerts = track_topic_evolution(df, labels, config, topic_info)
    alerts += detect_volume_spikes(df, config)

    # Report
    alerts = rank_alerts(alerts)
    write_report(alerts, output_dir)
    save_intermediate(output_dir, df, labels, topic_info, config)

    return alerts
