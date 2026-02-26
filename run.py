
import argparse
import os
import sys
import time

from detector.config import CONFIG
from detector.ingest import load_data
from detector.embeddings import embed_text
from detector.clustering import cluster_embeddings, compute_ctfidf
from detector.evolution import track_topic_evolution, detect_volume_spikes
from detector.report import rank_alerts, write_report, save_intermediate


def main():
    parser = argparse.ArgumentParser(description="Conway Detector — complaint pattern detection")
    parser.add_argument("--input", required=True, help="Preprocessed CSV path")
    parser.add_argument("--output", default="output/", help="Output directory")
    parser.add_argument("--cache-dir", default=None, help="Cache directory for embeddings")

    #Allow overriding any config key
    for key, default in CONFIG.items():
        flag = f"--{key.replace('_', '-')}"
        parser.add_argument(flag, type=type(default), default=None, help=f"Override {key} (default: {default})")

    args = parser.parse_args()

    #build config overrides from CLI args
    config = dict(CONFIG)
    for key in CONFIG:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val
            print(f"  Override: {key} = {val}")

    output_dir = args.output
    cache_dir = args.cache_dir or os.path.join(output_dir, "cache")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    t0 = time.time()

    #Ingest
    print("\n=== 1. Loading data ===")
    df = load_data(args.input)

    #embed
    print("\n=== 2. Embedding ===")
    cache_path = os.path.join(cache_dir, "embeddings.npy")
    embeddings = embed_text(df["text"].tolist(), config["embedding_model"], cache_path)

    #Cluster
    print("\n=== 3. Clustering ===")
    labels, topic_info = cluster_embeddings(embeddings, df["text"].tolist(), config)
    df["cluster"] = labels

    #c-TF-IDF keywords
    print("\n=== 4. Computing topic keywords ===")
    topic_info = compute_ctfidf(df["text"].tolist(), labels, topic_info)

    # Evolution & alerts
    print("\n=== 5. Tracking evolution ===")
    alerts = track_topic_evolution(df, labels, config, topic_info)
    alerts += detect_volume_spikes(df, config)

    # Report
    print("\n=== 6. Generating report ===")
    alerts = rank_alerts(alerts)
    top_n = config["top_alerts"]
    if len(alerts) > top_n:
        print(f"  Showing top {top_n} of {len(alerts)} alerts")
        alerts = alerts[:top_n]

    write_report(alerts, output_dir)
    save_intermediate(output_dir, df, labels, topic_info, config)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Results in {output_dir}/")


if __name__ == "__main__":
    main()
