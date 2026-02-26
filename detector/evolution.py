"""temporal evolution tracking and alert generation"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Alert:
    alert_type: str  # emerging, surging, novel, disappearing, volume_spike
    cluster_id: Optional[int] = None
    description: str = ""
    keywords: list = field(default_factory=list)
    growth_rate: float = 0.0
    recency: float = 0.0
    novelty: float = 0.0
    current_count: int = 0
    baseline_mean: float = 0.0
    statistical_score: float = 0.0
    top_products: list = field(default_factory=list)
    top_companies: list = field(default_factory=list)
    top_states: list = field(default_factory=list)
    example_ids: list = field(default_factory=list)
    grouping: str = ""  # for volume_spike alerts


def _top_values(series, n=3):
    """top n values from a series by frequency."""
    if series is None or len(series) == 0:
        return []
    counts = series.value_counts().head(n)
    return [f"{v} ({c})" for v, c in counts.items()]


def track_topic_evolution(df, labels, config, topic_info):
    """track each cluster's count across windows, classify alerts.
    Returns list of Alert objects.
    """
    windows = sorted(df["window"].unique())
    n_windows = len(windows)
    baseline_n = config["baseline_windows"]
    min_history = config["min_window_history"]

    df = df.copy()
    df["cluster"] = labels

    alerts = []

    cluster_ids = sorted(topic_info.keys())
    for cid in cluster_ids:
        cluster_mask = df["cluster"] == cid
        cluster_df = df[cluster_mask]

        #build time series: count per window
        counts = cluster_df.groupby("window").size()
        ts = np.array([counts.get(w, 0) for w in windows], dtype=float)

        total = ts.sum()
        if total == 0:
            continue

        keywords = topic_info[cid].get("keywords", [])

        #Tecency: fraction in last 2 windows
        last2 = ts[-2:].sum() if n_windows >= 2 else ts[-1:].sum()
        recency = last2 / total if total > 0 else 0.0

        #First window with nonzero count
        first_nonzero = next((i for i, v in enumerate(ts) if v > 0), 0)
        windows_active = n_windows - first_nonzero

        #Novelty: first appeared in last 2 windows
        novelty = 1.0 if first_nonzero >= n_windows - 2 else 0.0

        #Growth rate: current / rolling mean of baseline
        current = ts[-1]
        if windows_active >= min_history and n_windows > baseline_n:
            baseline = ts[-(baseline_n + 1):-1]
            baseline_mean = baseline.mean()
        else:
            baseline_mean = 0.0

        growth_rate = current / baseline_mean if baseline_mean > 0 else 0.0

        #Enrichment
        top_products = _top_values(cluster_df.get("product"))
        top_companies = _top_values(cluster_df.get("company")) if "company" in cluster_df.columns else []
        top_states = _top_values(cluster_df.get("state")) if "state" in cluster_df.columns else []
        example_ids = cluster_df["customer_id"].head(5).tolist()

        #Classify
        is_emerging = recency > config["recency_threshold"] and growth_rate > config["growth_threshold"]
        is_surging = growth_rate > config["surge_threshold"] and windows_active >= min_history
        is_novel = novelty == 1.0 and total >= config["hdbscan_min_cluster_size"]
        is_disappearing = (
            baseline_mean > 10
            and current < config["decay_threshold"] * baseline_mean
            and windows_active >= min_history
        )

        def _make_alert(atype, stat_score):
            return Alert(
                alert_type=atype,
                cluster_id=cid,
                description=f"Topic {cid}: {', '.join(keywords[:5])}",
                keywords=keywords,
                growth_rate=growth_rate,
                recency=recency,
                novelty=novelty,
                current_count=int(current),
                baseline_mean=float(baseline_mean),
                statistical_score=stat_score,
                top_products=top_products,
                top_companies=top_companies,
                top_states=top_states,
                example_ids=example_ids,
            )

        if is_surging:
            score = min(growth_rate / config["surge_threshold"], 1.0)
            alerts.append(_make_alert("surging", score))
        elif is_emerging:
            score = min(growth_rate / config["growth_threshold"], 1.0) * 0.8
            alerts.append(_make_alert("emerging", score))

        if is_novel:
            alerts.append(_make_alert("novel", 0.9))

        if is_disappearing:
            score = 1.0 - (current / baseline_mean) if baseline_mean > 0 else 0.5
            alerts.append(_make_alert("disappearing", min(score, 1.0)))

    #Noise fraction tracking
    noise_counts = []
    total_counts = []
    for w in windows:
        w_mask = df["window"] == w
        total_w = w_mask.sum()
        noise_w = (df.loc[w_mask, "cluster"] == -1).sum()
        total_counts.append(total_w)
        noise_counts.append(noise_w)

    noise_frac = np.array(noise_counts) / np.maximum(np.array(total_counts), 1)
    if n_windows > baseline_n:
        recent_noise = noise_frac[-1]
        baseline_noise = noise_frac[-(baseline_n + 1):-1].mean()
        baseline_std = noise_frac[-(baseline_n + 1):-1].std()
        if baseline_std > 0 and (recent_noise - baseline_noise) / baseline_std > config["volume_zscore_threshold"]:
            alerts.append(Alert(
                alert_type="noise_spike",
                description=f"Noise fraction spiked to {recent_noise:.1%} (baseline {baseline_noise:.1%})",
                statistical_score=0.7,
                current_count=int(noise_counts[-1]),
                baseline_mean=float(baseline_noise),
            ))

    print(f"Topic evolution: {len(alerts)} alerts from {len(cluster_ids)} clusters")
    return alerts


def detect_volume_spikes(df, config):
    """detect volume spikes per categorical grouping.
    checks: product, company, company x product.
    returns list of Alert objects.
    """
    windows = sorted(df["window"].unique())
    n_windows = len(windows)
    baseline_n = config["baseline_windows"]
    min_history = config["min_window_history"]
    z_thresh = config["volume_zscore_threshold"]

    alerts = []

    #Define groupings to check
    groupings = [("product",)]
    if "company" in df.columns:
        groupings.append(("company",))
        groupings.append(("company", "product"))

    for group_cols in groupings:
        group_cols = [c for c in group_cols if c in df.columns]
        if not group_cols:
            continue

        grouped = df.groupby(list(group_cols) + ["window"]).size().reset_index(name="count")

        for name, grp in grouped.groupby(list(group_cols)):
            if not isinstance(name, tuple):
                name = (name,)
            group_label = " | ".join(f"{c}={v}" for c, v in zip(group_cols, name))

            ts = np.zeros(n_windows)
            for _, row in grp.iterrows():
                idx = windows.index(row["window"])
                ts[idx] = row["count"]

            if n_windows <= baseline_n + 1:
                continue

            current = ts[-1]
            baseline = ts[-(baseline_n + 1):-1]
            b_mean = baseline.mean()
            b_std = baseline.std()

            if b_std == 0 or b_mean == 0:
                continue

            #  Require minimum history of nonzero windows
            nonzero_windows = (ts > 0).sum()
            if nonzero_windows < min_history:
                continue

            z = (current - b_mean) / b_std
            if z > z_thresh:
                weight = np.log1p(current)
                alerts.append(Alert(
                    alert_type="volume_spike",
                    description=f"Volume spike: {group_label}",
                    grouping=group_label,
                    current_count=int(current),
                    baseline_mean=float(b_mean),
                    growth_rate=float(current / b_mean) if b_mean > 0 else 0,
                    statistical_score=min(z / (z_thresh * 2), 1.0),
                ))

    print(f"Volume spikes: {len(alerts)} alerts")
    return alerts
