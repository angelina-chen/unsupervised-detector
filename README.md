# unsupervised-detector

Unsupervised detector for customer complaint patterns. Embeds text, clusters topics, tracks evolution over time, and flags emerging, surging, novel, and disappearing patterns. Takes in all complaints data as complaints.csv file from CFPB consumer complaint database (https://www.consumerfinance.gov/data-research/consumer-complaints/).

## Quick Start

```bash
pip install -r requirements.txt

# 1. Preprocess CFPB data (stratified sample)
python preprocess.py --input ~/complaints.csv --output data.csv --max-rows 100000

# 2. Run detector
python run.py --input data.csv --output output/
```

## Output Files

- `alerts.csv` — ranked alerts with type, keywords, growth rate, enrichment
- `summary.txt` — human-readable alert narrative

## Configuration

All hyperparameters are in `detector/config.py`. Override via CLI:

```bash
python run.py --input data.csv --hdbscan-min-cluster-size 50 --surge-threshold 4.0
```

