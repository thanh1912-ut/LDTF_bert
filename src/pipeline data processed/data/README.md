# Data layout

This directory separates immutable source data from future intermediate and
model-ready datasets.

## Directories

- `raw/ag_news`: Original AG News train and test CSV files.
- `raw/bbc_news`: Kaggle archive and extracted BBC News data.
- `raw/huffpost`: Kaggle archive and extracted HuffPost News Category data.
- `standardized`: Reserved for source data converted to a shared schema.
- `merged`: Combined data before and after duplicate removal.
- `processed`: Final model-ready train, validation, and test Parquet files.

Files under `raw` must not be edited in place. Future processing should write
new artifacts to `standardized`, `merged`, or `processed`.

## Sources

- BBC articles fulltext and category:
  https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category
- News Category Dataset (HuffPost):
  https://www.kaggle.com/datasets/rmisra/news-category-dataset

Review the source dataset pages and their license terms before redistribution.

The multi-source pipeline writes its audit files under `reports/multisource`
and `manifests/multisource`. It preserves the existing AG News validation and
test samples so experiments remain comparable.
