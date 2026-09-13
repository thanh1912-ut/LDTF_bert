# Multi-source data pipeline

This pipeline extends the fixed AG News training split with balanced external
examples while preserving the existing validation and official test splits.

## Label sources

| Normalized label | External source | Accepted source category |
| --- | --- | --- |
| World | HuffPost | `WORLD NEWS` |
| Sports | BBC News | `sport` |
| Business | BBC News | `business` |
| Sci/Tech | BBC News | `tech` |

The source rules, sample limit, duplicate threshold, and tokenizer settings are
stored in `configs/pipeline_config.json`.

## Run

From the `LDTF_bert` project directory on Windows:

```powershell
& '.venv/Scripts/python.exe' 'src/pipeline data processed/scripts/run_pipeline.py'
```

The command writes model-ready files to `data/processed` inside this pipeline
directory. It also writes reproducibility manifests and technical audit reports
under `manifests/multisource` and `reports/multisource`.

## Train with the generated dataset

Point the project at this pipeline's data directory before running training:

```powershell
$env:LDTF_DATA_DIR = (Resolve-Path -LiteralPath 'src/pipeline data processed/data').Path
```

The normal project configuration will then resolve:

- `research_train.parquet`
- `research_validation.parquet`
- `research_test.parquet`

Review `reports/multisource/manual_review_samples.csv` before publishing or
sharing a new dataset version. Automated checks are summarized in
`reports/multisource/final_data_report.json`.
