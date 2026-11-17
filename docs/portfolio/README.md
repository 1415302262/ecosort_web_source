# EcoSort portfolio bundle

Open [`index.html`](index.html) for the offline recruiter-facing experiment page.

When merged to `main`, the Pages workflow publishes this directory and the report PDF automatically. This static deployment does not load a model or call an external API.

The bundle contains only sanitized aggregate and per-sample evaluation fields. It intentionally excludes API keys, provider response identifiers, host-specific absolute paths, raw API caches, attempt ledgers, checkpoints, and source images.

Regenerate from the audited local experiment artifacts:

```bash
conda run -n cptac python scripts/analyze_ecosort_cascade_robustness.py
conda run -n cptac python scripts/export_ecosort_portfolio_data.py
conda run -n cptac python scripts/plot_ecosort_cascade_portfolio.py
```
