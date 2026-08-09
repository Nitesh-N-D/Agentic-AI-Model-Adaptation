# MLKTL Scenario & Fine-Tuning Module — Reference Implementation (Weeks 1–4)

A runnable reference implementation of the Week 1–4 design (drift detection,
dataset/baseline selection, fine-tuning planning, evaluation gates) on
synthetic data. This is a design-validation reference, not the production
Week 5+ build.

## Folder Structure

```
mlktl_python_implementation/
├── README.md
├── requirements.txt
├── data/
│   ├── sample_baseline_metrics.csv     # synthetic baseline feature snapshot
│   └── sample_current_metrics.csv      # synthetic current (drifted) snapshot
├── src/
│   ├── scenario_schema.py              # Week 1: MLKTL scenario data model
│   ├── drift_analysis.py               # Week 1: PSI / KL-divergence / classification
│   ├── dataset_selection.py            # Week 2: dataset eligibility rules
│   ├── baseline_selection.py           # Week 2: checkpoint selection rules
│   ├── finetuning_planner.py           # Week 3: strategy decision tree + hyperparams
│   ├── evaluation_planner.py           # Week 4: evaluation & validation gates
│   └── orchestrator.py                 # ties all weeks together end-to-end
├── tests/
│   └── test_pipeline.py                # pytest coverage for all modules
└── output/                             # generated scenario JSON reports land here
```

## Setup

```bash
cd mlktl_python_implementation
pip install -r requirements.txt --break-system-packages
```
(Drop `--break-system-packages` if you're using a virtual environment instead — recommended:)
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the End-to-End Pipeline

```bash
python3 -m src.orchestrator
```

This runs the full Week 1–4 flow on the synthetic CSVs in `data/`:
drift analysis → dataset selection → baseline selection → fine-tuning plan →
evaluation gate → a scenario JSON report written to `output/`.

## Run the Tests

```bash
python3 -m pytest tests/ -v
```

## Regenerate the Synthetic Sample Data (optional)

```bash
python3 -c "
import numpy as np, pandas as pd
np.random.seed(42)
baseline = np.random.normal(loc=50, scale=10, size=2000)
current = np.random.normal(loc=58, scale=12, size=2000)
pd.DataFrame({'feature_value': baseline}).to_csv('data/sample_baseline_metrics.csv', index=False)
pd.DataFrame({'feature_value': current}).to_csv('data/sample_current_metrics.csv', index=False)
"
```

## Notes

- All "compute" data (error rates, KPI deltas, evaluation metrics) inside
  `orchestrator.py` is synthetic/hardcoded for demonstration — replace with
  real monitoring/evaluation pipeline outputs during Week 5 implementation.
- The interface contract these modules produce (`MLKTLScenario.to_dict()`)
  matches the schema in the Engineering Design Document, Appendix B, so it
  can be handed directly to the Decision Engine, Model Catalog, and
  Deployment Manager without reshaping.
