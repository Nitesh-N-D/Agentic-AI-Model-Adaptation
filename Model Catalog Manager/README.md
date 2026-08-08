# Model Catalog Manager (Simulator)

A self-contained Python simulation of a **Model Catalog Manager**, built
around four responsibilities:

1. Search the Model Catalog for the incoming model.
2. Check whether a similar or existing model is already registered.
3. Replace the existing model entry if a match is found.
4. Maintain model versions and forward the entry to the Fine-tuning Engineer.

No external dependencies — pure Python 3 standard library (`json`,
`dataclasses`, `difflib`, `datetime`, `unittest`).

## Folder Structure

```
model_catalog_manager/
├── data/
│   ├── model_catalog.json       # "database" of registered models
│   ├── version_history.json     # archive of replaced/old versions
│   ├── finetuning_queue.json    # handoff queue read by the Fine-tuning Engineer
│   └── incoming_samples.json    # 3 sample incoming models (exact/similar/new)
├── src/
│   ├── __init__.py
│   ├── models.py                # ModelMetadata dataclass + validation
│   ├── catalog_store.py         # JSON file storage layer (load/save)
│   ├── search_engine.py         # search + exact/similar matching logic
│   ├── version_manager.py       # version bump + history archiving
│   ├── finetuning_handoff.py    # writes finalized model onto the handoff queue
│   └── catalog_manager.py       # orchestrator wiring the 4 responsibilities
├── tests/
│   ├── __init__.py
│   └── test_catalog_manager.py  # existing / similar / new / invalid test cases
├── main.py                      # demo runner (Model Catalog Manager side)
├── fine_tuning_engineer.py      # independent consumer (Fine-tuning Engineer side)
└── README.md
```

## How to Run

From inside the `model_catalog_manager/` folder:

```bash
python main.py
```

This processes 3 sample incoming models (exact match, similar match, new
model) through the full pipeline, updates `data/model_catalog.json` and
`data/version_history.json`, and forwards each result onto
`data/finetuning_queue.json`.

Then, to simulate the Fine-tuning Engineer picking up the forwarded jobs:

```bash
python fine_tuning_engineer.py
```

To run the automated tests:

```bash
python -m unittest discover tests -v
```

## Data Flow

1. `main.py` loads a sample incoming model (raw dict) from
   `data/incoming_samples.json`.
2. `ModelCatalogManager.process_incoming_model()` validates it into a
   `ModelMetadata` object (`src/models.py`). Missing required fields
   raise `InvalidModelDataError` and the pipeline stops safely.
3. **Search (Responsibility 1):** `SearchEngine.search()`
   (`src/search_engine.py`) loads all catalog entries via
   `CatalogStore` and scores every entry against the incoming model
   using: exact `model_id`, exact name+architecture+framework
   signature, and fuzzy name similarity (`difflib.SequenceMatcher`)
   blended with task/architecture bonuses.
4. **Match check (Responsibility 2):** `SearchEngine.determine_match()`
   classifies the best-scoring candidate as `"exact"`, `"similar"`, or
   `"new"`.
5. **Replace or register (Responsibility 3):**
   - `exact` / `similar` → `_replace_existing()` overwrites the
     matched catalog entry in place (keeping its original
     `model_id`), while `VersionManager.next_version()` bumps the
     version number.
   - `new` → `_register_new()` appends a brand-new entry to the
     catalog.
6. **Version history + forward (Responsibility 4):**
   - Before any replace, `VersionManager.archive_previous_version()`
     writes the old entry into `data/version_history.json` so no data
     is lost.
   - The final model (new or updated) is handed off via
     `FineTuningHandoff.forward()`, which appends a job record onto
     `data/finetuning_queue.json` — this is the "actual data handoff
     mechanism" (a file-based queue), not a print statement.
7. `fine_tuning_engineer.py` runs independently, reads
   `data/finetuning_queue.json`, "picks up" pending jobs, and marks
   them as `picked_up_by_fine_tuning_engineer` — demonstrating a real
   two-service handoff.

## Test Cases

`tests/test_catalog_manager.py` covers:

- **Existing model** — incoming model shares the exact `model_id` →
  detected as `exact`, version bumped (`1.0` → `1.1`), old version
  archived, catalog entry count stays the same.
- **Similar model** — incoming model has a different `model_id` but a
  closely related name (`"SentimentBERT"` vs `"SentimentBERT-v2"`) →
  detected as `similar`, the closest existing entry is replaced
  (identity/`model_id` preserved), not duplicated.
- **New model** — incoming model has no name/architecture overlap with
  anything in the catalog → detected as `new`, appended as a fresh
  entry, catalog count increases by one, nothing is archived.
- **Invalid input** — incoming model missing required fields is
  rejected during validation before touching the catalog.
