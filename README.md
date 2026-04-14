# COMP9417 Tabular Binary Classification Project

This project is a modular baseline for a COMP9417 group assignment on tabular binary classification.

Dataset assumptions:

- Input data is a CSV file placed in the project root or under `data/raw/`.
- Target column is `felt_rested`.
- Columns to drop are `person_id`, `sleep_disorder_risk`, and `cognitive_performance_score`.

Planned models:

- XGBoost
- xRFM

Planned preprocessing:

- Automatically detect numeric and categorical features
- Standard-scale numeric features
- One-hot encode categorical features
- Produce a fully numeric feature matrix for model training

Planned evaluation:

- Train/validation/test split
- Accuracy
- ROC-AUC
- Training time
- Inference time per sample

Current scaffold:

- `scripts/inspect_data.py` inspects a CSV and prints schema information
- `src/config.py` stores shared project constants

Next steps will be added incrementally after confirmation.
