# Internals_Basics

## MLOps Lab CIE — BMS College of Engineering

**Course:** MLOps (24AM6AEMLO)
**USN:** 1BM23AI173
**Date:** 04th May 2026

---

## Scenario

MLOps pipeline for EconPulse — an economic forecasting service that predicts inflation indices to advise policy makers and fund managers.

---

## Dataset Features

Feature | Range
---|---
money_supply_growth_pct | 2-15
crude_oil_price | 40-120
import_volume_index | 80-150
interest_rate | 2-10

Target: inflation_index

---

## Tasks

Task | Description | Marks
---|---|---
Task 1 | Experiment Tracking & Model Comparison | 6
Task 2 | Hyperparameter Tuning | 8
Task 3 | Model Versioning | 8
Task 4 | Model Promotion | 8

---

## Project Structure

MLOPs_Lab_CIE/
    data/
        training_data.csv
        new_data.csv
    src/
        train.py
        tune.py
        register_model.py
        promote_model.py
    models/
    results/
        step1_s1.json
        step2_s2.json
        step3_s6.json
        step4_s7.json
    requirements.txt
    .gitignore

---

## Setup and Run

1. cd into MLOPs_Lab_CIE
2. python -m venv venv
3. venv\Scripts\activate
4. pip install -r requirements.txt
5. set MLFLOW_TRACKING_URI
6. python src/train.py
7. python src/tune.py
8. python src/register_model.py
9. python src/promote_model.py

---

## Tools Used

- Python 3.13
- MLflow
- scikit-learn
- pandas
- numpy
