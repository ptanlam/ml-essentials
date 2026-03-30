# ML Essentials

A personal learning repository for exploring machine learning fundamentals and essential concepts.

## Voting Classifier (Hard & Soft)

**Notebook**: [VotingClassifier.ipynb](VotingClassifier.ipynb)
**Dataset**: [processed_dataset/PimaIndians_processed.csv](processed_dataset/PimaIndians_processed.csv)

Explored ensemble learning using the Pima Indians Diabetes dataset. Implemented and compared two voting strategies:

- **Hard voting** — each classifier casts a vote and the majority class wins
- **Soft voting** — classifiers output class probabilities, and the averaged probabilities determine the final prediction

## Bagging and Pasting

**Notebook**: [BaggingPasting.ipynb](BaggingPasting.ipynb)
**Dataset**: [dataset/insurance.csv](dataset/insurance.csv) → [processed_dataset/insurance_processed.csv](processed_dataset/insurance_processed.csv)

Explored bagging and pasting as ensemble techniques for regression using the Insurance Charges dataset. The goal was to predict medical insurance charges from patient features (age, sex, BMI, smoker status, region, etc.).

- **Bagging** (Bootstrap Aggregating) — each base learner trains on a random subset sampled *with* replacement; reduces variance without increasing bias
- **Pasting** — same idea but sampling is done *without* replacement; each instance can only appear in one subset

Both approaches use `BaggingRegressor` wrapping a `DecisionTreeRegressor` with 500 estimators.

Key concepts applied:

- **Out-of-Bag (OOB) evaluation** — with bagging, ~37% of samples are never drawn for a given tree; those samples act as a free validation set, enabling generalization estimation without a separate holdout split
- **`bootstrap=True`** → bagging with OOB scoring available
- **`bootstrap=False`** → pasting; OOB score is not applicable
- Model performance evaluated using **R² score** on a held-out test set (80/20 split)

## Gradient Boosting

**Notebook**: [GradientBoosting.ipynb](GradientBoosting.ipynb)
**Dataset**: [processed_dataset/insurance_processed.csv](processed_dataset/insurance_processed.csv)

Explored gradient boosting for regression using the same Insurance Charges dataset. Unlike bagging/pasting which trains estimators independently in parallel, gradient boosting trains trees **sequentially** — each tree fits the residual errors of the previous one.

Two approaches implemented:

- **Manual gradient boosting** — built step by step using three `DecisionTreeRegressor` instances where each subsequent tree trains on the residuals (`y - ŷ`) of the prior tree; final prediction is the sum of all trees
- **Sklearn `GradientBoostingRegressor`** — equivalent implementation using scikit-learn with `max_depth=3`, `n_estimators=3`, `learning_rate=1.0`

Key concepts:

- Each tree corrects the mistakes of the ensemble so far by fitting residual errors
- **Learning rate** scales each tree's contribution — lower values require more trees but can generalize better
- **`n_estimators`** controls the number of sequential trees (boosting stages)
- Model performance evaluated using **R² score** on a held-out test set (80/20 split)
