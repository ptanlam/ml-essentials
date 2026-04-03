# ML Essentials

A personal learning repository for exploring machine learning fundamentals and essential concepts.

## Multiple Types of Regression

**Notebook**: [MultipleTypesOfRegression.ipynb](MultipleTypesOfRegression.ipynb)
**Dataset**: [processed_dataset/diamons_processed.csv](processed_dataset/diamons_processed.csv)

Compared four regression algorithms on the processed Diamonds dataset to predict `price`, using a shared `build_and_train_model` helper for consistent 80/20 splits and R² evaluation.

Models implemented:

- **Linear Regression** — standard OLS baseline
- **Lasso** (`alpha=0.8`, `max_iter=10000`) — L1 regularization; drives less important feature weights to zero, performing implicit feature selection
- **Ridge** (`alpha=0.9`) — L2 regularization; shrinks all coefficients evenly, handles multicollinearity better than OLS
- **SGDRegressor** (`max_iter=2000`) — stochastic gradient descent; scales well to large datasets by updating weights on mini-batches

Key concepts:

- **L1 (Lasso) vs L2 (Ridge) regularization** — Lasso produces sparse models by zeroing out weak features; Ridge retains all features but shrinks them
- **`alpha`** — regularization strength; higher values impose stronger penalties on large coefficients
- **SGD** — approximates gradient descent iteratively, making it practical when full-batch optimization is too expensive

## Linear Regression with Categorical Data

**Notebook**: [RegressionUsingCategoricalData.ipynb](RegressionUsingCategoricalData.ipynb)
**Dataset**: [dataset/diamonds.csv](dataset/diamonds.csv) → [processed_dataset/diamons_processed.csv](processed_dataset/diamons_processed.csv)

Explored linear regression on the Diamonds dataset to predict `price` from a mix of numerical and categorical features (`cut`, `color`, `clarity`, `carat`, etc.).

Preprocessing pipeline:

- EDA with correlation heatmap, boxplots by `cut`/`color`, and a scatter plot of `carat` vs `price`
- **Ordinal encoding** for `clarity` — mapped 8 grades (IF → I1) to integers 7–0 to preserve natural ordering
- **One-hot encoding** for `cut` and `color` via `pd.get_dummies`
- **Feature scaling** — `sklearn.preprocessing.scale` applied to the 6 numerical features (`carat`, `depth`, `table`, `x`, `y`, `z`); scaled values merged back into the dataframe
- Dropped rows with NaN values before saving the processed dataset

Modelling:

- Trained `sklearn.linear_model.LinearRegression` with an 80/20 train/test split
- Evaluated with **R² score** on both training and test sets

Key concepts:

- **Ordinal vs nominal encoding** — `clarity` has a meaningful order so it's label-encoded; `cut` and `color` are nominal so they're one-hot encoded
- **Feature scaling before regression** — prevents features with large ranges from dominating the OLS solution

## Linear Regression with Numerical Data

**Notebook**: [LinearRegressionWithNumericalData.ipynb](LinearRegressionWithNumericalData.ipynb)
**Dataset**: [dataset/boston_data.csv](dataset/boston_data.csv)

Explored linear regression on the Boston Housing dataset to predict median home values (`medv`) from numerical features (rooms, age, pupil-teacher ratio, etc.).

Pipeline:

- EDA with `seaborn.lmplot` to visualize relationships between individual features (`rm`, `age`, `ptratio`) and the target
- Trained `sklearn.linear_model.LinearRegression` with an 80/20 train/test split
- Evaluated with **R² score** on both training and test sets
- Extended analysis with `statsmodels.OLS` to get full statistical summary (coefficients, p-values, confidence intervals)

Key concepts:

- **OLS (Ordinary Least Squares)** — minimizes the sum of squared residuals to find the best-fit hyperplane
- **`sm.add_constant`** — adds an intercept term to the design matrix for the statsmodels OLS fit
- **R² score** — measures proportion of variance in the target explained by the model

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

## Binary Classification on Numerical Data

**Notebook**: [ClassificationBinaryNumericalDataset.ipynb](ClassificationBinaryNumericalDataset.ipynb)
**Dataset**: [dataset/gender_voice_dataset.csv](dataset/gender_voice_dataset.csv)

Explored binary classification using the Gender Voice dataset, predicting speaker gender from acoustic features (mean frequency, SD, median, IQR, skew, etc.).

Pipeline:

- **Label encoding** — target column (`label`) encoded with `LabelEncoder` (male/female → 0/1)
- **Logistic Regression** with L2 penalty (`solver='liblinear'`) trained on an 80/20 split
- Evaluated with **confusion matrix**, **accuracy**, **precision**, and **recall**

Key concepts:

- **Logistic Regression** — models the probability of a binary outcome using the sigmoid function; decision boundary is linear in feature space
- **L2 penalty** — regularizes coefficients to prevent overfitting, equivalent to Ridge for classification
- **Confusion matrix** — breaks down TP, TN, FP, FN to understand error types beyond raw accuracy
- **Precision vs Recall** — precision measures correctness of positive predictions; recall measures coverage of actual positives

## Clustering Using Labeled Data

**Notebook**: [ClusteringUsingLabeledData.ipynb](ClusteringUsingLabeledData.ipynb)
**Dataset**: [dataset/cars.csv](dataset/cars.csv)

Explored K-Means clustering on the Cars dataset where ground-truth brand labels are available, enabling cluster quality evaluation beyond silhouette score.

Pipeline:

- Cleaned column names (stripped leading whitespace), coerced `cubicinches` and `weightlbs` to numeric, dropped NaN rows
- **Label encoded** `brand` column for use as ground-truth labels in supervised metrics
- Features used for clustering: `mpg`, `cubicinches`, `cylinders`, `hp`
- Used the **elbow method** (plotting inertia for k=1–14) to identify the optimal number of clusters
- Fit `KMeans(n_clusters=3)` and evaluated with five metrics

Evaluation metrics:

- **Silhouette score** — intra-cluster cohesion vs. inter-cluster separation (unsupervised)
- **Homogeneity** — each cluster contains only members of a single class
- **Completeness** — all members of a class are in the same cluster
- **V-measure** — harmonic mean of homogeneity and completeness
- **Adjusted Rand score** — similarity between predicted and true cluster assignments, corrected for chance
- **Adjusted Mutual Information** — mutual info between cluster labels and ground truth, adjusted for chance

Key concepts:

- **Elbow method** — plot inertia (sum of squared distances to centroid) vs. k; the "elbow" indicates diminishing returns from adding more clusters
- Having ground-truth labels unlocks supervised clustering metrics; without them, only unsupervised metrics like silhouette apply

## Clustering Using Unlabeled Data

**Notebook**: [ClusteringUsingUnlabeledData.ipynb](ClusteringUsingUnlabeledData.ipynb)
**Dataset**: [dataset/Mall_Customers.csv](dataset/Mall_Customers.csv)

Explored unsupervised clustering with K-Means on the Mall Customers dataset, grouping customers by Annual Income and Spending Score.

Pipeline:

- Dropped categorical/demographic columns (`Gender`, `Age`) to keep only the two numerical features
- Shuffled data with `sample(frac=1)` before fitting
- Fit `KMeans` with `n_clusters=2` and `n_clusters=5`, visualizing cluster assignments and centroids with scatter plots
- Evaluated cluster quality with **silhouette score** for both configurations

Key concepts:

- **K-Means** — partitions data into *k* clusters by iteratively assigning points to the nearest centroid and recomputing centroids until convergence
- **Cluster centroids** — the mean position of all points in a cluster; `cluster_centers_` from scikit-learn
- **Silhouette score** — measures how similar a point is to its own cluster vs. neighboring clusters; ranges from -1 to 1, higher is better
- Comparing `k=2` vs `k=5` illustrates how choice of *k* affects cluster granularity and cohesion

## Regression Using Neural Network

**Notebook**: [RegressionUsingNeuralNetwork.ipynb](RegressionUsingNeuralNetwork.ipynb)
**Dataset**: [dataset/Advertising.csv](dataset/Advertising.csv)

Explored regression with a feedforward neural network using PyTorch on the Advertising dataset. The goal was to predict sales from TV, radio, and newspaper ad spend.

Pipeline:

- Features (`TV`, `radio`, `newspaper`) scaled with `sklearn.preprocessing.scale` before training
- Data converted to `torch.float` tensors for PyTorch compatibility
- **Network architecture**: `Linear(3 → 100)` → `ReLU` → `Linear(100 → 1)`
- **Loss**: `MSELoss`
- **Optimizer**: Adam (`lr=0.0001`), trained for 10,000 iterations
- Predictions detached from the computation graph and evaluated with **R² score** against the test set (80/20 split)
