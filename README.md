# SARA - Spam Analysis & Risk Assistant


---

## 1. Problem Statement

The objective of this project is to develop a machine learning based spam email classification system that classifies emails as **Spam** or **Not Spam**.

Five machine learning classification models are implemented, tuned and evaluated on the selected dataset. Their performance is compared using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

A Streamlit application is provided to upload test data, select a trained model, and view its evaluation results.


---

## 2. Dataset Description

**Dataset:** UCI Spambase Dataset  |  **Source:** UCI Machine Learning Repository (`https://archive.ics.uci.edu/dataset/94/spambase`)

Spambase is a binary classification dataset used to identify whether an email is spam or not spam. It contains **4,601 instances and 57 features**: 48 word-frequency features, 6 character-frequency features and 3 features describing runs of capital letters. All features are numeric. The data was checked for missing values and none were found.

During preprocessing, **391 duplicate rows** were removed, leaving **4,210 unique instances**. A stratified train-test split was then performed using:

- Training data: 80%
- Test data: 20%
- Random seed: 7319
- Stratification: on the target variable


### Dataset Split

| Set | Rows | Not Spam (0) | Spam (1) | % Not Spam | % Spam |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full Data | 4210 | 2531 | 1679 | 60.12% | 39.88% |
| Train | 3368 | 2025 | 1343 | 60.12% | 39.88% |
| Test | 842 | 506 | 336 | 60.10% | 39.90% |


### Target Variable

| Value | Meaning |
| --- | --- |
| `0` | Not Spam |
| `1` | Spam |


### Preprocessing

- All 57 features are numeric, so no one-hot encoding was required.
- Logistic Regression, KNN and SVM use a `log1p` transform followed by standardisation (`StandardScaler`) to handle the heavily skewed word-frequency and capital-run-length features.
- Decision Tree and Naive Bayes (Gaussian) are trained on the raw features, as they do not depend on feature scale.
- Preprocessing and model are combined in a scikit-learn Pipeline, so the saved models accept raw test data directly.
- The SVM (RBF kernel) is wrapped in `CalibratedClassifierCV` to obtain probability estimates for AUC.
- The test set (842 rows, features plus the `is_spam` target) was saved as `test_data.csv` in the project root and is used for all evaluations. No model sees it during training or tuning.


### Hyperparameter Tuning

All five models were tuned with the same protocol: 5-fold stratified `GridSearchCV` on the training set only, selecting the parameters with the highest MCC. The best model of each search was then refitted on the full training set and evaluated once on the held-out test set. Training used scikit-learn 1.9.1.

| ML Model Name | Parameter grid searched | Best parameters |
| --- | --- | --- |
| Logistic Regression | `C` = 0.01, 0.1, 1, 10, 100 | `C` = 0.1 |
| Decision Tree | `max_depth` = 5, 10, 20, None; `min_samples_leaf` = 1, 3, 5 | `max_depth` = 10, `min_samples_leaf` = 1 |
| KNN | `n_neighbors` = 3, 5, 7, 11, 15; `weights` = uniform, distance | `n_neighbors` = 15, `weights` = distance |
| Naive Bayes | `var_smoothing` = 1e-9, 1e-8, 1e-7, 1e-6 | `var_smoothing` = 1e-6 |
| Support Vector Machine | `C` = 1, 10, 100; `gamma` = 0.001, 0.01, 0.1 | `C` = 10, `gamma` = 0.01 |

The same dataset and test split are used to evaluate all five machine learning models.


---

## 3. GitHub Repository Link

**GitHub Repository:** `https://github.com/prasanna3000/sara`

The repository contains the complete source code, `requirements.txt`, the test data, the saved trained models for all five classifiers, and this README file.


---

## 4. Models Used

The following five classification models were implemented, tuned and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Support Vector Machine (SVM)


### Model Comparison

All metrics are computed on the 842-row held-out test set. The best value in each column is shown in bold; the Support Vector Machine is best on all six metrics.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9311 | 0.9748 | 0.9187 | 0.9077 | 0.9132 | 0.8561 |
| Decision Tree | 0.9109 | 0.9078 | 0.9065 | 0.8661 | 0.8858 | 0.8134 |
| KNN | 0.9347 | 0.9779 | 0.9297 | 0.9048 | 0.9170 | 0.8634 |
| Naive Bayes | 0.8836 | 0.9321 | 0.8839 | 0.8155 | 0.8483 | 0.7557 |
| **Support Vector Machine** | **0.9430** | **0.9786** | **0.9364** | **0.9196** | **0.9279** | **0.8809** |


### Cross-Validation on the Training Set

The MCC below is the mean (and standard deviation) over the 5 validation folds for each model's best parameters. It agrees closely with the test-set MCC for the stronger models, which suggests the test results are not a quirk of one split.

| ML Model Name | CV MCC (mean +/- std) | Test MCC |
| --- | ---: | ---: |
| Logistic Regression | 0.8692 +/- 0.0110 | 0.8561 |
| Decision Tree | 0.8236 +/- 0.0199 | 0.8134 |
| KNN | 0.8492 +/- 0.0196 | 0.8634 |
| Naive Bayes | 0.7225 +/- 0.0534 | 0.7557 |
| **Support Vector Machine** | **0.8804 +/- 0.0093** | **0.8809** |


### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
| --- | --- |
| **Logistic Regression** | A strong, stable linear model (Accuracy 0.9311, AUC 0.9748, MCC 0.8561). Strong regularisation (C = 0.1) worked best, and Precision (0.9187) and Recall (0.9077) are well balanced. Its test MCC is consistent with its cross-validation MCC (0.8692), and it is fast and interpretable. This suggests the spam signal is largely linear in the log-scaled word-frequency features. |
| **Decision Tree** | Accuracy 0.9109 and the lowest AUC of all models (0.9078). Limiting the depth to 10 keeps the tree reasonable, but a single tree still produces coarse probability scores, and its Recall (0.8661) is lower than that of the other non-Naive-Bayes models. Test MCC is 0.8134. |
| **KNN** | Second best overall (Accuracy 0.9347, AUC 0.9779, MCC 0.8634), slightly ahead of Logistic Regression and almost level with the SVM on AUC. Distance-weighted voting with k = 15 worked best (k = 15 was the largest value searched). It depends on the log-transform and scaling and is slower at prediction time, since every query is compared with the training data. |
| **Naive Bayes** | The weakest model overall (Accuracy 0.8836, Precision 0.8839, Recall 0.8155, F1 0.8483, MCC 0.7557); only its AUC beats the Decision Tree's. It is also the least stable (cross-validation MCC standard deviation 0.0534). The independence assumption does not hold for correlated word-frequency features. Its AUC of 0.9321 shows the ranking of emails is still reasonable. |
| **Support Vector Machine** | Best on all six metrics: Accuracy 0.9430, AUC 0.9786, Precision 0.9364, Recall 0.9196, F1 0.9279 and MCC 0.8809. The RBF kernel captures non-linear interactions between features. It also has the highest cross-validation MCC (0.8804) with the smallest spread (0.0093), so its performance is consistent. It is the most expensive model to train. |
| **Overall Winner for your dataset?** | **Support Vector Machine (SVM)**. It has the highest MCC on both the test set (0.8809) and in cross-validation (0.8804), and the best value on all six metrics. MCC was used as the deciding criterion because it summarises all four confusion-matrix cells and stays reliable on the roughly 60/40 class split. KNN (MCC 0.8634) and Logistic Regression (0.8561) are close behind, so the margin over them is modest. |


### Overall Winner

**Support Vector Machine (SVM)** is the best-performing model on the test dataset. Its results:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.9430 |
| AUC | 0.9786 |
| Precision | 0.9364 |
| Recall | 0.9196 |
| F1 Score | 0.9279 |
| MCC | 0.8809 |


---

## 5. Streamlit Application

The project includes a Streamlit application (SARA) for evaluating the saved machine learning models. The application provides:

- Download of the sample test file (`test_data.csv`, the same 842-row held-out test set used above) from the sidebar
- Test CSV upload with validation of the required feature columns and the `is_spam` target
- Model selection dropdown for the five trained models
- All six evaluation metrics: Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix (counts or percentages) and classification report
- ROC curve for the selected model
- Side-by-side comparison of all five models, with the top-performing model identified
- Predictions table with misclassified-row filter and CSV download

The uploaded test data is used only for evaluation. The saved models are loaded and used to generate predictions without retraining on the uploaded data.

**Streamlit App:** `https://sara-spam-analyzer.streamlit.app`