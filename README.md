# SARA — Spam Analysis & Risk Assistant

## 1. Problem Statement

The objective of this project is to develop a machine learning based spam email classification system that classifies emails as **Spam** or **Not Spam**.

Five machine learning classification models are implemented and evaluated on the selected dataset. Their performance is compared using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The project also provides a Streamlit application that allows users to upload test data, select a trained model, and view its evaluation results.

---

## 2. Dataset Description

**Dataset:** UCI Spambase Dataset

**Source:** UCI Machine Learning Repository

The Spambase dataset is a binary classification dataset used to identify whether an email is spam or not spam.

The dataset contains **4,601 instances and 57 features**. The dataset was checked for missing values, and no missing values were found.

During preprocessing, **391 duplicate rows** were removed, resulting in **4,210 unique instances**.

The resulting dataset was divided using a **stratified train-test split** with:

- Training data: 80%
- Test data: 20%
- Random seed: 2026
- Stratification based on the target variable

The final split contains:

| Set | Rows | Not Spam (0) | Spam (1) | % Not Spam | % Spam |
|---|---:|---:|---:|---:|---:|
| Full Data | 4210 | 2531 | 1679 | 60.12% | 39.88% |
| Train | 3368 | 2025 | 1343 | 60.12% | 39.88% |
| Test | 842 | 506 | 336 | 60.10% | 39.90% |

### Target Variable

| Value | Meaning |
|---|---|
| `0` | Not Spam |
| `1` | Spam |

The same dataset and test split are used to evaluate all five machine learning models.

---

## 3. GitHub Repository Link

**GitHub Repository:**  
`https://github.com/prasanna3000/sara`

The repository contains the complete source code, required dependencies, test data, saved machine learning models, preprocessing artifacts, and README file.

---

## 4. Models Used

The following five classification models were implemented and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

### Model Comparison

The models were evaluated using the following six metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9264 | 0.9750 | 0.9335 | 0.8780 | 0.9049 | 0.8460 |
| Decision Tree | 0.9014 | 0.8970 | 0.8776 | 0.8750 | 0.8763 | 0.7944 |
| KNN | 0.9216 | 0.9695 | 0.9193 | 0.8810 | 0.8997 | 0.8359 |
| Naive Bayes | 0.8076 | 0.9405 | 0.6875 | 0.9494 | 0.7975 | 0.6526 |
| Support Vector Machine | **0.9371** | **0.9799** | **0.9464** | **0.8929** | **0.9188** | **0.8685** |

---

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Logistic Regression achieved strong overall performance with an Accuracy of **0.9264**, AUC of **0.9750**, F1 Score of **0.9049**, and MCC of **0.8460**. It provided a good balance between precision and recall. |
| **Decision Tree** | Decision Tree achieved an Accuracy of **0.9014** and an MCC of **0.7944**. Its performance was lower than Logistic Regression, KNN, and SVM across the main evaluation metrics. |
| **KNN** | KNN achieved an Accuracy of **0.9216**, AUC of **0.9695**, and MCC of **0.8359**. It performed well overall and produced results close to Logistic Regression. |
| **Naive Bayes** | Naive Bayes achieved the highest Recall among the models at **0.9494**, meaning it identified a high proportion of spam instances. However, its Precision of **0.6875**, F1 Score of **0.7975**, and MCC of **0.6526** were substantially lower, resulting in the lowest overall performance among the five models. |
| **Support Vector Machine** | Support Vector Machine achieved the best overall performance, with the highest Accuracy (**0.9371**), AUC (**0.9799**), Precision (**0.9464**), F1 Score (**0.9188**), and MCC (**0.8685**). |
| **Overall Winner for your dataset?** | **Support Vector Machine (SVM)** is the overall winner for this dataset, achieving the highest MCC of **0.8685** and the highest values for Accuracy, AUC, Precision, and F1 Score among the five models. |

---

### Overall Winner

**Support Vector Machine (SVM)** is the overall best-performing model on the test dataset.

Its evaluation results are:

| Metric | Score |
|---|---:|
| Accuracy | **0.9371** |
| AUC | **0.9799** |
| Precision | **0.9464** |
| Recall | **0.8929** |
| F1 Score | **0.9188** |
| MCC | **0.8685** |

The SVM model achieved the highest MCC of **0.8685**, making it the top-performing model based on the selected model comparison criterion.

---

## 5. Streamlit Application

The project includes a Streamlit application for evaluating the saved machine learning models.

The application provides:

- Test CSV upload
- Model selection
- Evaluation metrics
- Global performance metrics
- Spam and Not Spam class-specific metrics
- Confusion matrix
- ROC curve
- Model comparison
- Identification of the top-performing model

The uploaded test data is used only for evaluation. The saved models are loaded and used to generate predictions without retraining the models on the uploaded test data.

The deployed application allows the user to select a model and view its performance on the uploaded test dataset.

**Streamlit App:**  
`<PASTE YOUR STREAMLIT COMMUNITY CLOUD LINK HERE>`