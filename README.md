# Credit Card Fraud Detection

Machine learning project for detecting fraudulent credit card transactions using Logistic Regression, Decision Tree, Random Forest, and XGBoost models. The project includes a Streamlit app for deployment and evaluation of the selected XGBoost model.

## Dataset

[European credit card transactions (Sept 2013)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) from Kaggle

**Note**: The dataset uses anonymized features: `V1` to `V28` are PCA-derived components (for confidentiality and dimensionality reduction), plus `Amount` (transaction amount) and `Time` (seconds since the first transaction). The target is `Class`, where 0 denotes a legitimate transaction and 1 denotes fraud.

## Project Structure

```
root/
├── data/                 # Dataset files (raw and sample)
├── models/               # Trained model files
├── notebooks/            # Jupyter notebooks for model development and analysis
├── preprocessors/        # Saved preprocessors
└── scripts/              # Utility scripts
```

## Models Explored

The project compares four model architectures:

1. **Logistic Regression**: Linear baseline with a preprocessing pipeline (e.g. standard scaling).
    - Class imbalance handled via `class_weight='balanced'`.

2. **Decision Tree**: Single tree for non-linear decision boundaries and interactions.
    - Class imbalance handled via `class_weight='balanced'`.

3. **Random Forest**: Bagging ensemble of decision trees.
    - Class imbalance handled via `class_weight='balanced_subsample'` per bootstrap sample.

4. **XGBoost**: Gradient boosting for strong predictive performance.
    - Class imbalance handled via `scale_pos_weight` instead of class weights.

## Project Workflow

### Phase 1: Baseline Compairson

- Trained all four models on the same preprocessed data (V1–V28, Amount; Time dropped).
- Used stratified 5-fold cross-validation with AUPRC (average precision) as the metric, since accuracy and ROC-AUC are misleading with severe class imbalance.
- Compared models on AUPRC and test-set precision/recall/F1 on the fraud class.
- XGBoost had the highest AUPRC and best F1 on the fraud class, so it was chosen for the next phase.

### Phase 2: Hyperparameter Tuning

- Tuned XGBoost in two steps:
    1. `GridSearchCV` for `n_estimators` and `learning_rate`,
    2. `RandomizedSearchCV` for `max_depth`, `gamma`, `min_child_weight`, regularization, and sampling parameters.
- Kept AUPRC as the tuning metric.
- Saved the best-tuned XGBoost model for use in the Streamlit app.

## Dependencies Installation

### Option 1: Conda Environment (Recommended)

```bash
conda env create -f fraud_detection_env.yaml
conda activate fraud_detection_env
```

### Option 2: Pip

```bash
pip install -r requirements.txt
```

## Deployment Tips

### Local Host

Streamlit: http://localhost:8501

### Current Public Host

Streamlit: https://fraudulent-transactions-detector.streamlit.app