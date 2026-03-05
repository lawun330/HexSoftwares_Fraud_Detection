import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_resource
def load_artifacts():
    model = joblib.load('./models/xgboost_tuned_v1.joblib')
    sample_data = pd.read_csv('./data/creditcard_sample.csv')
    return model, sample_data
model, sample_data = load_artifacts()
st.title("Credit Card Fraud Detection (PCA Dataset)")

# upload data or use built‑in data
option = st.radio("Choose data source:", ["Built‑in data", "Upload CSV"])
if option == "Upload CSV":
    uploaded = st.file_uploader("Upload a CSV with columns like creditcard.csv", type=["csv"])
    if uploaded is None:
        st.stop()
    df = pd.read_csv(uploaded)
else:
    df = sample_data.copy()
st.write("Input data shape:", df.shape)

# preprocess data
df_model = df.drop(columns=['Time'])
X = df_model.drop(columns=['Class'])
y_true = df_model['Class'] if 'Class' in df_model.columns else None

# make predictions
proba = model.predict_proba(X)[:, 1]
pred = model.predict(X)
df_results = df_model.copy()
df_results['fraud_probability'] = proba
df_results['predicted_class'] = pred
st.subheader("Predictions")
st.write(df_results.head(20))

# evaluate predictions
if y_true is not None:
    st.subheader("Evaluation on this dataset")
    st.text(classification_report(y_true, pred, target_names=["Non-Fraud", "Fraud"]))
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Non-Fraud", "Fraud"],
                yticklabels=["Non-Fraud", "Fraud"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)