import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# load models (cached for performance)
@st.cache_resource
def load_artifacts():
    model = joblib.load('./models/fraud_detector_xgboost_v1.joblib')
    sample_data = pd.read_csv('./data/creditcard_sample.csv')
    return model, sample_data

model, sample_data = load_artifacts()

# title
st.title("Credit Card Fraud Detection")

# upload data or use built‑in data
option = st.radio("Choose data source:", ["Built‑in data", "Upload data"])

if option == "Upload data":
    uploaded = st.file_uploader("Upload a CSV with the same columns as built‑in data", type=["csv"])
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

# split into two logical views
df_data_view = df_results.drop(columns=['fraud_probability', 'predicted_class'])
df_pred_view = df_results[['fraud_probability', 'predicted_class']]

col_data, col_pred = st.columns(2)

with col_data:
    st.subheader("Transaction data")
    st.dataframe(df_data_view.head(20))

with col_pred:
    st.subheader("Predictions & probabilities")
    st.dataframe(df_pred_view.head(20))

# evaluate predictions
if y_true is not None:

    # classification metrics
    report_dict = classification_report(
        y_true,
        pred,
        target_names=["Non-Fraud", "Fraud"],
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).T

    st.subheader("Evaluation on the PCA-transformed dataset")
    st.markdown("Classification metrics")
    st.dataframe(report_df.style.format("{:.3f}"))

    # confusion matrix
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Non-Fraud", "Fraud"],
                yticklabels=["Non-Fraud", "Fraud"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

# footer
st.markdown("---")
st.markdown("© 2026 La Wun Nannda")