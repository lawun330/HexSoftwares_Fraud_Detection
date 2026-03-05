import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # remove index column if it exists
else:
    df = sample_data.copy()

st.write("Input data shape:", df.shape)

# preprocess data
df_model = df.drop(columns=["Time"], errors="ignore")
X = df_model.drop(columns=['Class'])
y_true = df_model['Class'] if 'Class' in df_model.columns else None
unique_classes = np.unique(y_true)

# make predictions
proba = model.predict_proba(X)[:, 1]
pred = model.predict(X)
df_results = df_model.copy()
df_results['fraud_probability'] = proba
df_results['predicted_class'] = pred

# split into two logical views
df_data_view = df_results.drop(columns=['fraud_probability', 'predicted_class'])
df_pred_view = df_results[['fraud_probability', 'predicted_class']]
col_data, col_pred = st.columns([3, 1])  # data column 3x wider than pred column

with col_data:
    st.subheader("Transaction Data:")
    st.dataframe(
        df_data_view.head(20),
        use_container_width=True,  # stretch to full column width
    )

with col_pred:
    st.subheader("Predictions:")
    st.dataframe(df_pred_view.head(20))

# evaluate predictions
if y_true is not None:
    st.subheader("Evaluation on the PCA-transformed Dataset")

    # classification metrics
    if unique_classes.size < 2:
        st.warning("Only one class present in the uploaded data!!!")

    report_dict = classification_report(
        y_true,
        pred,
        labels=[0, 1],  # force both classes even if data contains only one class
        target_names=["Non-Fraud", "Fraud"],
        output_dict=True,
        zero_division=0,  # avoid division-by-zero errors
    )

    # accuracy metric
    accuracy = report_dict["accuracy"]

    # other metrics
    rows = ["Non-Fraud", "Fraud", "macro avg", "weighted avg"]
    report_df = (
        pd.DataFrame({k: report_dict[k] for k in rows})
        .T[["precision", "recall", "f1-score", "support"]]
    )

    # display classification metrics
    st.markdown("#### Classification report:")
    left, left_center, left_center_center, center, right_center, right_center_center, right = st.columns([1, 1, 1, 3, 1, 1, 1])
    with center:
        # accuracy metric
        st.write(f"**accuracy**: {accuracy:.2f}")

        # other metrics
        st.markdown("**other metrics**:")
        report_styled = (
            report_df.style
            .format({
                "precision": "{:.2f}",
                "recall": "{:.2f}",
                "f1-score": "{:.2f}",
                "support": "{:.0f}",
            })
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "center")]},
            ])
        )

        # render the HTML table
        st.markdown(
            report_styled.to_html(),
            unsafe_allow_html=True,
        )

    # confusion matrix
    st.markdown("#### Confusion matrix:")
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Non-Fraud", "Fraud"],
                yticklabels=["Non-Fraud", "Fraud"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # display confusion matrix
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.pyplot(fig, use_container_width=False)

# footer
st.markdown("---")
st.markdown("© 2026 La Wun Nannda")