import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Telco Churn Classification App", page_icon="📊", layout="wide")

st.title("📊 Machine Learning Model Evaluation Dashboard")
st.markdown("### Telco Customer Churn Prediction & Model Benchmarking")
st.write("Upload your test dataset to evaluate trained Machine Learning models in real time.")

st.sidebar.header("🕹️ Controls & Model Selection")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)", 
    type=["csv"],
    help="Upload 'test_data.csv' containing features and the target 'Churn' column."
)

model_option = st.sidebar.selectbox(
    "Select ML Classification Model",
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest (Ensemble)")
)

model_file_map = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib"
}

@st.cache_resource
def load_artifacts(model_filename):
    try:
        model = joblib.load(f'model/{model_filename}')
        preprocessor = joblib.load('model/preprocessor.joblib')
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading saved artifacts from 'model/' directory: {e}")
        return None, None

def clean_target_series(series):
    """Safely converts string/boolean/numeric target series into binary 0/1 integers."""
    clean_series = series.astype(str).str.strip().str.capitalize()
    target_map = {
        'Yes': 1, 'No': 0, 
        '1': 1, '0': 0, 
        'True': 1, 'False': 0
    }
    mapped = clean_series.map(target_map)
    if mapped.isnull().any():
        mapped = pd.to_numeric(series, errors='coerce').fillna(0)
    return mapped.astype(int).values

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Test Dataset Preview")
    st.dataframe(df_test.head(5), width="stretch")
    
    if 'Churn' not in df_test.columns:
        st.error("Uploaded CSV must contain the target column 'Churn'.")
    else:
        model, preprocessor = load_artifacts(model_file_map[model_option])
        
        if model is not None and preprocessor is not None:
            y_true = clean_target_series(df_test['Churn'])
                
            df_test['TotalCharges'] = pd.to_numeric(df_test['TotalCharges'].replace(' ', np.nan), errors='coerce')
            df_test['TotalCharges'] = df_test['TotalCharges'].fillna(df_test['TotalCharges'].median())
            
            X_test = df_test.drop(columns=['customerID', 'Churn'], errors='ignore')
            
            X_test_prep = preprocessor.transform(X_test)
            
            y_pred = model.predict(X_test_prep).astype(int)
            y_proba = model.predict_proba(X_test_prep)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            st.markdown("---")
            st.subheader(f"📈 Evaluation Metrics: {model_option}")
            
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_proba)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")
            
            st.markdown("---")
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("🎯 Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                st.pyplot(fig)
                
            with col_right:
                st.subheader("📄 Classification Report")
                report_dict = classification_report(y_true, y_pred, output_dict=True, target_names=['No Churn', 'Churn'])
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=4), width="stretch")

            st.markdown("---")
            st.subheader("🔮 Model Predictions Summary")
            df_results = df_test[['customerID', 'tenure', 'MonthlyCharges', 'Contract', 'Churn']].copy() if 'customerID' in df_test.columns else df_test.iloc[:, :5].copy()
            df_results['Actual Churn'] = pd.Series(y_true).map({1: 'Yes', 0: 'No'})
            df_results['Predicted Churn'] = pd.Series(y_pred).map({1: 'Yes', 0: 'No'})
            df_results['Churn Probability'] = np.round(y_proba, 4)
            st.dataframe(df_results.head(10), width="stretch")
else:
    st.info("👆 Please upload the `test_data.csv` file using the sidebar to view evaluation results.")
