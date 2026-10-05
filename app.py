import streamlit as st
import pandas as pd
import pickle


# ==========================================
# Load saved model files
# ==========================================

with open("knn_model.pkl", "rb") as file:
    knn = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("encoders.pkl", "rb") as file:
    encoders = pickle.load(file)


# ==========================================
# Load dataset
# ==========================================

df = pd.read_csv(
    "credit_card_fraud_2026 (1).csv"
)


# Remove transaction_id
if "transaction_id" in df.columns:
    df = df.drop(
        "transaction_id",
        axis=1
    )


# ==========================================
# Streamlit page
# ==========================================

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter the transaction details below "
    "to predict whether the transaction is "
    "fraudulent or not."
)


# ==========================================
# Create input fields
# ==========================================

user_input = {}

feature_columns = df.drop(
    "is_fraud",
    axis=1
).columns


for column in feature_columns:

    # ------------------------------
    # Categorical columns
    # ------------------------------

    if column in encoders:

        options = list(
            encoders[column].classes_
        )

        user_input[column] = st.selectbox(
            column,
            options
        )

    # ------------------------------
    # Numeric columns
    # ------------------------------

    else:

        user_input[column] = st.number_input(
            column,
            value=0.0
        )


# ==========================================
# Prediction
# ==========================================

if st.button("🔍 Predict Fraud"):

    # Convert input to DataFrame
    input_df = pd.DataFrame(
        [user_input]
    )


    # ======================================
    # Apply Label Encoding
    # ======================================

    for column in encoders:

        input_df[column] = encoders[
            column
        ].transform(
            input_df[column]
        )


    # ======================================
    # Apply Standard Scaling
    # ======================================

    input_scaled = scaler.transform(
        input_df
    )


    # ======================================
    # KNN Prediction
    # ======================================

    prediction = knn.predict(
        input_scaled
    )


    # ======================================
    # Display result
    # ======================================

    if prediction[0] == 1:

        st.error(
            "⚠️ Fraudulent Transaction"
        )

    else:

        st.success(
            "✅ Not a Fraudulent Transaction"
        )