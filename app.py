import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
X_test = joblib.load('X_test.pkl')

st.title("Bank Customer Churn Risk Calculator")
st.header("Enter Customer Details")

credit_score = st.slider("Credit Score", 300, 850, 650)
age = st.slider("Age", 18, 92, 35)
tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
balance = st.number_input("Account Balance", 0.0, 250000.0, 50000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 100000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict Churn Risk"):

    has_cr_card_val = 1 if has_cr_card == "Yes" else 0
    is_active_val = 1 if is_active == "Yes" else 0
    geography_germany = 1 if geography == "Germany" else 0
    geography_spain = 1 if geography == "Spain" else 0
    gender_male = 1 if gender == "Male" else 0

    # Step 1: raw input row, same column order as training (before engineering, before scaling)
    input_data = pd.DataFrame([[
        credit_score, age, tenure, balance, num_products,
        has_cr_card_val, is_active_val, salary,
        geography_germany, geography_spain, gender_male
    ]], columns=[
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Geography_Germany', 'Geography_Spain', 'Gender_Male'
    ])

    # Step 2: scale numeric columns FIRST, exactly like training did
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Step 3: compute engineered features from the ALREADY-SCALED values
    input_data['Balance_Salary_Ratio'] = input_data['Balance'] / (input_data['EstimatedSalary'] + 1)
    input_data['Engagement_Product'] = input_data['IsActiveMember'] * input_data['NumOfProducts']
    input_data['Age_Tenure'] = input_data['Age'] * input_data['Tenure']
    input_data['Product_Density'] = input_data['NumOfProducts'] / (input_data['Tenure'] + 1)

    # Step 4: predict
    churn_probability = model.predict_proba(input_data)[0][1]

    st.header("Result")
    st.metric("Churn Probability", f"{churn_probability:.1%}")

    if churn_probability > 0.5:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")

    st.progress(float(churn_probability))
        # Module 2: Probability Distribution Visualization
    st.header("How This Compares to Other Customers")

    import matplotlib.pyplot as plt

    # Get churn probabilities for the entire test set
    all_probabilities = model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots()
    ax.hist(all_probabilities, bins=30, color='skyblue', edgecolor='black')
    ax.axvline(churn_probability, color='red', linestyle='--', linewidth=2, label='This Customer')
    ax.set_xlabel('Churn Probability')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Where This Customer Falls Among All Test Customers')
    ax.legend()

    st.pyplot(fig)