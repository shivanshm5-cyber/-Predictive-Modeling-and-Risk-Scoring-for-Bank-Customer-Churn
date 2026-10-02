import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
X_test = joblib.load('X_test.pkl')
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

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
st.header("What Drives Churn? (Feature Importance)")

import pandas as pd

feature_importance = pd.DataFrame({
    'Feature': model.feature_names_in_,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

st.bar_chart(feature_importance.set_index('Feature'))    
st.header("What-If Scenario Simulator")
st.write("See how changing engagement and product count affects churn risk, keeping other details fixed.")

wi_age = st.slider("Age (what-if)", 18, 92, 40, key="wi_age")
wi_balance = st.number_input("Balance (what-if)", 0.0, 250000.0, 75000.0, key="wi_balance")
wi_salary = st.number_input("Salary (what-if)", 0.0, 200000.0, 90000.0, key="wi_salary")
wi_tenure = st.slider("Tenure (what-if)", 0, 10, 5, key="wi_tenure")
wi_credit = st.slider("Credit Score (what-if)", 300, 850, 650, key="wi_credit")

wi_num_products = st.slider("Number of Products (adjust this)", 1, 4, 2, key="wi_products")
wi_is_active = st.radio("Active Member? (adjust this)", ["Yes", "No"], key="wi_active")

wi_is_active_val = 1 if wi_is_active == "Yes" else 0

wi_input = pd.DataFrame([[
    wi_credit, wi_age, wi_tenure, wi_balance, wi_num_products,
    1, wi_is_active_val, wi_salary, 0, 0, 0
]], columns=[
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_Germany', 'Geography_Spain', 'Gender_Male'
])

wi_input[num_cols] = scaler.transform(wi_input[num_cols])

wi_input['Balance_Salary_Ratio'] = wi_input['Balance'] / (wi_input['EstimatedSalary'] + 1)
wi_input['Engagement_Product'] = wi_input['IsActiveMember'] * wi_input['NumOfProducts']
wi_input['Age_Tenure'] = wi_input['Age'] * wi_input['Tenure']
wi_input['Product_Density'] = wi_input['NumOfProducts'] / (wi_input['Tenure'] + 1)

wi_probability = model.predict_proba(wi_input)[0][1]

st.metric("What-If Churn Probability", f"{wi_probability:.1%}")
st.progress(float(wi_probability))
