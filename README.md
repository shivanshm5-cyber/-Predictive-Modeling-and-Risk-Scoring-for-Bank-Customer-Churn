# Bank Customer Churn Prediction

This project predicts which bank customers are likely to leave (churn), so the bank can take action before they actually leave.

## What is Churn?
Churn means a customer leaves the bank. Predicting churn early helps the bank offer better service or deals to keep them.

## Dataset
- 10,000 bank customers
- Includes info like age, gender, country, balance, and whether they churned

## Tools Used
- Python
- Pandas, NumPy (data handling)
- Matplotlib (charts)
- Scikit-learn (machine learning) — coming soon
- Streamlit (web app) — coming soon

## Status
Currently working on: Exploratory Data Analysis (EDA)

## What I Found So Far
- Only 20% of customers churned, 80% stayed
- Customers in Germany churn twice as much as France or Spain
- Inactive customers churn twice as much as active ones
- Older customers churn more than younger ones

## What's Next
- Finish EDA
- Prepare the data for machine learning
- Build and test prediction models
- Build a simple web app to show predictions

## What I Found So Far
- Number of products is the biggest churn driver — churn is lowest with 2 products (7.6%) but jumps to 82.7% with 3 products and 100% with 4 products
- Customers in Germany churn twice as much as customers in France or Spain
- Inactive customers churn nearly twice as much as active customers
- Older customers churn more than younger customers
- Female customers churn more than male customers
- Customers with higher account balances churn slightly more
- Credit score, tenure, salary, and having a credit card don't seem to affect churn much