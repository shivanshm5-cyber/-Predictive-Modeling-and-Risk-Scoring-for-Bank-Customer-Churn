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
- Matplotlib, Seaborn (charts)
- Scikit-learn (machine learning)
- SHAP (model explainability)
- Streamlit (web app)

## Status
Currently working on: Streamlit dashboard (Day 5)

## What I Found So Far
- Number of products is the biggest churn driver — churn is lowest with 2 products (7.6%) but jumps to 82.7% with 3 products and 100% with 4 products
- Customers in Germany churn twice as much as customers in France or Spain
- Inactive customers churn nearly twice as much as active customers
- Older customers churn more than younger customers
- Female customers churn more than male customers
- Customers with higher account balances churn slightly more
- Credit score, tenure, salary, and having a credit card don't seem to affect churn much

## Feature Engineering
Added 4 new features based on existing data:
- Balance-to-Salary ratio
- Engagement-Product interaction (activity level × number of products)
- Age-Tenure interaction
- Product Density (products relative to tenure)

These gave a small but real improvement in model performance, with Engagement-Product being the most useful of the four.

## Models Trained
Tested 4 models and compared their performance:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.808 | 0.589 | 0.187 | 0.284 | 0.577 |
| Decision Tree | 0.791 | 0.487 | 0.501 | 0.494 | 0.683 |
| Random Forest | 0.867 | 0.790 | 0.472 | 0.591 | 0.720 |
| **Gradient Boosting (Best)** | **0.8705** | **0.789** | **0.496** | **0.609** | **0.731** |

Gradient Boosting performed best overall and was selected as the final model.

## Model Explainability
Used feature importance and SHAP values to understand what drives predictions. The top churn drivers were:
1. Age
2. Number of Products
3. Active Membership status
4. Account Balance
5. Geography (Germany)

## What's Next
- Finish Streamlit dashboard (churn calculator, probability view, feature importance view, what-if simulator)
- Write research paper
- Write executive summary
- Final submission