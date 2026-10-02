
import streamlit as st
import pandas as pd
import plotly.express as px
# Page configuration
st.set_page_config(
    page_title="Financial Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Load dataset
df = pd.read_csv("data/raw/financial_fraud_detection_dataset.csv")
# Sidebar risk-level filter

st.sidebar.header("Dashboard Filters")

risk_filter = st.multiselect(
    "Select Risk Level",
    options=["Low", "Medium", "High", "Critical"],
    default=["High", "Critical"]
)

# Calculate KPIs
total_transactions = len(df)
fraud_transactions = int(df["Fraudulent"].sum())
fraud_rate = (fraud_transactions / total_transactions) * 100
fraud_amount = df.loc[df["Fraudulent"] == 1, "Transaction_Amount"].sum()

# Dashboard title
st.title("🔍 Financial Fraud Detection Dashboard")

st.markdown(
    """
    ### Financial Transaction Fraud Overview

    This dashboard provides insights into transaction fraud,
    fraud risk, and high-risk transaction patterns.
    """
)

# KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Fraud Transactions",
    f"{fraud_transactions:,}"
)

col3.metric(
    "Fraud Rate",
    f"{fraud_rate:.2f}%"
)

col4.metric(
    "Fraudulent Amount",
    f"{fraud_amount:,.2f}"
)
# Fraud distribution
# Fraud trend over time
st.subheader("Fraud Trend Over Time")

df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    format="%d-%m-%Y %H:%M"
)

monthly_fraud = (
    df.groupby(df["Transaction_Date"].dt.to_period("M"))["Fraudulent"]
    .mean()
    .mul(100)
    .reset_index()
)

monthly_fraud["Transaction_Date"] = monthly_fraud["Transaction_Date"].astype(str)

fig = px.line(
    monthly_fraud,
    x="Transaction_Date",
    y="Fraudulent",
    markers=True,
    labels={
        "Transaction_Date": "Month",
        "Fraudulent": "Fraud Rate (%)"
    },
    title="Monthly Fraud Rate"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Transaction Fraud Distribution")

fraud_distribution = pd.DataFrame({
    "Status": ["Non-Fraud", "Fraud"],
    "Transactions": [
        total_transactions - fraud_transactions,
        fraud_transactions
    ]
})

st.bar_chart(
    fraud_distribution.set_index("Status")["Transactions"]
)

# Fraud rate by merchant category

st.subheader("Fraud Rate by Merchant Category")

merchant_fraud = (
    df.groupby("Merchant_Category")["Fraudulent"]
    .mean()
    .mul(100)
    .round(2)
    .sort_values(ascending=False)
)

st.bar_chart(merchant_fraud)

# Key fraud indicators
# Fraud rate by payment method
st.subheader("Fraud Rate by Payment Method")

payment_fraud = (
    df.groupby("Payment_Method")["Fraudulent"]
    .mean()
    .mul(100)
    .round(2)
    .sort_values(ascending=False)
)

fig = px.bar(
    payment_fraud.reset_index(),
    x="Payment_Method",
    y="Fraudulent",
    labels={
        "Payment_Method": "Payment Method",
        "Fraudulent": "Fraud Rate (%)"
    },
    title=None
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Key Fraud Indicators")

col1, col2 = st.columns(2)

with col1:
    international_fraud = (
        df.groupby("Is_International")["Fraudulent"]
        .mean()
        .mul(100)
        .round(2)
    )

    international_fraud.index = [
        "Domestic",
        "International"
    ]

    st.bar_chart(international_fraud)

with col2:
    keyword_fraud = (
        df.groupby("Suspicious_Keyword")["Fraudulent"]
        .mean()
        .mul(100)
        .round(2)
    )

    st.bar_chart(keyword_fraud)

    # Model-based fraud risk distribution

st.subheader("Model-Based Fraud Risk Distribution")

risk_summary = pd.read_csv(
    "reports/risk_level_summary.csv"
)

risk_summary = risk_summary.set_index("Risk_Level")
risk_summary = risk_summary.reindex(["Low", "Medium", "High", "Critical"])

st.bar_chart(
    risk_summary["Transaction_Count"]
)

# Observed fraud rate by model risk level

st.subheader("Observed Fraud Rate by Risk Level")

risk_performance = pd.read_csv(
    "reports/risk_level_performance.csv"
)

risk_performance["Risk_Level"] = pd.Categorical(
    risk_performance["Risk_Level"],
    categories=["Low", "Medium", "High", "Critical"],
    ordered=True
)

risk_performance = risk_performance.sort_values("Risk_Level")

fig = px.bar(
    risk_performance,
    x="Risk_Level",
    y="Fraud_Rate",
    title=None
)

st.plotly_chart(fig, use_container_width=True)

# High-risk transaction review

st.subheader("High-Risk Transactions")

high_risk = pd.read_csv(
    "reports/high_risk_transactions.csv"
)

display_columns = [
    "Transaction_Amount",
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
    "Is_International",
    "Suspicious_Keyword",
    "Fraud_Probability",
    "Risk_Level"
]

filtered_high_risk = high_risk[
    high_risk["Risk_Level"].isin(risk_filter)
]

st.dataframe(
    filtered_high_risk[display_columns],
    use_container_width=True,
    hide_index=True
)

# Model performance

st.subheader("Fraud Detection Model Performance")

model_metrics = pd.read_csv(
    "reports/final_model_metrics.csv"
)

performance_columns = [
    "Model",
    "Threshold",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC",
    "PR_AUC"
]

st.dataframe(
    model_metrics[performance_columns],
    use_container_width=True,
    hide_index=True
)

# Auto reload test 2
