
import numpy as np
import pandas as pd

REFERENCE_DATE = pd.Timestamp("2026-01-01")

def create_credit_risk_features(data: pd.DataFrame) -> pd.DataFrame:
    # Return a feature-engineered copy; do not mutate the input.
    result = data.copy()
    for column in ["age", "annual_income", "loan_amount", "credit_score",
                   "transaction_count", "spending_ratio"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    income = result["annual_income"].where(result["annual_income"] > 0)
    transactions = result["transaction_count"].where(result["transaction_count"] > 0)
    result["loan_to_income_ratio"] = result["loan_amount"] / income
    result["income_per_transaction"] = result["annual_income"] / transactions
    result["customer_tenure_days"] = (
        REFERENCE_DATE - pd.to_datetime(result["join_date"], errors="coerce")
    ).dt.days
    result.loc[result["customer_tenure_days"] < 0, "customer_tenure_days"] = np.nan
    result["age_band"] = pd.cut(
        result["age"], bins=[0, 25, 35, 50, 65, np.inf],
        labels=["18-25", "26-35", "36-50", "51-65", "66+"]
    )
    result["repayment_risk_level"] = result["repayment_history"].map(
        {"Excellent": 0, "Good": 1, "Fair": 2, "Poor": 3}
    )
    return result
