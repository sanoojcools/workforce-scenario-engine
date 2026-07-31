import pandas as pd
import sys
sys.path.append("..")
from src.attrition_model import compute_attrition_probability

def test_risk_range():
    df = pd.DataFrame({
        "performance_rating":["ME (Meets)"]*5,
        "tenure_months":[10,20,30,40,70],
        "department":["Sales"]*5,
        "level":["L4"]*5,
        "salary_lakhs":[20]*5,
        "promotion_readiness":["Not Ready"]*5,
        "gender":["M"]*5
    })
    out = compute_attrition_probability(df)
    assert out["attrition_probability"].between(0.02,0.85).all()
    assert "attrition_risk_bucket" in out.columns

def test_no_nan():
    df = pd.DataFrame({
        "performance_rating":["LE (Low)"],
        "tenure_months":[12],
        "department":["Engineering"],
        "level":["L3"],
        "salary_lakhs":[12],
        "promotion_readiness":["Ready Now"],
        "gender":["F"]
    })
    out = compute_attrition_probability(df)
    assert not out["attrition_probability"].isna().any()
