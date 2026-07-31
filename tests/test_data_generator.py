import sys
sys.path.append("..")
from src.data_generator import generate_demo_data

def test_generate():
    df = generate_demo_data(n=100, random_seed=42)
    assert len(df) == 100
    assert "employee_id" in df.columns
    assert df["employee_id"].nunique() == 100

def test_columns():
    df = generate_demo_data(n=10)
    required = ["level","department","tenure_months","performance_rating","salary_lakhs"]
    for c in required:
        assert c in df.columns
