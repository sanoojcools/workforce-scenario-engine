import pandas as pd
import sys
sys.path.append("..")
from src.data_generator import generate_demo_data
from src.simulator import WorkforceSimulator

def test_scenario_runs():
    df = generate_demo_data(n=10, random_seed=1)
    sim = WorkforceSimulator(df)
    hist, final_df, sev, re = sim.run_scenario(
        planned_hires_per_month=2, promotion_rate=0.1, salary_inflation=5,
        attrition_multiplier=1.0, hire_dist={"L3":0.5,"L4":0.5},
        promo_bump_pct=10, backfill_delay=1, cost_per_hire_lakhs=1,
        promotion_cycle_months=6, restructuring=None
    )
    assert len(hist) == 12
    assert hist["monthly_cost_cr"].min() >= 0
