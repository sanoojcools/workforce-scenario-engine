
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Workforce Scenario Engine", layout="wide")

# ============================================================
# 1. ATTRITION MODEL (Compute for uploaded CSVs)
# ============================================================
def compute_attrition_probability(df):
    """Compute attrition_probability and risk_bucket for any workforce dataframe."""
    df = df.copy()

    # Base risk
    base_risk = 0.15
    risk = np.full(len(df), base_risk)

    # Performance
    perf_map = {
        "LE (Low)": 0.25,
        "ME (Meets)": 0.05,
        "EE (Exceeds)": -0.03,
        "GE (Greatly Exceeds)": -0.08
    }
    for perf, delta in perf_map.items():
        risk += np.where(df["performance_rating"] == perf, delta, 0)

    # Tenure (months)
    tenure = df["tenure_months"].astype(float)
    risk += np.where((tenure >= 12) & (tenure <= 24), 0.12, 0)
    risk += np.where(tenure < 6, 0.08, 0)
    risk += np.where(tenure > 60, -0.10, 0)

    # Department
    dept = df["department"].astype(str).str.lower()
    risk += np.where(dept.str.contains("sales|operations"), 0.08, 0)
    risk += np.where(dept.str.contains("engineer|tech|rd"), 0.05, 0)

    # Level
    level = df["level"].astype(str)
    risk += np.where(level.isin(["L4", "L5"]), 0.06, 0)

    # Salary compression vs level median
    level_medians = df.groupby("level")["salary_lakhs"].transform("median")
    ratio = df["salary_lakhs"] / level_medians
    risk += np.where(ratio < 0.85, 0.10, 0)
    risk += np.where(ratio > 1.20, -0.08, 0)

    # Promotion readiness
    ready_map = {
        "Ready Now": 0.07,
        "Ready 1-2Y": 0.02,
        "Ready 2+Y": 0.0,
        "Not Ready": 0.0
    }
    for ready, delta in ready_map.items():
        risk += np.where(df["promotion_readiness"] == ready, delta, 0)

    # Gender x Engineering
    gender = df["gender"].astype(str).str.upper()
    risk += np.where((gender == "F") & (dept.str.contains("engineer|tech|rd")), 0.03, 0)

    # Clip
    df["attrition_probability"] = np.clip(risk, 0.02, 0.85)

    # Risk bucket
    conditions = [
        df["attrition_probability"] < 0.10,
        df["attrition_probability"] < 0.20,
        df["attrition_probability"] < 0.35,
        df["attrition_probability"] >= 0.35
    ]
    choices = ["Low", "Medium", "High", "Critical"]
    df["attrition_risk_bucket"] = np.select(conditions, choices, default="Medium")

    return df

# ============================================================
# 2. DEMO DATA GENERATOR
# ============================================================
@st.cache_data
def generate_demo_data(n=500, random_seed=42):
    np.random.seed(random_seed)
    levels = np.random.choice(["L3","L4","L5","L6","L7","L8"], n, p=[0.36,0.30,0.17,0.10,0.06,0.02])
    depts = np.random.choice(["Engineering","Finance","HR","Sales","Product","Operations"], n, p=[0.18,0.19,0.17,0.16,0.16,0.14])
    tenure = np.clip(np.random.gamma(3, 12, n), 0, 180).astype(int)
    perf = np.random.choice(["LE (Low)","ME (Meets)","EE (Exceeds)","GE (Greatly Exceeds)"], n, p=[0.12,0.54,0.28,0.07])
    age = np.clip(np.random.normal(32, 7, n).astype(int), 22, 60)
    gender = np.random.choice(["M","F","NB"], n, p=[0.60,0.38,0.02])
    readiness = np.random.choice(["Not Ready","Ready Now","Ready 1-2Y","Ready 2+Y"], n, p=[0.15,0.24,0.37,0.25])

    # Salary by level with performance multiplier
    level_base = {"L3":12,"L4":22,"L5":40,"L6":70,"L7":100,"L8":200}
    perf_mult = {"LE (Low)":0.85,"ME (Meets)":1.0,"EE (Exceeds)":1.15,"GE (Greatly Exceeds)":1.30}
    salary = np.array([level_base[l] * np.random.uniform(0.8,1.2) * perf_mult[p] for l,p in zip(levels, perf)])

    df = pd.DataFrame({
        "employee_id": [f"EMP_{i+1:04d}" for i in range(n)],
        "level": levels,
        "department": depts,
        "tenure_months": tenure,
        "performance_rating": perf,
        "salary_lakhs": np.round(salary, 2),
        "age": age,
        "gender": gender,
        "promotion_readiness": readiness
    })
    df = compute_attrition_probability(df)
    df["manager_id"] = -1
    return df

# ============================================================
# 3. SIMULATION ENGINE
# ============================================================
class WorkforceSimulator:
    def __init__(self, baseline_df):
        self.baseline = baseline_df.copy()
        self.months = 12

    def run_scenario(self, planned_hires_per_month, promotion_rate, salary_inflation,
                     attrition_multiplier, hire_dist, promo_bump_pct, backfill_delay,
                     cost_per_hire_lakhs, promotion_cycle_months):

        current = self.baseline.copy()
        history = []
        backfill_queue = {}

        # Normalize hire distribution
        hire_total = sum(hire_dist.values())
        if hire_total == 0:
            hire_dist = {"L3":0.41,"L4":0.33,"L5":0.21,"L6":0.05}
        else:
            hire_dist = {k:v/hire_total for k,v in hire_dist.items()}

        for month in range(1, self.months + 1):
            # Attrition
            current["monthly_attrition_prob"] = current["attrition_probability"] / 12 * attrition_multiplier
            current["leaves_this_month"] = np.random.binomial(1, current["monthly_attrition_prob"])
            leavers = current[current["leaves_this_month"] == 1].copy()
            current = current[current["leaves_this_month"] == 0].copy()

            # Queue backfills
            n_backfill = len(leavers)
            if n_backfill > 0 and backfill_delay >= 0:
                backfill_queue[month + backfill_delay] = backfill_queue.get(month + backfill_delay, 0) + n_backfill

            # Promotions (every N months)
            n_promote = 0
            if month % promotion_cycle_months == 0:
                eligible = current[
                    (current["promotion_readiness"].isin(["Ready Now", "Ready 1-2Y"])) &
                    (current["performance_rating"].isin(["EE (Exceeds)", "GE (Greatly Exceeds)"])) &
                    (current["level"] != "L8")
                ]
                n_promote = int(len(eligible) * promotion_rate)
                if n_promote > 0 and len(eligible) > 0:
                    promote_idx = eligible.sample(min(n_promote, len(eligible)), random_state=month).index
                    # Level up
                    level_order = ["L3","L4","L5","L6","L7","L8"]
                    current.loc[promote_idx, "level"] = current.loc[promote_idx, "level"].apply(
                        lambda x: level_order[min(level_order.index(x)+1, len(level_order)-1)]
                    )
                    current.loc[promote_idx, "salary_lakhs"] *= (1 + promo_bump_pct/100)
                    current.loc[promote_idx, "promotion_readiness"] = "Not Ready"

            # Planned hires
            n_hire = planned_hires_per_month
            # Backfills arriving this month
            n_hire += backfill_queue.get(month, 0)

            if n_hire > 0:
                hire_levels = np.random.choice(list(hire_dist.keys()), n_hire, p=list(hire_dist.values()))
                new_hires = pd.DataFrame({
                    "employee_id": [f"HIRE_{month}_{i}" for i in range(n_hire)],
                    "level": hire_levels,
                    "department": np.random.choice(current["department"].unique(), n_hire),
                    "tenure_months": np.random.randint(1, 12, n_hire),
                    "performance_rating": np.random.choice(["ME (Meets)","EE (Exceeds)","LE (Low)"], n_hire, p=[0.6,0.3,0.1]),
                    "salary_lakhs": [{"L3":12,"L4":22,"L5":40,"L6":70}.get(l,12)*np.random.uniform(0.9,1.1) for l in hire_levels],
                    "age": np.random.randint(24, 34, n_hire),
                    "gender": np.random.choice(["M","F","NB"], n_hire, p=[0.60,0.38,0.02]),
                    "promotion_readiness": ["Not Ready"]*n_hire,
                    "attrition_probability": 0.15,
                    "attrition_risk_bucket": "Medium",
                    "manager_id": -1
                })
                current = pd.concat([current, new_hires], ignore_index=True)

            # Salary inflation
            current["salary_lakhs"] *= (1 + salary_inflation/12/100)

            # Cost = payroll + hire cost
            monthly_payroll = current["salary_lakhs"].sum() / 100  # Cr
            hire_cost = n_hire * cost_per_hire_lakhs / 100  # Cr
            total_cost = monthly_payroll + hire_cost

            history.append({
                "month": month,
                "headcount": len(current),
                "monthly_cost_cr": round(total_cost, 2),
                "payroll_cr": round(monthly_payroll, 2),
                "hire_cost_cr": round(hire_cost, 2),
                "attrition": len(leavers),
                "hires": n_hire,
                "promotions": n_promote
            })

        return pd.DataFrame(history), current

# ============================================================
# 4. TORNADO ANALYSIS
# ============================================================
def tornado_analysis(sim, base_params, var_name, var_range):
    results = []
    for val in var_range:
        params = base_params.copy()
        params[var_name] = val
        hist, _ = sim.run_scenario(**params)
        results.append({
            var_name: val,
            "final_headcount": hist["headcount"].iloc[-1],
            "final_cost": hist["monthly_cost_cr"].iloc[-1]
        })
    return pd.DataFrame(results)

# ============================================================
# 5. AI INSIGHTS
# ============================================================
def generate_insights(baseline, scenario_hist, final_df, params):
    insights = []

    # Insight 1: Headcount trajectory
    net_growth = scenario_hist["headcount"].iloc[-1] - len(baseline)
    if net_growth > 100:
        insights.append(("🔥 Aggressive Growth", f"Net +{net_growth} heads in 12 months. Ensure leadership bench (L6+) scales proportionally.", "High"))
    elif net_growth < 0:
        insights.append(("⚠️ Shrinking Workforce", f"Net {net_growth} heads. Review critical role coverage immediately.", "Critical"))
    else:
        insights.append(("📊 Moderate Growth", f"Net +{net_growth} heads. Sustainable trajectory.", "Medium"))

    # Insight 2: Cost
    cost_growth = (scenario_hist["monthly_cost_cr"].iloc[-1] / scenario_hist["monthly_cost_cr"].iloc[0] - 1) * 100
    insights.append(("💰 Cost Trajectory", f"Monthly cost up {cost_growth:.1f}%. {(params['planned_hires_per_month']*12)} hires drive {(cost_growth*0.6):.0f}% of this.", "High" if cost_growth > 30 else "Medium"))

    # Insight 3: Attrition
    total_attrition = scenario_hist["attrition"].sum()
    attr_rate = total_attrition / len(baseline) * 100
    insights.append(("🚪 Attrition Alert", f"{total_attrition} exits ({attr_rate:.1f}% annualized). At {params['attrition_multiplier']}x multiplier, this is {'above' if params['attrition_multiplier'] > 1 else 'below'} baseline.", "Critical" if attr_rate > 25 else "High" if attr_rate > 15 else "Medium"))

    # Insight 4: Promotion pipeline
    ready_now = len(baseline[baseline["promotion_readiness"] == "Ready Now"])
    total_promos = scenario_hist["promotions"].sum()
    ratio = ready_now / max(total_promos, 1)
    if ratio > 10:
        insights.append(("🎯 Promotion Chokepoint", f"{ready_now} Ready Now → {total_promos} projected promotions ({ratio:.0f}:1 ratio). Flight risk for top performers is elevated.", "Critical"))
    else:
        insights.append(("✅ Promotion Flow", f"Healthy {ratio:.1f}:1 ready-to-promoted ratio.", "Low"))

    # Insight 5: Department risk
    dept_risk = baseline.groupby("department").apply(lambda x: (x["attrition_risk_bucket"].isin(["High","Critical"])).mean()*100).sort_values(ascending=False)
    if len(dept_risk) > 0 and dept_risk.iloc[0] > 60:
        insights.append(("🏢 Department Risk", f"{dept_risk.index[0]} shows {dept_risk.iloc[0]:.0f}% High/Critical flight risk. Targeted retention needed.", "Critical"))

    # Insight 6: Level pyramid
    l5_plus = (final_df["level"].isin(["L5","L6","L7","L8"])).mean()*100
    insights.append(("🏗️ Leadership Density", f"L5+ representation: {l5_plus:.1f}%. {'Dilution risk' if l5_plus < 25 else 'Healthy'} as headcount scales.", "High" if l5_plus < 25 else "Medium"))

    return insights

# ============================================================
# 6. STREAMLIT UI
# ============================================================
st.title("🧮 Workforce Scenario Engine")
st.markdown("Predictive org modeling with Monte Carlo simulation & tornado sensitivity analysis")

# Initialize session state
for key, val in {
    "planned_hires": 8,
    "promotion_rate": 0.09,
    "salary_inflation": 6.0,
    "attrition_multiplier": 1.0,
    "ai_provider": "Stochastic (Rule-Based)",
    "scenario_preset": "Custom",
    "last_preset": "Custom",
    "hire_l3": 41, "hire_l4": 33, "hire_l5": 21, "hire_l6": 5,
    "promo_bump": 20,
    "backfill_delay": 1,
    "cost_per_hire": 3.0,
    "promotion_cycle": 3
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---- SIDEBAR ----
with st.sidebar:
    st.header("🤖 AI Insights")
    ai_provider = st.selectbox("Provider", ["Stochastic (Rule-Based)", "Kimi", "Claude"], key="ai_provider")

    st.header("🎯 Scenario Preset")
    preset = st.selectbox("Preset", ["Custom", "Status Quo", "Aggressive Growth", "Cost Optimization"], key="scenario_preset")

    if preset != st.session_state.last_preset:
        if preset == "Status Quo":
            st.session_state.planned_hires = 8
            st.session_state.promotion_rate = 0.09
            st.session_state.salary_inflation = 6.0
            st.session_state.attrition_multiplier = 1.0
        elif preset == "Aggressive Growth":
            st.session_state.planned_hires = 20
            st.session_state.promotion_rate = 0.12
            st.session_state.salary_inflation = 8.0
            st.session_state.attrition_multiplier = 1.1
        elif preset == "Cost Optimization":
            st.session_state.planned_hires = 2
            st.session_state.promotion_rate = 0.05
            st.session_state.salary_inflation = 3.0
            st.session_state.attrition_multiplier = 0.85
        st.session_state.last_preset = preset
        st.rerun()

    st.header("⚙️ Simulation Parameters")
    planned_hires = st.slider("Planned Hires / Month", 0, 50, st.session_state.planned_hires, key="planned_hires")
    promotion_rate = st.slider("Promotion Rate", 0.0, 0.30, st.session_state.promotion_rate, step=0.01, key="promotion_rate")
    salary_inflation = st.slider("Salary Inflation % (Annual)", 0.0, 20.0, st.session_state.salary_inflation, step=0.5, key="salary_inflation")
    attrition_multiplier = st.slider("Attrition Multiplier", 0.5, 1.5, st.session_state.attrition_multiplier, step=0.05, key="attrition_multiplier")

    st.header("🏗️ Adjustable Assumptions")
    hire_l3 = st.slider("Hire % L3", 0, 100, st.session_state.hire_l3, key="hire_l3")
    hire_l4 = st.slider("Hire % L4", 0, 100, st.session_state.hire_l4, key="hire_l4")
    hire_l5 = st.slider("Hire % L5", 0, 100, st.session_state.hire_l5, key="hire_l5")
    hire_l6 = st.slider("Hire % L6", 0, 100, st.session_state.hire_l6, key="hire_l6")
    promo_bump = st.slider("Promotion Salary Bump %", 10, 50, st.session_state.promo_bump, key="promo_bump")
    backfill_delay = st.slider("Backfill Delay (months)", 0, 6, st.session_state.backfill_delay, key="backfill_delay")
    cost_per_hire = st.slider("Cost Per Hire (₹ Lakhs)", 0.0, 10.0, st.session_state.cost_per_hire, step=0.5, key="cost_per_hire")
    promotion_cycle = st.selectbox("Promotion Cycle (months)", [1,2,3,6,12], index=[1,2,3,6,12].index(st.session_state.promotion_cycle), key="promotion_cycle")

    # Validate hire distribution
    hire_total_pct = hire_l3 + hire_l4 + hire_l5 + hire_l6
    if hire_total_pct != 100:
        st.warning(f"Hire distribution sums to {hire_total_pct}%. Normalizing...")
        if hire_total_pct > 0:
            hire_l3, hire_l4, hire_l5, hire_l6 = int(hire_l3/hire_total_pct*100), int(hire_l4/hire_total_pct*100), int(hire_l5/hire_total_pct*100), int(hire_l6/hire_total_pct*100)

    st.header("📁 Data Source")
    use_demo = st.checkbox("Use Demo Data (500 employees)", value=True)
    uploaded_file = st.file_uploader("Upload Workforce CSV", type="csv")

    if uploaded_file is not None:
        use_demo = False

# ---- LOAD DATA ----
if use_demo:
    df = generate_demo_data()
    st.sidebar.success("Loaded 500 demo employees")
else:
    try:
        df = pd.read_csv(uploaded_file)
        # Compute attrition for uploaded data
        df = compute_attrition_probability(df)
        df["manager_id"] = df.get("manager_id", -1)
        st.sidebar.success(f"Loaded {len(df)} employees")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

# ---- BASELINE SUMMARY ----
st.subheader("📋 Baseline Workforce Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Employees", len(df))
c2.metric("Monthly Payroll", f"₹{df['salary_lakhs'].sum()/100:.2f} Cr")
c3.metric("High/Critical Risk", f"{(df['attrition_risk_bucket'].isin(['High','Critical'])).sum()} ({(df['attrition_risk_bucket'].isin(['High','Critical'])).mean()*100:.0f}%)")
c4.metric("Ready Now", (df["promotion_readiness"]=="Ready Now").sum())

c5, c6, c7, c8 = st.columns(4)
c5.metric("Largest Dept", df["department"].value_counts().index[0])
c6.metric("Median Tenure", f"{df['tenure_months'].median():.0f} mo")
c7.metric("Female %", f"{(df['gender']=='F').mean()*100:.0f}%")
c8.metric("Avg Age", f"{df['age'].mean():.0f}")

# ---- MODEL ASSUMPTIONS ----
with st.expander("⚙️ Model Assumptions & Methodology", expanded=False):
    assumptions = [
        ("👥 Workforce", "Every exit is replaced by a new hire (backfill). Empty seats are filled after the backfill delay.", "If seats stay empty, cost drops more than projected."),
        ("📉 Replacement", "Leavers are backfilled at the hire level distribution (L3-L6), not necessarily same-level.", "L5→L4 backfill creates apparent 'savings' that ignore knowledge loss."),
        ("💰 Cost", "Cost per hire is added to monthly trajectory. No recruitment fees, onboarding, or ramp-time productivity loss modeled.", "Real cost is 15-25% higher than modeled."),
        ("📈 Salary", "Uniform monthly inflation applied to all salaries. No pay-for-performance or compa-ratio adjustments.", "High performers may inflate faster; low performers slower."),
        ("🎯 Promotion", "Promotions occur every N months for Ready Now / Ready 1-2Y with EE/GE ratings. Bump is uniform %.", "Does not account for open headcount or business need."),
        ("🚪 Attrition", "Monthly attrition = annual probability ÷ 12 × multiplier. Independent across employees.", "Ignores attrition cascades and seasonality (post-bonus spikes)."),
        ("⏱️ Horizon", "12-month projection window. All scenarios terminate at month 12.", "Misses 18-month promotion bottlenecks and long-term pipeline health."),
        ("🏢 Department", "Sales/Operations +8% risk, Engineering +5% risk. Gender×Eng interaction +3%.", "Other department interactions not modeled."),
        ("📊 Performance", "LE +25%, ME +5%, EE -3%, GE -8% attrition risk. Performance is static over 12 months.", "Real-world performance changes quarterly."),
        ("🔢 Tenure", "Peak risk at 12-24 months (+12%). <6 months +8%. >60 months -10%.", "Industry/role-specific tenure curves may differ."),
        ("💵 Compression", "Salary <85% of level median → +10% risk. >120% → -8%.", "Median is computed from current snapshot only."),
        ("🎲 Stochastic", "Monte Carlo with single-run display. No confidence intervals or distribution shown.", "Rerun for variance; add multi-run for P10/P90."),
        ("🔄 Lateral", "No lateral moves modeled. Only up, out, or stay.", "Real orgs have 10-20% lateral movement."),
        ("🌍 Market", "Hiring yield always achievable. No talent shortage or market shock modeled.", "Talent shortages may require 20% salary premiums."),
    ]
    for icon_title, desc, impact in assumptions:
        st.markdown(f"**{icon_title}** — {desc}")
        st.caption(f"⚠️ Impact if false: {impact}")
        st.divider()
    st.info("All projections are directional estimates. Validate against historical ground truth before board presentation.")

# ---- RUN SIMULATION ----
hire_dist = {"L3": hire_l3, "L4": hire_l4, "L5": hire_l5, "L6": hire_l6}

sim = WorkforceSimulator(df)
params = {
    "planned_hires_per_month": planned_hires,
    "promotion_rate": promotion_rate,
    "salary_inflation": salary_inflation,
    "attrition_multiplier": attrition_multiplier,
    "hire_dist": hire_dist,
    "promo_bump_pct": promo_bump,
    "backfill_delay": backfill_delay,
    "cost_per_hire_lakhs": cost_per_hire,
    "promotion_cycle_months": promotion_cycle
}

# Run main scenario
np.random.seed(42)
hist, final_df = sim.run_scenario(**params)

# Run baseline for comparison
np.random.seed(42)
base_params = params.copy()
base_params["planned_hires_per_month"] = 0
base_params["attrition_multiplier"] = 1.0
base_params["salary_inflation"] = 0.0
hist_base, _ = sim.run_scenario(**base_params)

# ---- METRICS ----
st.subheader("📊 Scenario Metrics")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Final Headcount", hist["headcount"].iloc[-1], delta=int(hist["headcount"].iloc[-1] - len(df)))
m2.metric("Monthly Cost", f"₹{hist['monthly_cost_cr'].iloc[-1]:.2f} Cr")
m3.metric("Total Attrition", hist["attrition"].sum())
m4.metric("Total Hires", hist["hires"].sum())
m5.metric("Total Promotions", hist["promotions"].sum())

# ---- CHARTS ----
st.subheader("📈 Trajectory Charts")
col1, col2 = st.columns(2)

with col1:
    fig_hc = go.Figure()
    fig_hc.add_trace(go.Scatter(x=hist["month"], y=hist["headcount"], mode="lines+markers", name="Scenario", line=dict(color="#1f77b4", width=3)))
    fig_hc.add_trace(go.Scatter(x=hist_base["month"], y=hist_base["headcount"], mode="lines", name="Baseline (No hires)", line=dict(color="gray", dash="dash")))
    fig_hc.update_layout(title="Headcount Trajectory", xaxis_title="Month", yaxis_title="Headcount", height=350)
    st.plotly_chart(fig_hc, use_container_width=True)

with col2:
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(x=hist["month"], y=hist["monthly_cost_cr"], mode="lines+markers", name="Total Cost", line=dict(color="#ff7f0e", width=3)))
    fig_cost.add_trace(go.Scatter(x=hist["month"], y=hist["payroll_cr"], mode="lines", name="Payroll", line=dict(color="#2ca02c")))
    fig_cost.add_trace(go.Scatter(x=hist["month"], y=hist["hire_cost_cr"], mode="lines", name="Hire Cost", line=dict(color="#d62728")))
    fig_cost.update_layout(title="Cost Trajectory (₹ Cr)", xaxis_title="Month", yaxis_title="₹ Crores", height=350)
    st.plotly_chart(fig_cost, use_container_width=True)

# ---- TORNADO ----
st.subheader("🌪️ Tornado Sensitivity Analysis")
tc1, tc2 = st.columns(2)

with tc1:
    st.markdown("**Headcount Sensitivity**")
    tornado_vars = {
        "planned_hires_per_month": [max(0, planned_hires-5), planned_hires, planned_hires+5],
        "attrition_multiplier": [max(0.5, attrition_multiplier-0.2), attrition_multiplier, min(1.5, attrition_multiplier+0.2)],
        "promotion_rate": [max(0, promotion_rate-0.03), promotion_rate, min(0.3, promotion_rate+0.03)],
        "salary_inflation": [max(0, salary_inflation-3), salary_inflation, min(20, salary_inflation+3)]
    }

    tornado_hc = []
    for var, vals in tornado_vars.items():
        r = tornado_analysis(sim, params, var, vals)
        low = r["final_headcount"].min() - hist["headcount"].iloc[-1]
        high = r["final_headcount"].max() - hist["headcount"].iloc[-1]
        tornado_hc.append({"Variable": var.replace("_"," ").title(), "Low": low, "High": high})

    td_hc = pd.DataFrame(tornado_hc)
    fig_t1 = go.Figure()
    for _, row in td_hc.iterrows():
        fig_t1.add_trace(go.Bar(name=row["Variable"], y=[row["Variable"]], x=[row["High"]], orientation="h", marker_color="#1f77b4"))
        fig_t1.add_trace(go.Bar(name=row["Variable"], y=[row["Variable"]], x=[row["Low"]], orientation="h", marker_color="#d62728", showlegend=False))
    fig_t1.update_layout(barmode="overlay", xaxis_title="Impact on Final Headcount", height=300, showlegend=False)
    st.plotly_chart(fig_t1, use_container_width=True)

with tc2:
    st.markdown("**Cost Sensitivity**")
    tornado_cost = []
    for var, vals in tornado_vars.items():
        r = tornado_analysis(sim, params, var, vals)
        low = r["final_cost"].min() - hist["monthly_cost_cr"].iloc[-1]
        high = r["final_cost"].max() - hist["monthly_cost_cr"].iloc[-1]
        tornado_cost.append({"Variable": var.replace("_"," ").title(), "Low": low, "High": high})

    td_c = pd.DataFrame(tornado_cost)
    fig_t2 = go.Figure()
    for _, row in td_c.iterrows():
        fig_t2.add_trace(go.Bar(name=row["Variable"], y=[row["Variable"]], x=[row["High"]], orientation="h", marker_color="#2ca02c"))
        fig_t2.add_trace(go.Bar(name=row["Variable"], y=[row["Variable"]], x=[row["Low"]], orientation="h", marker_color="#ff7f0e", showlegend=False))
    fig_t2.update_layout(barmode="overlay", xaxis_title="Impact on Final Cost (₹ Cr)", height=300, showlegend=False)
    st.plotly_chart(fig_t2, use_container_width=True)

# ---- LEVEL DISTRIBUTION & ATTRITION CURVE ----
st.subheader("📉 Workforce Composition")
co1, co2 = st.columns(2)

with co1:
    level_counts = final_df["level"].value_counts().reindex(["L3","L4","L5","L6","L7","L8"]).fillna(0)
    fig_donut = go.Figure(data=[go.Pie(labels=level_counts.index, values=level_counts.values, hole=0.4)])
    fig_donut.update_layout(title="Final Level Distribution", height=350)
    st.plotly_chart(fig_donut, use_container_width=True)

with co2:
    fig_attr = go.Figure()
    fig_attr.add_trace(go.Bar(x=hist["month"], y=hist["attrition"], name="Monthly Attrition", marker_color="#d62728"))
    fig_attr.add_trace(go.Scatter(x=hist["month"], y=hist["attrition"].cumsum(), name="Cumulative", mode="lines+markers", yaxis="y2", line=dict(color="#ff7f0e")))
    fig_attr.update_layout(title="Attrition Curve", xaxis_title="Month", yaxis_title="Monthly Exits", yaxis2=dict(title="Cumulative", overlaying="y", side="right"), height=350)
    st.plotly_chart(fig_attr, use_container_width=True)

# ---- AI INSIGHTS ----
st.subheader("🤖 Strategic Insights")
insights = generate_insights(df, hist, final_df, params)

for title, text, severity in insights:
    color = {"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}.get(severity, "⚪")
    with st.container():
        st.markdown(f"{color} **{title}** — {text}")

# ---- EXECUTIVE SUMMARY ----
st.subheader("📝 Executive Summary")
total_cost_delta = hist["monthly_cost_cr"].iloc[-1] - hist["monthly_cost_cr"].iloc[0]
st.markdown(f"""
Over 12 months, this scenario projects **{hist['headcount'].iloc[-1]} employees** (net {'+' if hist['headcount'].iloc[-1] > len(df) else ''}{hist['headcount'].iloc[-1] - len(df)}), 
with a monthly cost of **₹{hist['monthly_cost_cr'].iloc[-1]:.2f} Cr** ({'+' if total_cost_delta > 0 else ''}₹{total_cost_delta:.2f} Cr). 
**{hist['attrition'].sum()} exits** are expected against **{hist['hires'].sum()} hires** ({hist['promotions'].sum()} promotions). 
The biggest levers are hiring volume and attrition containment.
""")

# ---- DOWNLOADS ----
st.subheader("⬇️ Export")
dc1, dc2 = st.columns(2)
with dc1:
    csv = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Final Workforce CSV", csv, "workforce_projection.csv", "text/csv")
with dc2:
    summary = {
        "parameters": params,
        "baseline_headcount": int(len(df)),
        "final_headcount": int(hist["headcount"].iloc[-1]),
        "total_attrition": int(hist["attrition"].sum()),
        "total_hires": int(hist["hires"].sum()),
        "total_promotions": int(hist["promotions"].sum()),
        "final_monthly_cost_cr": float(hist["monthly_cost_cr"].iloc[-1])
    }
    st.download_button("Download Scenario JSON", json.dumps(summary, indent=2), "scenario_summary.json", "application/json")
