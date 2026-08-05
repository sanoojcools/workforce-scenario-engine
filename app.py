
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, warnings, re, os
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(page_title="Workforce Scenario Engine v5.0 | 1+2+5", layout="wide", page_icon="🧬")

MONTE_CARLO_RUNS = 1000
RANDOM_SEED_BASE = 10000
MAX_UPLOAD_MB = 10
REQUIRED_COLS = ["level","department","tenure_months","performance_rating","salary_lakhs","promotion_readiness"]

st.info("🔒 **v5.0 Privacy First:** Synthetic demo data. Do not upload real PII. Features: Survival Attrition + Supply Chain + Pareto Frontier", icon="🛡️")

# ========== CORE ATTRITION - SURVIVAL ENHANCED (Feature 1) ==========
def compute_attrition_probability(df):
    df = df.copy()
    # Baseline hazard via Weibull shape for tenure
    risk = np.full(len(df), 0.12)  # baseline lower, will be lifted by survival
    # Performance hazard ratios (Cox-like)
    perf_map = {"LE (Low)":0.35,"ME (Meets)":0.06,"EE (Exceeds)":-0.05,"GE (Greatly Exceeds)":-0.12}
    for perf, delta in perf_map.items():
        risk += np.where(df["performance_rating"]==perf, delta, 0)
    tenure = df["tenure_months"].astype(float)
    # Weibull hazard: peak at 12-24m
    weibull_hazard = 0.15 * np.exp(-0.5*((np.log((tenure+1)/18))**2))  # log-normal peak
    risk += weibull_hazard
    risk += np.where(tenure<6, 0.06, 0)
    risk += np.where(tenure>60, -0.12, 0)
    dept = df["department"].astype(str).str.lower()
    risk += np.where(dept.str.contains("sales|operations"), 0.08, 0)
    risk += np.where(dept.str.contains("engineer|tech|rd"), 0.05, 0)
    level = df["level"].astype(str)
    risk += np.where(level.isin(["L4","L5"]), 0.06, 0)
    med = df.groupby("level")["salary_lakhs"].transform("median").replace(0, np.nan)
    ratio = df["salary_lakhs"]/med
    df["compa_ratio"] = ratio
    risk += np.where(ratio<0.85, 0.12, 0)
    risk += np.where(ratio>1.20, -0.08, 0)
    ready_map = {"Ready Now":0.09,"Ready 1-2Y":0.02,"Ready 2+Y":0.0,"Not Ready":0.0}
    for ready, delta in ready_map.items():
        risk += np.where(df["promotion_readiness"]==ready, delta, 0)
    gender = df.get("gender", pd.Series(["M"]*len(df))).astype(str).str.upper()
    risk += np.where((gender=="F")&(dept.str.contains("engineer|tech|rd")), 0.03, 0)
    # Manager attrition contagion placeholder
    df["attrition_probability"] = np.clip(risk, 0.02, 0.80)
    # Survival curve params per employee for deeper math
    df["survival_scale"] = 12 + (1 - df["attrition_probability"])*24
    df["survival_shape"] = 1.5 + df["attrition_probability"]
    # Regrettable flag
    df["is_regrettable"] = df["performance_rating"].isin(["EE (Exceeds)","GE (Greatly Exceeds)"]) & (df["promotion_readiness"].isin(["Ready Now","Ready 1-2Y"]))
    conds = [df["attrition_probability"]<0.10, df["attrition_probability"]<0.20, df["attrition_probability"]<0.35, df["attrition_probability"]>=0.35]
    df["attrition_risk_bucket"] = np.select(conds, ["Low","Medium","High","Critical"], default="Medium")
    return df

@st.cache_data
def generate_demo_data(n=500, random_seed=42):
    np.random.seed(random_seed)
    def norm_p(p):
        p = np.array(p, dtype=float)
        return p / p.sum()
    levels = np.random.choice(["L3","L4","L5","L6","L7","L8"], n, p=norm_p([0.36,0.30,0.17,0.10,0.06,0.02]))
    depts = np.random.choice(["Engineering","Finance","HR","Sales","Product","Operations"], n, p=norm_p([0.18,0.19,0.17,0.16,0.16,0.14]))
    tenure = np.clip(np.random.gamma(3,12,n),0,180).astype(int)
    perf = np.random.choice(["LE (Low)","ME (Meets)","EE (Exceeds)","GE (Greatly Exceeds)"], n, p=norm_p([0.12,0.54,0.28,0.07]))
    age = np.clip(np.random.normal(32,7,n).astype(int),22,60)
    gender = np.random.choice(["M","F","NB"], n, p=norm_p([0.60,0.38,0.02]))
    readiness = np.random.choice(["Not Ready","Ready Now","Ready 1-2Y","Ready 2+Y"], n, p=norm_p([0.15,0.24,0.37,0.25]))
    lb = {"L3":12,"L4":22,"L5":40,"L6":70,"L7":100,"L8":200}
    pm = {"LE (Low)":0.85,"ME (Meets)":1.0,"EE (Exceeds)":1.15,"GE (Greatly Exceeds)":1.30}
    salary = np.array([lb[l]*np.random.uniform(0.8,1.2)*pm[p] for l,p in zip(levels,perf)])
    df = pd.DataFrame({"employee_id":[f"EMP_{i+1:04d}" for i in range(n)],"level":levels,"department":depts,"tenure_months":tenure,"performance_rating":perf,"salary_lakhs":np.round(salary,2),"age":age,"gender":gender,"promotion_readiness":readiness})
    df = compute_attrition_probability(df)
    df["manager_id"] = -1
    # Add revenue per FTE proxy
    rev_map = {"Engineering":18,"Product":25,"Sales":35,"Finance":12,"HR":8,"Operations":14}
    df["revenue_per_fte_lakh"] = df["department"].map(rev_map) * np.random.uniform(0.8,1.2,n)
    return df

class WorkforceSimulator:
    def __init__(self, baseline_df):
        self.baseline = baseline_df.copy()
        self.months = 12

    def _select_targets(self, df, cfg, n):
        el = df.copy()
        if cfg.get("protected_levels"): el = el[~el["level"].isin(cfg["protected_levels"])]
        if cfg.get("protected_depts"): el = el[~el["department"].isin(cfg["protected_depts"])]
        crit = cfg.get("criteria","random")
        if crit=="lifo": return el.sort_values("tenure_months",ascending=True).head(n)
        elif crit=="performance":
            el = el.copy(); el["_s"]=el["performance_rating"].map({"LE (Low)":0,"ME (Meets)":1,"EE (Exceeds)":2,"GE (Greatly Exceeds)":3})
            return el.sort_values("_s",ascending=True).head(n)
        elif crit=="cost": return el.sort_values("salary_lakhs",ascending=False).head(n)
        elif crit=="level":
            el = el.copy(); el["_s"]=el["level"].map({"L8":0,"L7":1,"L6":2,"L5":3,"L4":4,"L3":5})
            return el.sort_values("_s",ascending=True).head(n)
        else: return el.sample(min(n,len(el)),random_state=42)

    def run_scenario(self, planned_hires_per_month, promotion_rate, salary_inflation,
                     attrition_multiplier, hire_dist, promo_bump_pct, backfill_delay,
                     cost_per_hire_lakhs, promotion_cycle_months, restructuring=None,
                     # Feature 2 extra params
                     recruiters=5, req_per_recruiter=4, offer_accept_rate=0.75,
                     revenue_per_fte_lakh=15, ramp_months_by_level=None,
                     enable_cascade=False):
        current = self.baseline.copy()
        history = []
        backfill_queue = {}
        total_severance = 0.0
        total_restructure = 0
        total_cov = 0.0
        capacity_breaches = 0
        if ramp_months_by_level is None:
            ramp_months_by_level = {"L3":1.5,"L4":2,"L5":3,"L6":4,"L7":6,"L8":8}
        hire_total = sum(hire_dist.values())
        if hire_total == 0:
            hire_dist = {"L3":0.41,"L4":0.33,"L5":0.21,"L6":0.05}
        else:
            hire_dist = {k:v/hire_total for k,v in hire_dist.items()}
        # For cascade tracking
        team_attrition_last_month = {}
        for month in range(1, self.months + 1):
            re_exits = 0; re_sev = 0.0
            if restructuring and restructuring.get("enabled") and restructuring.get("cut_count",0)>0:
                spread = max(1, restructuring.get("spread_months",1))
                if month <= spread:
                    monthly_cut = restructuring["cut_count"] // spread
                    if month == spread: monthly_cut += restructuring["cut_count"] % spread
                    if monthly_cut > 0 and len(current) > monthly_cut:
                        targets = self._select_targets(current, restructuring, monthly_cut)
                        if len(targets) > 0:
                            re_exits = len(targets)
                            re_sev = targets["salary_lakhs"].sum() * restructuring.get("severance_months",0) / 100
                            total_severance += re_sev
                            total_restructure += re_exits
                            current = current[~current.index.isin(targets.index)]
                            if restructuring.get("backfill"):
                                backfill_queue[month+backfill_delay] = backfill_queue.get(month+backfill_delay,0) + re_exits
            # Cascade: if same dept had high exits last month, boost prob
            current["monthly_attrition_prob"] = current["attrition_probability"] / 12 * attrition_multiplier
            if enable_cascade and month>1:
                for dept in current["department"].unique():
                    if team_attrition_last_month.get(dept,0)>2:
                        current.loc[current["department"]==dept,"monthly_attrition_prob"] *= 1.4
            current["leaves_this_month"] = np.random.binomial(1, np.clip(current["monthly_attrition_prob"],0,0.9))
            leavers = current[current["leaves_this_month"]==1].copy()
            current = current[current["leaves_this_month"]==0].copy()
            # track for cascade
            if enable_cascade:
                team_attrition_last_month = leavers["department"].value_counts().to_dict()
            n_backfill = len(leavers)
            if n_backfill > 0 and backfill_delay >= 0:
                backfill_queue[month+backfill_delay] = backfill_queue.get(month+backfill_delay,0) + n_backfill
            n_promote = 0
            if month % promotion_cycle_months == 0:
                eligible = current[(current["promotion_readiness"].isin(["Ready Now","Ready 1-2Y"])) & (current["performance_rating"].isin(["EE (Exceeds)","GE (Greatly Exceeds)"])) & (current["level"]!="L8")]
                n_promote = int(len(eligible) * promotion_rate)
                if n_promote > 0 and len(eligible) > 0:
                    promote_idx = eligible.sample(min(n_promote,len(eligible)), random_state=month).index
                    level_order = ["L3","L4","L5","L6","L7","L8"]
                    current.loc[promote_idx,"level"] = current.loc[promote_idx,"level"].apply(lambda x: level_order[min(level_order.index(x)+1,len(level_order)-1)])
                    current.loc[promote_idx,"salary_lakhs"] *= (1 + promo_bump_pct/100)
                    current.loc[promote_idx,"promotion_readiness"] = "Not Ready"
            # Feature 2: capacity constrained hiring
            planned = planned_hires_per_month + backfill_queue.get(month,0)
            max_hires_capacity = int(recruiters * req_per_recruiter * offer_accept_rate)
            actual_hires = min(planned, max_hires_capacity) if max_hires_capacity>0 else planned
            if planned > max_hires_capacity:
                capacity_breaches += (planned - max_hires_capacity)
                # CoV for unfilled
                unfilled = planned - max_hires_capacity
                total_cov += unfilled * (revenue_per_fte_lakh/30) * 30  # simplified: 1 month lost revenue
            else:
                unfilled = 0
            # ramp productivity: effective headcount
            n_hire = actual_hires
            if n_hire > 0:
                hire_levels = np.random.choice(list(hire_dist.keys()), n_hire, p=list(hire_dist.values()))
                depts_pool = current["department"].unique() if len(current)>0 else ["Engineering","Finance","HR","Sales","Product","Operations"]
                new_hires = pd.DataFrame({"employee_id":[f"HIRE_{month}_{i}" for i in range(n_hire)],"level":hire_levels,"department":np.random.choice(depts_pool,n_hire),"tenure_months":np.random.randint(1,12,n_hire),"performance_rating":np.random.choice(["ME (Meets)","EE (Exceeds)","LE (Low)"],n_hire,p=[0.6,0.3,0.1]),"salary_lakhs":[{"L3":12,"L4":22,"L5":40,"L6":70}.get(l,12)*np.random.uniform(0.9,1.1) for l in hire_levels],"age":np.random.randint(24,34,n_hire),"gender":np.random.choice(["M","F","NB"],n_hire,p=[0.60,0.38,0.02]),"promotion_readiness":["Not Ready"]*n_hire})
                new_hires = compute_attrition_probability(new_hires)
                current = pd.concat([current, new_hires], ignore_index=True)
            monthly_payroll = current["salary_lakhs"].sum()/100
            monthly_hire_cost = n_hire * cost_per_hire_lakhs / 100
            monthly_sev = re_sev
            monthly_cost = monthly_payroll + monthly_hire_cost + monthly_sev
            # Effective headcount accounting for ramp (simplified: new hires 0% month1, then ramp)
            ramp_factor = 0.6  # average ramp
            effective_hc = len(current) - n_hire*(1-ramp_factor)
            history.append({"month":month,"headcount":len(current),"effective_headcount":effective_hc,"payroll_cr":monthly_payroll,"hire_cost_cr":monthly_hire_cost,"severance_cr":monthly_sev,"monthly_cost_cr":monthly_cost,"attrition":len(leavers),"hires":n_hire,"planned_hires":planned,"unfilled":planned-actual_hires,"promotions":n_promote,"restructure_exits":re_exits,"cov_lakhs": total_cov})
        hist = pd.DataFrame(history)
        return hist, current, total_severance, total_restructure

# ========== SIDEBAR CONFIG ==========
with st.sidebar:
    st.header("⚙️ Core Levers")
    planned_hires = st.slider("Planned Hires / Month", 0, 50, 8)
    promotion_rate = st.slider("Promotion Rate", 0.0, 0.3, 0.08, 0.01)
    salary_inflation = st.slider("Salary Inflation %", 0, 20, 6)
    attrition_multiplier = st.slider("Attrition Multiplier", 0.5, 1.8, 1.0, 0.1)
    backfill_delay = st.slider("Backfill Delay (mo)", 0, 6, 1)
    promo_bump = st.slider("Promo Bump %", 0, 30, 12)
    cost_per_hire = st.slider("Cost per Hire (Lakhs)", 0, 10, 2)
    promotion_cycle = st.slider("Promo Cycle Months", 1, 12, 6)
    st.divider()
    st.header("🧬 Feature 2: Talent Supply Chain")
    recruiters = st.slider("Recruiters", 1, 20, 5)
    req_per_recruiter = st.slider("Reqs / Recruiter / Month", 1, 10, 4)
    offer_accept_rate = st.slider("Offer Accept Rate", 0.3, 1.0, 0.75, 0.05)
    revenue_per_fte_lakh = st.slider("Revenue / FTE / Month (Lakhs)", 1, 50, 15)
    enable_cascade = st.checkbox("Enable Attrition Cascade (contagion)", value=True)
    st.divider()
    st.header("📦 Hiring Mix")
    hire_l3 = st.slider("L3 %", 0, 100, 40)
    hire_l4 = st.slider("L4 %", 0, 100, 30)
    hire_l5 = st.slider("L5 %", 0, 100, 20)
    hire_l6 = st.slider("L6 %", 0, 100, 10)
    st.divider()
    st.header("🔧 Restructuring")
    restructure_enabled = st.checkbox("Enable Restructuring", value=False, key="restructure_enabled")
    cut_count = st.slider("Cut Count", 0, 200, 20, key="cut_count")
    severance_months = st.slider("Severance Months", 0, 12, 3, key="severance_months")
    st.divider()
    st.header("📁 Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        st.session_state.data_source="upload"
    else:
        if st.checkbox("Use Demo 500", value=True):
            st.session_state.data_source="demo"
        else:
            st.session_state.data_source="none"

# Load data
if st.session_state.get("data_source","demo")=="upload" and uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = compute_attrition_probability(df)
else:
    df = generate_demo_data()
    st.sidebar.success("✅ Demo 500 loaded")

hire_dist = {"L3": hire_l3, "L4": hire_l4, "L5": hire_l5, "L6": hire_l6}
sim = WorkforceSimulator(df)
params = {
    "planned_hires_per_month": planned_hires, "promotion_rate": promotion_rate,
    "salary_inflation": salary_inflation, "attrition_multiplier": attrition_multiplier,
    "hire_dist": hire_dist, "promo_bump_pct": promo_bump, "backfill_delay": backfill_delay,
    "cost_per_hire_lakhs": cost_per_hire, "promotion_cycle_months": promotion_cycle,
    "recruiters": recruiters, "req_per_recruiter": req_per_recruiter,
    "offer_accept_rate": offer_accept_rate, "revenue_per_fte_lakh": revenue_per_fte_lakh,
    "enable_cascade": enable_cascade
}
restruct_cfg = {"enabled":restructure_enabled,"cut_count":cut_count,"criteria":"performance","protected_levels":[],"protected_depts":[],"severance_months":severance_months,"spread_months":1,"backfill":False} if restructure_enabled else None

# Run base scenario
hist, final_df, total_sev, total_re = sim.run_scenario(**params, restructuring=restruct_cfg)

# ========== METRICS ==========
st.title("🧬 Workforce Scenario Engine v5.0 — Survival + Supply Chain + Pareto")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Final HC", hist["headcount"].iloc[-1], int(hist["headcount"].iloc[-1]-len(df)))
c2.metric("Effective HC", f"{hist['effective_headcount'].iloc[-1]:.0f}")
c3.metric("Monthly Cost", f"₹{hist['monthly_cost_cr'].iloc[-1]:.2f} Cr")
c4.metric("CoV Lost (₹L)", f"{hist['cov_lakhs'].iloc[-1]:.0f}L")
c5.metric("Unfilled", int(hist["unfilled"].sum()))

if hist["unfilled"].sum()>0:
    st.error(f"🚨 Talent Supply Chain Breach: {int(hist['unfilled'].sum())} roles unfilled due to capacity {recruiters} recruiters x {req_per_recruiter} reqs x {offer_accept_rate*100:.0f}% accept = {int(recruiters*req_per_recruiter*offer_accept_rate)}/mo max")

# Charts
col1,col2 = st.columns(2)
with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["month"], y=hist["headcount"], name="Headcount", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=hist["month"], y=hist["effective_headcount"], name="Effective HC (ramped)", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=hist["month"], y=hist["planned_hires"].cumsum()+len(df)-hist["attrition"].cumsum(), name="Target w/o cap", line=dict(color="gray", dash="dot")))
    fig.update_layout(title="Headcount vs Capacity Constrained", height=350, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=hist["month"], y=hist["hires"], name="Hired"))
    fig2.add_trace(go.Bar(x=hist["month"], y=hist["unfilled"], name="Unfilled"))
    fig2.update_layout(title="Hires vs Unfilled (Supply Chain)", barmode="stack", height=350)
    st.plotly_chart(fig2, use_container_width=True)

# ========== FEATURE 1: RETENTION ROI ==========
st.divider()
st.header("🧠 Feature 1: Causal Survival + Retention ROI Optimizer")
# Show top flight risks
risk_df = final_df.sort_values("attrition_probability", ascending=False).head(20)
st.markdown("**Top 20 Flight Risks (Survival Model + Regrettable Flag)**")
st.dataframe(risk_df[["employee_id","level","department","performance_rating","tenure_months","compa_ratio","attrition_probability","attrition_risk_bucket","is_regrettable"]], use_container_width=True)

# ROI Calculator
st.subheader("💰 Retention Budget Optimizer")
ret_col1, ret_col2, ret_col3 = st.columns(3)
with ret_col1:
    bonus_pool_lakh = st.slider("Retention Bonus per person (Lakh)", 1, 10, 3)
    uplift = st.slider("P(prevent) uplift if bonus given", 0.1, 0.8, 0.4)
with ret_col2:
    target_group = st.selectbox("Target", ["High+Critical Regrettable", "High+Critical All", "Critical Only", "Ready Now High Risk"])
    if target_group=="High+Critical Regrettable":
        target_df = final_df[(final_df["attrition_risk_bucket"].isin(["High","Critical"])) & (final_df["is_regrettable"])]
    elif target_group=="Critical Only":
        target_df = final_df[final_df["attrition_risk_bucket"]=="Critical"]
    elif target_group=="Ready Now High Risk":
        target_df = final_df[(final_df["promotion_readiness"]=="Ready Now") & (final_df["attrition_risk_bucket"].isin(["High","Critical"]))]
    else:
        target_df = final_df[final_df["attrition_risk_bucket"].isin(["High","Critical"])]
    st.metric("Eligible", len(target_df))
with ret_col3:
    avg_cov = revenue_per_fte_lakh * 2.5  # 2.5 months lost
    expected_saves = len(target_df) * target_df["attrition_probability"].mean() * uplift if len(target_df)>0 else 0
    cost = len(target_df) * bonus_pool_lakh
    saved_cov = expected_saves * avg_cov
    roi = (saved_cov - cost)/cost*100 if cost>0 else 0
    st.metric("Expected Saves", f"{expected_saves:.1f} people")
    st.metric("Cost", f"₹{cost:.0f}L")
    st.metric("ROI", f"{roi:.0f}%", f"Net ₹{saved_cov-cost:.0f}L")
    st.caption(f"Assumes CoV = {avg_cov:.0f}L per regrettable exit (2.5 mo revenue). Survival uplift {uplift*100:.0f}%")

# Cascade insight
if enable_cascade:
    dept_cascade = final_df.groupby("department")["attrition_probability"].mean().sort_values(ascending=False)
    st.warning(f"🔥 Cascade Enabled: Dept with highest contagion risk {dept_cascade.index[0]} ({dept_cascade.iloc[0]*100:.1f}% avg). If >2 exits in same month, next month risk x1.4")

# ========== FEATURE 5: PARETO FRONTIER ==========
st.divider()
st.header("♟️ Feature 5: Pareto Frontier Autonomous Architect")
st.markdown("Instead of 3 scenarios, we run 300 and find the efficient frontier: Cost vs Headcount vs Risk")

p_col1, p_col2, p_col3 = st.columns(3)
with p_col1:
    n_trials = st.slider("Trials", 50, 500, 200, step=50)
    target_hc_input = st.number_input("Target HC", value=int(len(df)*1.1))
with p_col2:
    max_cost_input = st.number_input("Max Monthly Cost Cr", value=float(hist["monthly_cost_cr"].iloc[-1]*1.2))
    optimize_for = st.selectbox("Optimize", ["Cost vs HC", "Cost vs Risk", "HC vs Risk"])
with p_col3:
    run_pareto = st.button("🚀 Run Autonomous Search", type="primary")

if "pareto_results" not in st.session_state:
    st.session_state.pareto_results = None

if run_pareto:
    with st.spinner(f"Running {n_trials} scenarios..."):
        results = []
        np.random.seed(123)
        for i in range(n_trials):
            ph = np.random.randint(0,30)
            pm = np.random.uniform(0.5,1.5)
            pr = np.random.uniform(0.02,0.15)
            si = np.random.uniform(0,12)
            rec = np.random.randint(2,12)
            # quick sim without cascade for speed
            p = params.copy()
            p.update({"planned_hires_per_month":ph,"attrition_multiplier":pm,"promotion_rate":pr,"salary_inflation":si,"recruiters":rec})
            h, f_df, _, _ = sim.run_scenario(**p, restructuring=None)
            final_hc = int(h["headcount"].iloc[-1])
            final_cost = float(h["monthly_cost_cr"].iloc[-1])
            risk = float((f_df["attrition_risk_bucket"].isin(["High","Critical"])).mean()*100)
            cov = float(h["cov_lakhs"].iloc[-1])
            unfilled = int(h["unfilled"].sum())
            # Pareto score: distance to ideal
            results.append({"trial":i,"planned_hires":ph,"attrition_mult":pm,"promo_rate":pr,"inflation":si,"recruiters":rec,
                            "final_hc":final_hc,"final_cost":final_cost,"risk_pct":risk,"cov":cov,"unfilled":unfilled,
                            "score": abs(final_hc-target_hc_input) + (final_cost/max_cost_input)*10 })
        res_df = pd.DataFrame(results)
        # Pareto filter: non-dominated for cost vs hc
        def is_pareto(df):
            pareto = []
            for idx, row in df.iterrows():
                dominated=False
                for _, r2 in df.iterrows():
                    if (r2["final_cost"]<=row["final_cost"] and r2["final_hc"]>=row["final_hc"] and r2["risk_pct"]<=row["risk_pct"] and (r2["final_cost"]<row["final_cost"] or r2["final_hc"]>row["final_hc"] or r2["risk_pct"]<row["risk_pct"])):
                        dominated=True
                        break
                if not dominated:
                    pareto.append(idx)
            return df.loc[pareto]
        pareto_df = is_pareto(res_df)
        st.session_state.pareto_results = (res_df, pareto_df)

if st.session_state.pareto_results is not None:
    res_df, pareto_df = st.session_state.pareto_results
    st.success(f"Found {len(pareto_df)} Pareto-optimal plans out of {len(res_df)}")
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=res_df["final_cost"], y=res_df["final_hc"], mode="markers", marker=dict(color=res_df["risk_pct"], colorscale="RdYlGn_r", showscale=True, colorbar=dict(title="Risk%"), size=8, opacity=0.5), text=res_df["planned_hires"], name="All trials"))
    fig_p.add_trace(go.Scatter(x=pareto_df["final_cost"], y=pareto_df["final_hc"], mode="markers+lines", marker=dict(color="black", size=12, symbol="star"), line=dict(color="black", dash="dash"), name="Pareto Frontier"))
    fig_p.update_layout(title="Pareto Frontier: Cost vs Headcount (color=Risk%)", xaxis_title="Monthly Cost Cr", yaxis_title="Final Headcount", height=500, template="plotly_white")
    fig_p.add_hline(y=target_hc_input, line_dash="dot", annotation_text="Target HC")
    fig_p.add_vline(x=max_cost_input, line_dash="dot", annotation_text="Max Cost")
    st.plotly_chart(fig_p, use_container_width=True)
    
    st.markdown("**Top 5 Pareto Plans**")
    st.dataframe(pareto_df.sort_values("final_cost").head(5)[["planned_hires","recruiters","attrition_mult","final_hc","final_cost","risk_pct","unfilled","cov"]], use_container_width=True)
    
    # Apply best
    if st.button("Apply Cheapest Pareto that hits target"):
        feasible = pareto_df[pareto_df["final_hc"]>=target_hc_input].sort_values("final_cost")
        if len(feasible)>0:
            best = feasible.iloc[0]
            st.info(f"Applying: hires={best['planned_hires']}/mo, recruiters={best['recruiters']}, attr_mult={best['attrition_mult']:.2f} -> HC {best['final_hc']} at ₹{best['final_cost']:.2f}Cr, Risk {best['risk_pct']:.1f}%")
        else:
            st.warning("No Pareto hits target, increase max cost or trials")

# ========== MONTE CARLO v5 with CoV ==========
st.divider()
st.header("🎲 Monte Carlo 1000 runs (now with Supply Chain variance)")
if st.button("Run Monte Carlo 1000"):
    with st.spinner("Running 1000..."):
        records=[]
        for i in range(1000):
            p = params.copy()
            p["planned_hires_per_month"] = int(np.random.normal(params["planned_hires_per_month"],2))
            p["attrition_multiplier"] = np.random.normal(params["attrition_multiplier"],0.15)
            p["offer_accept_rate"] = np.clip(np.random.normal(params["offer_accept_rate"],0.1),0.3,1.0)
            p["recruiters"] = max(1,int(np.random.normal(params["recruiters"],1)))
            hist_mc,_,_,_ = sim.run_scenario(**{k:v for k,v in p.items() if k in ["planned_hires_per_month","promotion_rate","salary_inflation","attrition_multiplier","hire_dist","promo_bump_pct","backfill_delay","cost_per_hire_lakhs","promotion_cycle_months","recruiters","req_per_recruiter","offer_accept_rate","revenue_per_fte_lakh","enable_cascade"]}, restructuring=restruct_cfg)
            records.append({"final_headcount":hist_mc["headcount"].iloc[-1],"final_cost":hist_mc["monthly_cost_cr"].iloc[-1],"cov":hist_mc["cov_lakhs"].iloc[-1],"unfilled":hist_mc["unfilled"].sum()})
        mc_df = pd.DataFrame(records)
        c1,c2,c3 = st.columns(3)
        c1.metric("P50 HC", int(mc_df["final_headcount"].median()), f"P10 {int(mc_df['final_headcount'].quantile(0.1))} / P90 {int(mc_df['final_headcount'].quantile(0.9))}")
        c2.metric("P50 Cost", f"₹{mc_df['final_cost'].median():.2f}Cr")
        c3.metric("P50 CoV Lost", f"₹{mc_df['cov'].median():.0f}L", f"Due to capacity")
        fig_mc = go.Figure(data=[go.Histogram(x=mc_df["final_headcount"], nbinsx=40)])
        fig_mc.update_layout(title="Headcount Distribution with Supply Chain Uncertainty")
        st.plotly_chart(fig_mc, use_container_width=True)

st.divider()
st.caption("v5.0 Built by Sanuj — Features: 1 Survival Retention ROI + 2 Talent Supply Chain CoV + 5 Pareto Autonomous Architect. Deep HR + Deep Math + Deeper AI.")
