import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, json, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Workforce Scenario Engine", page_icon="🧮", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header { font-size: 2.4rem; font-weight: 700; color: #0F172A; margin-bottom: 0.2rem; letter-spacing: -0.02em; }
    .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 2.5rem; font-weight: 400; }
    .metric-card { background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); padding: 1.4rem 1rem; border-radius: 16px; color: white; text-align: center; box-shadow: 0 4px 20px rgba(79, 70, 229, 0.25); }
    .metric-value { font-size: 2.2rem; font-weight: 700; line-height: 1.1; }
    .metric-label { font-size: 0.8rem; opacity: 0.85; margin-top: 0.3rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-delta { font-size: 0.85rem; margin-top: 0.4rem; font-weight: 600; background: rgba(255,255,255,0.15); padding: 0.2rem 0.6rem; border-radius: 20px; display: inline-block; }
    .section-header { font-size: 1.3rem; font-weight: 700; color: #0F172A; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #E2E8F0; }
    .baseline-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; height: 100%; }
    .baseline-card-title { font-size: 0.75rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem; }
    .baseline-card-value { font-size: 1.6rem; font-weight: 700; color: #0F172A; }
    .baseline-card-sub { font-size: 0.8rem; color: #94A3B8; margin-top: 0.2rem; }
    .assumption-card { background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.6rem; display: flex; align-items: flex-start; gap: 0.8rem; }
    .assumption-icon { font-size: 1.2rem; line-height: 1.4; flex-shrink: 0; }
    .assumption-content { flex: 1; }
    .assumption-title { font-weight: 600; color: #0F172A; font-size: 0.9rem; margin-bottom: 0.15rem; }
    .assumption-desc { color: #64748B; font-size: 0.82rem; line-height: 1.4; }
    .assumption-tag { display: inline-block; font-size: 0.65rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 4px; margin-top: 0.4rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .tag-workforce { background: #DBEAFE; color: #1E40AF; }
    .tag-financial { background: #D1FAE5; color: #065F46; }
    .tag-temporal { background: #FEF3C7; color: #92400E; }
    .tag-structural { background: #F3E8FF; color: #6B21A8; }
    .tag-stochastic { background: #FEE2E2; color: #991B1B; }
    .insight-critical { background: linear-gradient(90deg, #FEF2F2 0%, #FFFFFF 100%); border-left: 4px solid #EF4444; padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 0.6rem; }
    .insight-warning { background: linear-gradient(90deg, #FFFBEB 0%, #FFFFFF 100%); border-left: 4px solid #F59E0B; padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 0.6rem; }
    .insight-info { background: linear-gradient(90deg, #EFF6FF 0%, #FFFFFF 100%); border-left: 4px solid #3B82F6; padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 0.6rem; }
    .insight-success { background: linear-gradient(90deg, #ECFDF5 0%, #FFFFFF 100%); border-left: 4px solid #10B981; padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 0.6rem; }
    .source-badge { display: inline-flex; align-items: center; gap: 0.4rem; background: #F1F5F9; border: 1px solid #E2E8F0; padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.8rem; color: #475569; font-weight: 500; }
    .fancy-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, #CBD5E1 50%, transparent 100%); margin: 2rem 0; }
    .ai-panel { background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; }
    .ai-panel pre { color: #E2E8F0; white-space: pre-wrap; font-family: 'Inter', sans-serif; font-size: 0.9rem; line-height: 1.6; }
    .preset-active { background: #DBEAFE; color: #1E40AF; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE INITIALIZATION (must happen before any widgets) ───
defaults = {
    'planned_hires': 8, 'promotion_rate': 0.08, 'salary_inflation': 0.06, 'attrition_factor': 1.0,
    'ai_provider': 'Stochastic (Rule-Based)',
    'hire_l3': 40, 'hire_l4': 35, 'hire_l5': 20, 'hire_l6': 5,
    'promotion_bump': 20, 'backfill_delay': 1, 'cost_per_hire': 3.0,
    'promotion_cycle': 3,
    'last_preset': 'Custom'
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("## 🤖 AI Insights")
    ai_options = ['Stochastic (Rule-Based)', 'Kimi', 'Claude']
    ai_provider = st.selectbox("AI Provider", ai_options, key='ai_provider')

    if ai_provider == 'Kimi':
        if 'KIMI_API_KEY' in st.secrets:
            st.success("✅ Kimi API key found in secrets")
        else:
            st.error("❌ Add KIMI_API_KEY to Streamlit secrets")
    elif ai_provider == 'Claude':
        if 'ANTHROPIC_API_KEY' in st.secrets:
            st.success("✅ Claude API key found in secrets")
        else:
            st.error("❌ Add ANTHROPIC_API_KEY to Streamlit secrets")

    st.markdown("---")
    st.markdown("## 🎯 Scenario Preset")
    preset_options = ["Custom", "Status Quo", "Aggressive Growth", "Cost Optimization"]

    try:
        preset_idx = preset_options.index(st.session_state.last_preset)
    except ValueError:
        preset_idx = 0

    preset = st.selectbox("Select Preset", preset_options, index=preset_idx, key='preset_selector')

    # Apply preset values BEFORE rendering parameter sliders
    if preset != st.session_state.last_preset:
        if preset == "Status Quo":
            st.session_state.planned_hires = 8
            st.session_state.promotion_rate = 0.08
            st.session_state.salary_inflation = 0.06
            st.session_state.attrition_factor = 1.0
        elif preset == "Aggressive Growth":
            st.session_state.planned_hires = 20
            st.session_state.promotion_rate = 0.12
            st.session_state.salary_inflation = 0.08
            st.session_state.attrition_factor = 1.1
        elif preset == "Cost Optimization":
            st.session_state.planned_hires = 2
            st.session_state.promotion_rate = 0.05
            st.session_state.salary_inflation = 0.03
            st.session_state.attrition_factor = 0.85
        # Custom: leave current values as-is
        st.session_state.last_preset = preset
        st.rerun()

    st.markdown("---")
    st.markdown("## ⚙️ Simulation Parameters")
    st.caption("Adjustable when 'Custom' preset is active. Presets load defaults below.")

    planned_hires = st.slider("📥 Planned Hires / Month", 0, 50, key='planned_hires',
                               help="Net new headcount added each month (excludes backfills)")
    promotion_rate = st.slider("📈 Promotion Rate (annual)", 0.0, 0.30, key='promotion_rate', step=0.01,
                                help="% of eligible employees promoted per year")
    salary_inflation = st.slider("💰 Salary Inflation (annual)", 0.0, 0.20, key='salary_inflation', step=0.01,
                                  help="Annual salary increase % applied uniformly")
    attrition_factor = st.slider("🌊 Attrition Multiplier", 0.5, 1.5, key='attrition_factor', step=0.05,
                                  help="1.0 = baseline, >1 = higher attrition, <1 = lower attrition")

    st.markdown("---")
    st.markdown("## 🏗️ Adjustable Assumptions")

    with st.expander("📊 Hire Level Distribution", expanded=False):
        hire_l3 = st.slider("L3 %", 0, 100, key='hire_l3')
        hire_l4 = st.slider("L4 %", 0, 100, key='hire_l4')
        hire_l5 = st.slider("L5 %", 0, 100, key='hire_l5')
        hire_l6 = st.slider("L6 %", 0, 100, key='hire_l6')
        total_pct = hire_l3 + hire_l4 + hire_l5 + hire_l6
        if total_pct != 100:
            st.warning(f"Total = {total_pct}% (must equal 100%)")
        else:
            st.success("Total = 100% ✓")

    promotion_bump = st.slider("⬆️ Promotion Salary Bump %", 10, 50, key='promotion_bump',
                                help="Salary increase on promotion")
    backfill_delay = st.slider("⏱️ Backfill Delay (months)", 0, 6, key='backfill_delay',
                                help="Months between attrition and replacement hire")
    cost_per_hire = st.slider("💸 Cost Per Hire (₹ Lakhs)", 0.0, 10.0, key='cost_per_hire', step=0.5,
                               help="Recruitment cost per external hire")
    promotion_cycle = st.selectbox("🔄 Promotion Cycle (months)", [1, 2, 3, 6, 12], key='promotion_cycle',
                                    help="Frequency of promotion evaluation cycles")

    st.markdown("---")
    st.markdown("### 📁 Data Source")
    uploaded_file = st.file_uploader("Upload Workforce CSV", type="csv",
                                      help="Expected: employee_id, level, department, tenure_months, performance_rating, salary_lakhs, age, gender, promotion_readiness")
    use_demo = st.checkbox("Use Demo Data (500 employees)", value=True)

# ─── DATA LOADING ───
@st.cache_data
def generate_demo_data(seed=42):
    np.random.seed(seed)
    n = 500
    levels = ['L3', 'L4', 'L5', 'L6', 'L7', 'L8']
    level_weights = [0.35, 0.30, 0.18, 0.10, 0.05, 0.02]
    depts = ['Engineering', 'Product', 'Sales', 'Operations', 'HR', 'Finance']
    perf = ['LE (Low)', 'ME (Meets)', 'EE (Exceeds)', 'GE (Greatly Exceeds)']
    perf_w = [0.10, 0.55, 0.28, 0.07]
    level_base = {'L3': 12, 'L4': 20, 'L5': 35, 'L6': 60, 'L7': 95, 'L8': 150}
    level_std = {'L3': 2, 'L4': 4, 'L5': 7, 'L6': 12, 'L7': 20, 'L8': 35}
    df = pd.DataFrame({
        'employee_id': [f'EMP_{str(i).zfill(4)}' for i in range(1, n+1)],
        'level': np.random.choice(levels, n, p=level_weights),
        'department': np.random.choice(depts, n),
        'tenure_months': np.random.gamma(3, 12, n).astype(int),
        'performance_rating': np.random.choice(perf, n, p=perf_w),
        'salary_lakhs': np.zeros(n),
        'age': np.clip(np.random.normal(32, 7, n).astype(int), 22, 60),
        'gender': np.random.choice(['M', 'F', 'NB'], n, p=[0.60, 0.38, 0.02]),
        'promotion_readiness': np.random.choice(['Not Ready', 'Ready Now', 'Ready 1-2Y', 'Ready 2+Y'], n, p=[0.15, 0.20, 0.40, 0.25])
    })
    for idx, row in df.iterrows():
        base = level_base[row['level']]
        std = level_std[row['level']]
        mult = {'LE (Low)': 0.85, 'ME (Meets)': 1.0, 'EE (Exceeds)': 1.15, 'GE (Greatly Exceeds)': 1.30}[row['performance_rating']]
        df.at[idx, 'salary_lakhs'] = round(np.random.normal(base * mult, std), 1)
    def calc_risk(row):
        r = 0.15
        if row['performance_rating'] == 'LE (Low)': r += 0.25
        elif row['performance_rating'] == 'ME (Meets)': r += 0.05
        elif row['performance_rating'] == 'EE (Exceeds)': r -= 0.03
        else: r -= 0.08
        if 12 <= row['tenure_months'] <= 24: r += 0.12
        elif row['tenure_months'] < 6: r += 0.08
        elif row['tenure_months'] > 60: r -= 0.10
        if row['department'] in ['Sales', 'Operations']: r += 0.08
        elif row['department'] == 'Engineering': r += 0.05
        if row['level'] in ['L4', 'L5']: r += 0.06
        med = df[df['level'] == row['level']]['salary_lakhs'].median()
        if row['salary_lakhs'] < med * 0.85: r += 0.10
        elif row['salary_lakhs'] > med * 1.20: r -= 0.08
        if row['promotion_readiness'] == 'Ready Now': r += 0.07
        if row['gender'] == 'F' and row['department'] == 'Engineering': r += 0.03
        return min(max(r, 0.02), 0.85)
    df['attrition_probability'] = df.apply(calc_risk, axis=1)
    df['attrition_risk_bucket'] = pd.cut(df['attrition_probability'], bins=[0, 0.10, 0.20, 0.35, 1.0], labels=['Low', 'Medium', 'High', 'Critical'])
    return df

data_source_label = "Demo Dataset"
if uploaded_file is not None and not use_demo:
    df = pd.read_csv(uploaded_file)
    data_source_label = f"Uploaded: {uploaded_file.name}"
    st.sidebar.success(f"✅ Loaded {len(df)} employees")
elif use_demo:
    df = generate_demo_data()
    data_source_label = "Demo Dataset (500 employees)"
    st.sidebar.info(f"📊 Demo Data: {len(df)} employees")
else:
    st.warning("Please upload a CSV or enable demo data")
    st.stop()

# ─── SIMULATOR ───
class WorkforceSimulator:
    def __init__(self, baseline_df):
        self.baseline = baseline_df.copy()
        self.months = 12
        self.level_base = {'L3': 12, 'L4': 20, 'L5': 35, 'L6': 60, 'L7': 95, 'L8': 150}
        self.level_std = {'L3': 2, 'L4': 4, 'L5': 7, 'L6': 12, 'L7': 20, 'L8': 35}
        self.depts = ['Engineering', 'Product', 'Sales', 'Operations', 'HR', 'Finance']
        self.perf = ['LE (Low)', 'ME (Meets)', 'EE (Exceeds)', 'GE (Greatly Exceeds)']
        self.perf_w = [0.10, 0.55, 0.28, 0.07]

    def run_scenario(self, planned_hires_per_month=8, promotion_rate=0.08, salary_inflation=0.06,
                     external_attrition_factor=1.0, seed=42,
                     hire_dist=None, promotion_bump=0.20, backfill_delay=0,
                     cost_per_hire=0.0, promotion_cycle=3):
        np.random.seed(seed)
        if hire_dist is None:
            hire_dist = {'L3': 0.40, 'L4': 0.35, 'L5': 0.20, 'L6': 0.05}

        current = self.baseline.copy()
        monthly_costs, headcounts, attrition_counts, promotion_counts = [], [], [], []
        backfill_queue = {}
        total_backfills = 0

        for month in range(1, self.months + 1):
            current['monthly_attrition_prob'] = current['attrition_probability'] / 12 * external_attrition_factor
            current['leaves_this_month'] = np.random.random(len(current)) < current['monthly_attrition_prob']
            n_attrition = current['leaves_this_month'].sum()
            current = current[~current['leaves_this_month']].copy()

            # Queue backfills
            if n_attrition > 0:
                backfill_month = month + backfill_delay
                backfill_queue[backfill_month] = backfill_queue.get(backfill_month, 0) + n_attrition

            # Promotions
            n_promotions = 0
            if month % promotion_cycle == 0:
                eligible = current[(current['promotion_readiness'].isin(['Ready Now', 'Ready 1-2Y'])) &
                                   (current['performance_rating'].isin(['EE (Exceeds)', 'GE (Greatly Exceeds)']))]
                n_promote = int(len(eligible) * promotion_rate / (12 / promotion_cycle))
                if n_promote > 0 and len(eligible) > 0:
                    promoted = eligible.sample(min(n_promote, len(eligible)))
                    level_map = {'L3': 'L4', 'L4': 'L5', 'L5': 'L6', 'L6': 'L7', 'L7': 'L8'}
                    for idx_p in promoted.index:
                        old = current.at[idx_p, 'level']
                        if old in level_map:
                            current.at[idx_p, 'level'] = level_map[old]
                            current.at[idx_p, 'salary_lakhs'] *= (1 + promotion_bump)
                            current.at[idx_p, 'promotion_readiness'] = 'Not Ready'
                            n_promotions += 1

            # Hiring
            backfill_hires = backfill_queue.get(month, 0)
            total_backfills += backfill_hires
            total_hires_this_month = planned_hires_per_month + backfill_hires

            new_hires = []
            levels = list(hire_dist.keys())
            probs = np.array(list(hire_dist.values()), dtype=float)
            probs = probs / probs.sum()

            for _ in range(int(total_hires_this_month)):
                lvl = np.random.choice(levels, p=probs)
                base, std = self.level_base[lvl], self.level_std[lvl]
                new_hires.append({
                    'employee_id': f'HIRE_{month}_{_}', 'level': lvl,
                    'department': np.random.choice(self.depts),
                    'tenure_months': 0,
                    'performance_rating': np.random.choice(self.perf, p=self.perf_w),
                    'salary_lakhs': round(np.random.normal(base, std), 1),
                    'age': np.random.randint(24, 35),
                    'gender': np.random.choice(['M', 'F', 'NB'], p=[0.60, 0.38, 0.02]),
                    'manager_id': -1,
                    'promotion_readiness': 'Not Ready',
                    'attrition_probability': np.random.uniform(0.10, 0.25),
                    'attrition_risk_bucket': 'Medium'
                })
            if new_hires:
                current = pd.concat([current, pd.DataFrame(new_hires)], ignore_index=True)

            # Costs
            salary_cost = current['salary_lakhs'].sum()
            hiring_cost = cost_per_hire * total_hires_this_month
            monthly_costs.append(salary_cost + hiring_cost)

            current['salary_lakhs'] *= (1 + salary_inflation / 12)
            current['tenure_months'] += 1
            headcounts.append(len(current))
            attrition_counts.append(n_attrition)
            promotion_counts.append(n_promotions)

        return {
            'final_headcount': len(current),
            'final_monthly_cost': monthly_costs[-1],
            'total_attrition': sum(attrition_counts),
            'total_promotions': sum(promotion_counts),
            'total_hires': planned_hires_per_month * self.months + total_backfills,
            'monthly_headcount': headcounts,
            'monthly_costs': monthly_costs,
            'monthly_attrition': attrition_counts,
            'final_df': current
        }

# Normalize hire distribution
hire_total = st.session_state.hire_l3 + st.session_state.hire_l4 + st.session_state.hire_l5 + st.session_state.hire_l6
if hire_total == 0:
    hire_dist = {'L3': 0.40, 'L4': 0.35, 'L5': 0.20, 'L6': 0.05}
else:
    hire_dist = {
        'L3': st.session_state.hire_l3 / hire_total,
        'L4': st.session_state.hire_l4 / hire_total,
        'L5': st.session_state.hire_l5 / hire_total,
        'L6': st.session_state.hire_l6 / hire_total,
    }

sim = WorkforceSimulator(df)
result = sim.run_scenario(
    planned_hires_per_month=st.session_state.planned_hires,
    promotion_rate=st.session_state.promotion_rate,
    salary_inflation=st.session_state.salary_inflation,
    external_attrition_factor=st.session_state.attrition_factor,
    seed=42,
    hire_dist=hire_dist,
    promotion_bump=st.session_state.promotion_bump / 100,
    backfill_delay=st.session_state.backfill_delay,
    cost_per_hire=st.session_state.cost_per_hire,
    promotion_cycle=st.session_state.promotion_cycle
)

baseline_cost = df['salary_lakhs'].sum()
cost_change = ((result['final_monthly_cost'] / baseline_cost) - 1) * 100

# ─── HEADER ───
st.markdown('<div class="main-header">🧮 Workforce Scenario Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predictive Org Modeling | Monte Carlo Simulation | Tornado Sensitivity | AI-Powered Insights</div>', unsafe_allow_html=True)

preset_badge = f'<span class="preset-active">{st.session_state.last_preset}</span>'
ai_badge = f" | AI: {st.session_state.ai_provider}" if st.session_state.ai_provider != 'Stochastic (Rule-Based)' else ""
st.markdown(f'<div class="source-badge">📊 {data_source_label} | {len(df)} employees | Preset: {preset_badge}{ai_badge} | July 2026</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ─── ACTIVE ASSUMPTIONS BAR ───
with st.expander("📋 Currently Active Model Assumptions (click to expand)", expanded=False):
    a1, a2, a3, a4, a5 = st.columns(5)
    with a1:
        st.metric("Hire Mix", f"L3:{st.session_state.hire_l3}%")
    with a2:
        st.metric("Promo Bump", f"{st.session_state.promotion_bump}%")
    with a3:
        st.metric("Backfill Delay", f"{st.session_state.backfill_delay} mo")
    with a4:
        st.metric("Cost/Hire", f"₹{st.session_state.cost_per_hire:.1f}L")
    with a5:
        st.metric("Promo Cycle", f"{st.session_state.promotion_cycle} mo")

# ─── METRIC CARDS ───
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown('<div class="metric-card"><div class="metric-value">' + str(result['final_headcount']) + '</div><div class="metric-label">Final Headcount</div><div class="metric-delta">' + f"{result['final_headcount'] - len(df):+}" + ' vs baseline</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><div class="metric-value">₹' + f"{result['final_monthly_cost']/100:.1f}" + 'Cr</div><div class="metric-label">Monthly Cost</div><div class="metric-delta">' + f"{cost_change:+.1f}" + '% vs baseline</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="metric-value">' + str(result['total_attrition']) + '</div><div class="metric-label">Total Attrition</div><div class="metric-delta">' + f"{result['total_attrition']/len(df)*100:.1f}" + '% annual rate</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><div class="metric-value">' + str(result['total_hires']) + '</div><div class="metric-label">Total Hires</div><div class="metric-delta">' + str(st.session_state.planned_hires) + '/mo + backfill</div></div>', unsafe_allow_html=True)
with m5:
    st.markdown('<div class="metric-card"><div class="metric-value">' + str(result['total_promotions']) + '</div><div class="metric-label">Promotions</div><div class="metric-delta">' + f"{st.session_state.promotion_rate*100:.0f}" + '% annual rate</div></div>', unsafe_allow_html=True)

st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

# ─── BASELINE SUMMARY ───
st.markdown('<div class="section-header">📋 Baseline Workforce Summary</div>', unsafe_allow_html=True)

baseline_stats = {
    'total_employees': len(df), 'monthly_payroll': df['salary_lakhs'].sum(),
    'annual_payroll': df['salary_lakhs'].sum() * 12, 'avg_salary': df['salary_lakhs'].mean(),
    'median_salary': df['salary_lakhs'].median(), 'median_tenure': df['tenure_months'].median(),
    'avg_age': df['age'].mean(), 'female_pct': (df['gender'] == 'F').mean() * 100,
    'high_performers': (df['performance_rating'].isin(['EE (Exceeds)', 'GE (Greatly Exceeds)'])).sum(),
    'critical_attrition': (df['attrition_risk_bucket'] == 'Critical').sum(),
    'l3_count': (df['level'] == 'L3').sum(), 'l4_count': (df['level'] == 'L4').sum(),
    'l5_plus_count': (df['level'].isin(['L5', 'L6', 'L7', 'L8'])).sum(),
    'ready_now': (df['promotion_readiness'] == 'Ready Now').sum(),
}
top_dept = df['department'].value_counts().index[0]
top_dept_count = df['department'].value_counts().iloc[0]

col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1:
    st.markdown('<div class="baseline-card"><div class="baseline-card-title">Total Employees</div><div class="baseline-card-value">' + str(baseline_stats['total_employees']) + '</div><div class="baseline-card-sub">' + str(baseline_stats['l3_count']) + ' L3 | ' + str(baseline_stats['l4_count']) + ' L4 | ' + str(baseline_stats['l5_plus_count']) + ' L5+</div></div>', unsafe_allow_html=True)
with col_b2:
    st.markdown('<div class="baseline-card"><div class="baseline-card-title">Monthly Payroll</div><div class="baseline-card-value">₹' + f"{baseline_stats['monthly_payroll']/100:.1f}" + ' Cr</div><div class="baseline-card-sub">Annual: ₹' + f"{baseline_stats['annual_payroll']/100:.1f}" + ' Cr | Avg: ₹' + f"{baseline_stats['avg_salary']:.1f}" + 'L</div></div>', unsafe_allow_html=True)
with col_b3:
    st.markdown('<div class="baseline-card"><div class="baseline-card-title">Flight Risk Profile</div><div class="baseline-card-value">' + str(baseline_stats['critical_attrition']) + '</div><div class="baseline-card-sub">Critical risk | ' + str((df['attrition_risk_bucket'] == 'High').sum()) + ' High | ' + str((df['attrition_risk_bucket'] == 'Medium').sum()) + ' Medium</div></div>', unsafe_allow_html=True)
with col_b4:
    st.markdown('<div class="baseline-card"><div class="baseline-card-title">Talent Pipeline</div><div class="baseline-card-value">' + str(baseline_stats['ready_now']) + '</div><div class="baseline-card-sub">Ready Now for promotion | ' + str(baseline_stats['high_performers']) + ' top performers</div></div>', unsafe_allow_html=True)

col_b5, col_b6, col_b7, col_b8 = st.columns(4)
with col_b5:
    st.markdown('<div class="baseline-card"><div class="baseline-card-title">Largest Department</div><div class="baseline-card-value">' + top_dept + '</div><div class="baseline-card-sub">' + str(top_dept_count) + ' employees (' + f"{top_dept_count/len(df)*100:.0f}" + '% of workforce)</div></div>', unsafe_allow_html=True)
with col_b6:
    st.markdown('<div class="baseline-card"><div class="baseline-card-title">Median Tenure</div><div class="baseline-card-value">' + f"{baseline_stats['median_tenure']:.0f}" + ' mo</div><div class="baseline-card-sub">~' + f"{baseline_stats['median_tenure']/12:.1f}" + ' years | Peak attrition risk: 12-24 mo</div></div>', unsafe_allow_html=True)
with col_b7:
    st.markdown('<div class="baseline-card"><div class="baseline-card-title">Gender Diversity</div><div class="baseline-card-value">' + f"{baseline_stats['female_pct']:.0f}" + '%</div><div class="baseline-card-sub">Female representation | ' + f"{(df['gender'] == 'M').mean()*100:.0f}" + '% Male | ' + f"{(df['gender'] == 'NB').mean()*100:.0f}" + '% Non-binary</div></div>', unsafe_allow_html=True)
with col_b8:
    st.markdown('<div class="baseline-card"><div class="baseline-card-title">Average Age</div><div class="baseline-card-value">' + f"{baseline_stats['avg_age']:.0f}" + '</div><div class="baseline-card-sub">Years | Range: ' + str(df['age'].min()) + '-' + str(df['age'].max()) + '</div></div>', unsafe_allow_html=True)

st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

# ─── CHARTS ───
months = list(range(1, 13))

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("📈 Headcount Trajectory")
    fig_hc = go.Figure()
    fig_hc.add_trace(go.Scatter(x=months, y=result['monthly_headcount'], mode='lines+markers', name='Projected', line=dict(color='#4F46E5', width=3), fill='tozeroy', fillcolor='rgba(79, 70, 229, 0.08)', marker=dict(size=6, color='#4F46E5')))
    fig_hc.add_hline(y=len(df), line_dash="dash", line_color="#EF4444", annotation_text="Baseline", annotation_position="top right")
    fig_hc.update_layout(height=380, margin=dict(l=40, r=40, t=30, b=30), xaxis_title="Month", yaxis_title="Headcount", template="plotly_white", font=dict(family="Inter, sans-serif"), plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hc, use_container_width=True)

with col_right:
    st.subheader("💰 Monthly Cost Trajectory")
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(x=months, y=[c/100 for c in result['monthly_costs']], mode='lines+markers', name='Projected', line=dict(color='#10B981', width=3), fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.08)', marker=dict(size=6, color='#10B981')))
    fig_cost.add_hline(y=baseline_cost/100, line_dash="dash", line_color="#EF4444", annotation_text="Baseline", annotation_position="top right")
    fig_cost.update_layout(height=380, margin=dict(l=40, r=40, t=30, b=30), xaxis_title="Month", yaxis_title="Monthly Cost (₹ Cr)", template="plotly_white", font=dict(family="Inter, sans-serif"), plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cost, use_container_width=True)

# Tornado
base_hc = result['final_headcount']
base_cost = result['final_monthly_cost']
sensitivity_params = [
    ('Planned Hires / Month', max(0, st.session_state.planned_hires-6), st.session_state.planned_hires+12, st.session_state.planned_hires),
    ('Attrition Multiplier', max(0.5, st.session_state.attrition_factor-0.3), min(1.5, st.session_state.attrition_factor+0.3), st.session_state.attrition_factor),
    ('Promotion Rate (%)', max(0.0, st.session_state.promotion_rate-0.05), min(0.30, st.session_state.promotion_rate+0.05), st.session_state.promotion_rate),
    ('Salary Inflation (%)', max(0.0, st.session_state.salary_inflation-0.04), min(0.20, st.session_state.salary_inflation+0.04), st.session_state.salary_inflation),
]
tornado_hc, tornado_cost = [], []
for name, low_val, high_val, base_val in sensitivity_params:
    if name == 'Planned Hires / Month':
        low_res = sim.run_scenario(planned_hires_per_month=int(low_val), promotion_rate=st.session_state.promotion_rate, salary_inflation=st.session_state.salary_inflation, external_attrition_factor=st.session_state.attrition_factor, seed=42, hire_dist=hire_dist, promotion_bump=st.session_state.promotion_bump/100, backfill_delay=st.session_state.backfill_delay, cost_per_hire=st.session_state.cost_per_hire, promotion_cycle=st.session_state.promotion_cycle)
        high_res = sim.run_scenario(planned_hires_per_month=int(high_val), promotion_rate=st.session_state.promotion_rate, salary_inflation=st.session_state.salary_inflation, external_attrition_factor=st.session_state.attrition_factor, seed=42, hire_dist=hire_dist, promotion_bump=st.session_state.promotion_bump/100, backfill_delay=st.session_state.backfill_delay, cost_per_hire=st.session_state.cost_per_hire, promotion_cycle=st.session_state.promotion_cycle)
    elif name == 'Attrition Multiplier':
        low_res = sim.run_scenario(planned_hires_per_month=st.session_state.planned_hires, promotion_rate=st.session_state.promotion_rate, salary_inflation=st.session_state.salary_inflation, external_attrition_factor=low_val, seed=42, hire_dist=hire_dist, promotion_bump=st.session_state.promotion_bump/100, backfill_delay=st.session_state.backfill_delay, cost_per_hire=st.session_state.cost_per_hire, promotion_cycle=st.session_state.promotion_cycle)
        high_res = sim.run_scenario(planned_hires_per_month=st.session_state.planned_hires, promotion_rate=st.session_state.promotion_rate, salary_inflation=st.session_state.salary_inflation, external_attrition_factor=high_val, seed=42, hire_dist=hire_dist, promotion_bump=st.session_state.promotion_bump/100, backfill_delay=st.session_state.backfill_delay, cost_per_hire=st.session_state.cost_per_hire, promotion_cycle=st.session_state.promotion_cycle)
    elif name == 'Promotion Rate (%)':
        low_res = sim.run_scenario(planned_hires_per_month=st.session_state.planned_hires, promotion_rate=low_val, salary_inflation=st.session_state.salary_inflation, external_attrition_factor=st.session_state.attrition_factor, seed=42, hire_dist=hire_dist, promotion_bump=st.session_state.promotion_bump/100, backfill_delay=st.session_state.backfill_delay, cost_per_hire=st.session_state.cost_per_hire, promotion_cycle=st.session_state.promotion_cycle)
        high_res = sim.run_scenario(planned_hires_per_month=st.session_state.planned_hires, promotion_rate=high_val, salary_inflation=st.session_state.salary_inflation, external_attrition_factor=st.session_state.attrition_factor, seed=42, hire_dist=hire_dist, promotion_bump=st.session_state.promotion_bump/100, backfill_delay=st.session_state.backfill_delay, cost_per_hire=st.session_state.cost_per_hire, promotion_cycle=st.session_state.promotion_cycle)
    else:
        low_res = sim.run_scenario(planned_hires_per_month=st.session_state.planned_hires, promotion_rate=st.session_state.promotion_rate, salary_inflation=low_val, external_attrition_factor=st.session_state.attrition_factor, seed=42, hire_dist=hire_dist, promotion_bump=st.session_state.promotion_bump/100, backfill_delay=st.session_state.backfill_delay, cost_per_hire=st.session_state.cost_per_hire, promotion_cycle=st.session_state.promotion_cycle)
        high_res = sim.run_scenario(planned_hires_per_month=st.session_state.planned_hires, promotion_rate=st.session_state.promotion_rate, salary_inflation=high_val, external_attrition_factor=st.session_state.attrition_factor, seed=42, hire_dist=hire_dist, promotion_bump=st.session_state.promotion_bump/100, backfill_delay=st.session_state.backfill_delay, cost_per_hire=st.session_state.cost_per_hire, promotion_cycle=st.session_state.promotion_cycle)
    tornado_hc.append({'parameter': name, 'low': low_res['final_headcount'] - base_hc, 'high': high_res['final_headcount'] - base_hc, 'range': abs(high_res['final_headcount'] - low_res['final_headcount'])})
    tornado_cost.append({'parameter': name, 'low': (low_res['final_monthly_cost'] - base_cost) / 100, 'high': (high_res['final_monthly_cost'] - base_cost) / 100, 'range': abs(high_res['final_monthly_cost'] - low_res['final_monthly_cost']) / 100})
tornado_hc = sorted(tornado_hc, key=lambda x: x['range'], reverse=True)
tornado_cost = sorted(tornado_cost, key=lambda x: x['range'], reverse=True)

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.subheader("🌪️ Tornado: Headcount Sensitivity")
    params_hc = [t['parameter'] for t in tornado_hc]
    low_hc = [t['low'] for t in tornado_hc]
    high_hc = [t['high'] for t in tornado_hc]
    fig_t1 = go.Figure()
    fig_t1.add_trace(go.Bar(y=params_hc, x=low_hc, orientation='h', name='Low Parameter', marker_color='#EF4444', opacity=0.85, text=[f'{v:+.0f}' for v in low_hc], textposition='inside', showlegend=False))
    fig_t1.add_trace(go.Bar(y=params_hc, x=high_hc, orientation='h', name='High Parameter', marker_color='#10B981', opacity=0.85, text=[f'{v:+.0f}' for v in high_hc], textposition='inside', showlegend=False))
    fig_t1.add_vline(x=0, line_width=2, line_color="#0F172A")
    fig_t1.update_layout(height=380, barmode='overlay', margin=dict(l=40, r=40, t=30, b=30), xaxis_title="Δ Final Headcount", template="plotly_white", font=dict(family="Inter, sans-serif"), plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_t1, use_container_width=True)

with col_t2:
    st.subheader("🌪️ Tornado: Cost Sensitivity")
    params_c = [t['parameter'] for t in tornado_cost]
    low_c = [t['low'] for t in tornado_cost]
    high_c = [t['high'] for t in tornado_cost]
    fig_t2 = go.Figure()
    fig_t2.add_trace(go.Bar(y=params_c, x=low_c, orientation='h', name='Low Parameter', marker_color='#EF4444', opacity=0.85, text=[f'{v:+.1f} Cr' for v in low_c], textposition='inside', showlegend=False))
    fig_t2.add_trace(go.Bar(y=params_c, x=high_c, orientation='h', name='High Parameter', marker_color='#10B981', opacity=0.85, text=[f'{v:+.1f} Cr' for v in high_c], textposition='inside', showlegend=False))
    fig_t2.add_vline(x=0, line_width=2, line_color="#0F172A")
    fig_t2.update_layout(height=380, barmode='overlay', margin=dict(l=40, r=40, t=30, b=30), xaxis_title="Δ Final Monthly Cost (₹ Cr)", template="plotly_white", font=dict(family="Inter, sans-serif"), plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_t2, use_container_width=True)

col_l1, col_l2 = st.columns(2)
with col_l1:
    st.subheader("📊 Final Level Distribution")
    final_df = result['final_df']
    level_counts = final_df['level'].value_counts().reindex(['L3','L4','L5','L6','L7','L8'], fill_value=0)
    colors_lvl = ['#93C5FD', '#60A5FA', '#3B82F6', '#2563EB', '#1D4ED8', '#1E40AF']
    fig_lvl = go.Figure(data=[go.Pie(labels=level_counts.index, values=level_counts.values, hole=0.5, marker_colors=colors_lvl, textinfo='label+percent', textfont_size=12, pull=[0.02]*6)])
    fig_lvl.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), showlegend=False, template="plotly_white", font=dict(family="Inter, sans-serif"), annotations=[dict(text=f"<b>{len(final_df)}</b><br>Total", x=0.5, y=0.5, font_size=14, showarrow=False, font_color="#0F172A")])
    st.plotly_chart(fig_lvl, use_container_width=True)

with col_l2:
    st.subheader("📉 Cumulative Attrition")
    cum_attr = np.cumsum(result['monthly_attrition'])
    fig_attr = go.Figure()
    fig_attr.add_trace(go.Scatter(x=months, y=cum_attr, mode='lines+markers', line=dict(color='#F59E0B', width=3), fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.08)', marker=dict(size=6)))
    fig_attr.update_layout(height=380, margin=dict(l=40, r=40, t=30, b=30), xaxis_title="Month", yaxis_title="Cumulative Attrition", template="plotly_white", font=dict(family="Inter, sans-serif"), plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_attr, use_container_width=True)

st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

# ─── AI INSIGHTS ───
st.markdown('<div class="section-header">🤖 Strategic Insights</div>', unsafe_allow_html=True)

ai_text = None
use_ai = st.session_state.ai_provider != 'Stochastic (Rule-Based)'

if use_ai:
    with st.spinner(f"Consulting {st.session_state.ai_provider} for strategic analysis..."):
        try:
            # Validate API key exists
            if st.session_state.ai_provider == 'Kimi' and 'KIMI_API_KEY' not in st.secrets:
                st.warning("Kimi API key not found in secrets. Falling back to rule-based insights.")
                use_ai = False
            elif st.session_state.ai_provider == 'Claude' and 'ANTHROPIC_API_KEY' not in st.secrets:
                st.warning("Claude API key not found in secrets. Falling back to rule-based insights.")
                use_ai = False
            else:
                prompt = f"""You are a Chief People Officer with 15 years at Amazon, Trianz, and top tech firms. Analyze this workforce simulation and provide exactly 3 strategic insights.

BASELINE WORKFORCE:
- Total employees: {len(df)}
- Monthly payroll: ₹{df['salary_lakhs'].sum()/100:.1f} Cr
- Top departments: {dict(df['department'].value_counts().head(3))}
- Flight risk: {(df['attrition_risk_bucket'] == 'Critical').sum()} critical, {(df['attrition_risk_bucket'] == 'High').sum()} high
- Ready for promotion: {(df['promotion_readiness'] == 'Ready Now').sum()}

SIMULATION PARAMETERS:
- Planned hires/month: {st.session_state.planned_hires}
- Promotion rate: {st.session_state.promotion_rate*100:.0f}%
- Salary inflation: {st.session_state.salary_inflation*100:.0f}%
- Attrition multiplier: {st.session_state.attrition_factor}

MODEL ASSUMPTIONS:
- Hire distribution: L3 {st.session_state.hire_l3}%, L4 {st.session_state.hire_l4}%, L5 {st.session_state.hire_l5}%, L6 {st.session_state.hire_l6}%
- Promotion salary bump: {st.session_state.promotion_bump}%
- Backfill delay: {st.session_state.backfill_delay} months
- Cost per hire: ₹{st.session_state.cost_per_hire:.1f} Lakhs
- Promotion cycle: Every {st.session_state.promotion_cycle} months
- Preset: {st.session_state.last_preset}

RESULTS AFTER 12 MONTHS:
- Final headcount: {result['final_headcount']} (net change: {result['final_headcount'] - len(df):+})
- Final monthly cost: ₹{result['final_monthly_cost']/100:.1f} Cr
- Total attrition: {result['total_attrition']} ({result['total_attrition']/len(df)*100:.1f}% annualized)
- Total hires: {result['total_hires']} (planned + backfill)
- Total promotions: {result['total_promotions']}

FORMAT each insight as:
**[CRITICAL / WARNING / INFO] Title**: Explanation with specific numbers and one concrete action item.

Be concise, data-driven, and actionable. Write like a CPO presenting to a board."""

                if st.session_state.ai_provider == 'Kimi':
                    from openai import OpenAI
                    client = OpenAI(api_key=st.secrets['KIMI_API_KEY'], base_url="https://api.moonshot.ai/v1")
                    response = client.chat.completions.create(
                        model="kimi-k3",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=1500
                    )
                    ai_text = response.choices[0].message.content
                elif st.session_state.ai_provider == 'Claude':
                    import anthropic
                    client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=1500,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    ai_text = response.content[0].text
        except Exception as e:
            st.error(f"AI API Error: {str(e)}")
            use_ai = False

if ai_text and use_ai:
    st.markdown('<div class="ai-panel"><pre>' + ai_text + '</pre></div>', unsafe_allow_html=True)
else:
    # Rule-based fallback
    insights = []
    high_risk = final_df[final_df['attrition_risk_bucket'].isin(['High', 'Critical'])]
    if len(high_risk) > len(final_df) * 0.25:
        top_dept_risk = high_risk['department'].value_counts().index[0]
        insights.append(('CRITICAL', f"🚨 {len(high_risk)} employees ({len(high_risk)/len(final_df)*100:.0f}%) at High/Critical flight risk. Top department: {top_dept_risk}."))
    l6_plus = len(final_df[final_df['level'].isin(['L6', 'L7', 'L8'])])
    l5_ready = len(final_df[(final_df['level'] == 'L5') & (final_df['promotion_readiness'] == 'Ready Now')])
    if l6_plus < 40 and l5_ready < 15:
        insights.append(('WARNING', f"⚠️ L6+ bench at {l6_plus} with only {l5_ready} L5s ready now. Succession gap emerging."))
    eng_df = final_df[final_df['department'] == 'Engineering']
    if len(eng_df) > 0:
        f_eng = len(eng_df[eng_df['gender'] == 'F']) / len(eng_df)
        if f_eng < 0.30:
            insights.append(('ATTENTION', f"💡 Engineering female representation at {f_eng*100:.0f}%. Below 30% threshold."))
    cost_change_abs = result['final_monthly_cost'] - baseline_cost
    if cost_change_abs > 0:
        insights.append(('WARNING', f"💰 Monthly cost up ₹{cost_change_abs/100:.1f} Cr vs baseline. Seniority drift or inflation pressure."))
    elif cost_change_abs < -10:
        insights.append(('INFO', f"✅ Monthly cost down ₹{abs(cost_change_abs)/100:.1f} Cr. Verify capability retention."))
    l3_ratio = len(final_df[final_df['level'] == 'L3']) / len(final_df)
    if l3_ratio > 0.45:
        insights.append(('WARNING', f"📊 L3-heavy pyramid ({l3_ratio*100:.0f}%). Experience dilution risk. Accelerate L4 readiness."))
    net_change = result['total_hires'] - result['total_attrition']
    if net_change < 0:
        insights.append(('CRITICAL', f"🚨 Net headcount loss: {net_change} (Hired {result['total_hires']}, Lost {result['total_attrition']}). Growth plan at risk."))

    for severity, msg in insights:
        css_class = {'CRITICAL': 'insight-critical', 'WARNING': 'insight-warning', 'ATTENTION': 'insight-info', 'INFO': 'insight-success'}.get(severity, 'insight-info')
        st.markdown(f'<div class="{css_class}"><b>{severity}</b><br>{msg}</div>', unsafe_allow_html=True)
    if not insights:
        st.success("✅ All workforce metrics within healthy ranges for this scenario.")
    if st.session_state.ai_provider == 'Stochastic (Rule-Based)':
        st.info("💡 Select Kimi or Claude in the sidebar to generate LLM-powered strategic narratives.")

st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

# ─── EXECUTIVE SUMMARY ───
st.markdown('<div class="section-header">📝 Executive Summary</div>', unsafe_allow_html=True)
net_change = result['final_headcount'] - len(df)
st.markdown(f'<div style="background: #F8FAFC; border-radius: 12px; padding: 1.5rem; border: 1px solid #E2E8F0; line-height: 1.8; color: #334155;">Under the <b>{st.session_state.last_preset}</b> preset with current assumptions, workforce evolves from <b>{len(df)}</b> to <b>{result["final_headcount"]}</b> employees over 12 months (<b>{net_change:+,}</b> net change). Monthly payroll shifts from <b>₹{baseline_cost/100:.1f} Cr</b> to <b>₹{result["final_monthly_cost"]/100:.1f} Cr</b> (<b>{cost_change:+.1f}%</b> cost movement). Total attrition of <b>{result["total_attrition"]}</b> employees is offset by <b>{result["total_hires"]}</b> hires (including backfills) and <b>{result["total_promotions"]}</b> promotions. Hire mix is L3:{st.session_state.hire_l3}% L4:{st.session_state.hire_l4}% L5:{st.session_state.hire_l5}% L6:{st.session_state.hire_l6}%, with a {st.session_state.backfill_delay}-month backfill delay and ₹{st.session_state.cost_per_hire:.1f}L cost per hire.</div>', unsafe_allow_html=True)

st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

# ─── MODEL ASSUMPTIONS SECTION ───
st.markdown('<div class="section-header">⚙️ Model Assumptions & Methodology</div>', unsafe_allow_html=True)
st.markdown('<div style="background: #FEFCE8; border: 1px solid #FDE68A; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1.5rem;"><span style="color: #92400E; font-size: 0.9rem;">⚠️ <b>Important:</b> All projections are directional estimates based on the assumptions below. Results should be validated against historical data before use in strategic planning.</span></div>', unsafe_allow_html=True)

assumptions = [
    {"icon": "👥", "title": "Backfill Replacement with Delay", "desc": f"Every departing employee is replaced after {st.session_state.backfill_delay} month(s). Until then, the role remains vacant. Zero onboarding productivity ramp is modeled.", "tag": "workforce", "impact": f"{'Overstates' if st.session_state.backfill_delay == 0 else 'Understates'} capacity by {st.session_state.backfill_delay} months per exit."},
    {"icon": "💸", "title": "Cost Per Hire", "desc": f"Each external hire incurs a fixed recruitment cost of ₹{st.session_state.cost_per_hire:.1f} Lakhs. This includes agency fees, signing bonuses, and onboarding overhead.", "tag": "financial", "impact": f"Adds ₹{(st.session_state.cost_per_hire * result['total_hires'])/100:.1f} Cr total to 12-month cost."},
    {"icon": "📊", "title": "Hire Level Distribution", "desc": f"New hires follow a static distribution: L3 {st.session_state.hire_l3}%, L4 {st.session_state.hire_l4}%, L5 {st.session_state.hire_l5}%, L6 {st.session_state.hire_l6}%. This does not adjust based on open roles or market availability.", "tag": "structural", "impact": "May create level imbalances if hiring heavily skews to one department."},
    {"icon": "📈", "title": "Static Attrition Probability", "desc": "Each employee's attrition risk is calculated once at baseline and does not evolve with tenure, performance changes, or market conditions during the 12-month period.", "tag": "workforce", "impact": "May miss tenure-hump dynamics (e.g., 18-month flight risk spike)."},
    {"icon": "🎲", "title": "Monte Carlo with Fixed Seed", "desc": "Attrition events are probabilistic (Bernoulli trials) but use a fixed random seed (seed=42) for reproducibility. The same inputs always produce the same outputs.", "tag": "stochastic", "impact": "No confidence intervals shown; single deterministic path per scenario."},
    {"icon": "📈", "title": "Uniform Salary Inflation", "desc": f"All employees receive the same monthly inflation adjustment ({st.session_state.salary_inflation*100:.0f}% annual). No differentiation by performance, level, department, or market benchmarking.", "tag": "financial", "impact": "Top performers typically get 2-3x inflation; model flattens this."},
    {"icon": "🔄", "title": "Promotion Cycle", "desc": f"Promotions are evaluated every {st.session_state.promotion_cycle} months. Eligible employees must be 'Ready Now' or 'Ready 1-2Y' AND rated EE or GE.", "tag": "structural", "impact": "Real promotion cycles vary by company; some do bi-annual or continuous."},
    {"icon": "⬆️", "title": "Promotion Salary Bump", "desc": f"Every promoted employee receives exactly a {st.session_state.promotion_bump}% salary increase. No variation by level jump (L4→L5 vs L6→L7) or department.", "tag": "financial", "impact": "Senior promotions typically carry larger % increases."},
    {"icon": "⏱️", "title": "12-Month Fixed Horizon", "desc": "The simulation runs exactly 12 months. No carryover of unfinished projects, pending hires, or in-flight promotions into Month 13+.", "tag": "temporal", "impact": "Seasonal patterns (e.g., Q4 hiring freezes) not captured."},
    {"icon": "🏢", "title": "No Department-Level Constraints", "desc": "Hiring and attrition are company-wide aggregates. Individual department headcount limits, budget caps, and skills-mix requirements are not enforced.", "tag": "structural", "impact": "Could produce unrealistic scenarios (e.g., 200 Sales hires, 0 Engineering)."},
    {"icon": "🧬", "title": "No External Market Data", "desc": "Salary benchmarks, competitor attrition rates, and labor market conditions are not integrated. The model is purely internal-facing.", "tag": "workforce", "impact": "May miss market-driven salary compression or talent shortage effects."},
    {"icon": "📉", "title": "No Layoff or Restructuring Logic", "desc": "The model only simulates voluntary attrition. Involuntary terminations, performance-based exits, or strategic workforce reductions are not modeled.", "tag": "workforce", "impact": "Cost Optimization scenarios may understate achievable headcount reduction."},
    {"icon": "🌍", "title": "Single Currency & Geography", "desc": "All costs are in INR Lakhs. Multi-currency workforces, geo-differential pay, and cross-border tax implications are not handled.", "tag": "financial", "impact": "Global companies need FX-adjusted modeling."},
    {"icon": "🤖", "title": "AI Insights", "desc": f"The strategic insights panel uses {'Claude/Kimi LLM' if st.session_state.ai_provider != 'Stochastic (Rule-Based)' else 'hardcoded business rules'} to generate recommendations. Thresholds are based on industry benchmarks.", "tag": "stochastic", "impact": f"{'LLM insights may vary in tone and specificity.' if st.session_state.ai_provider != 'Stochastic (Rule-Based)' else 'Rule-based insights may not generalize to all industries.'}"},
]

for a in assumptions:
    st.markdown(f'<div class="assumption-card"><div class="assumption-icon">{a["icon"]}</div><div class="assumption-content"><div class="assumption-title">{a["title"]}</div><div class="assumption-desc">{a["desc"]}</div><div style="margin-top: 0.3rem;"><span class="assumption-tag tag-{a["tag"]}">{a["tag"]}</span><span style="color: #94A3B8; font-size: 0.78rem; margin-left: 0.5rem;">📌 Impact: {a["impact"]}</span></div></div></div>', unsafe_allow_html=True)

st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

# ─── DOWNLOAD ───
st.markdown('<div class="section-header">📥 Export Results</div>', unsafe_allow_html=True)
col_d1, col_d2 = st.columns(2)
with col_d1:
    csv = final_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="⬇️ Download Final Workforce CSV", data=csv, file_name='workforce_projection.csv', mime='text/csv', use_container_width=True)
with col_d2:
    summary = {
        'scenario': {
            'preset': st.session_state.last_preset,
            'planned_hires_per_month': st.session_state.planned_hires,
            'promotion_rate': st.session_state.promotion_rate,
            'salary_inflation': st.session_state.salary_inflation,
            'attrition_multiplier': st.session_state.attrition_factor,
            'ai_provider': st.session_state.ai_provider
        },
        'assumptions': {
            'hire_distribution': {'L3': st.session_state.hire_l3, 'L4': st.session_state.hire_l4, 'L5': st.session_state.hire_l5, 'L6': st.session_state.hire_l6},
            'promotion_bump_pct': st.session_state.promotion_bump,
            'backfill_delay_months': st.session_state.backfill_delay,
            'cost_per_hire_lakhs': st.session_state.cost_per_hire,
            'promotion_cycle_months': st.session_state.promotion_cycle
        },
        'results': {
            'baseline_headcount': int(len(df)),
            'final_headcount': int(result['final_headcount']),
            'baseline_monthly_cost_cr': round(baseline_cost/100, 2),
            'final_monthly_cost_cr': round(result['final_monthly_cost']/100, 2),
            'total_attrition': int(result['total_attrition']),
            'total_hires': int(result['total_hires']),
            'total_promotions': int(result['total_promotions'])
        }
    }
    json_str = json.dumps(summary, indent=2)
    st.download_button(label="⬇️ Download Summary JSON", data=json_str, file_name='scenario_summary.json', mime='application/json', use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Plotly | Workforce Scenario Engine v2.1")
