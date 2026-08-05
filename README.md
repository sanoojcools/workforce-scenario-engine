# 🧬 Workforce Scenario Engine v5.0
### The first open-source workforce planner that models *whether your plan is actually possible*

**Built by an HR leader (ex-Amazon, ex-AWS) for CHROs who are tired of Excel that lies.**

[![Streamlit](https://img.shields.io/badge/Built_with-Streamlit-FF4B4B)]()
[![Monte Carlo](https://img.shields.io/badge/Engine-Monte_Carlo_1000_runs-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

Live Demo: `sanoojcools-workforce-scenario-engine.streamlit.app` | Video: Coming soon

---

### Why this exists
CHROs spend 2 days rebuilding headcount models every time Finance asks "what-if". The answer is a single-point estimate with no confidence interval. The board doesn't trust it.

Finance asks: "Can we actually hire 120 engineers in Bangalore in 90 days?"  
HR answers: "We'll try."  

**This engine answers: "No. You have 5 recruiters x 4 reqs x 75% accept = 15 hires/mo max. You will be 42 roles short, costing ₹3.2Cr in lost capacity. Here are 73 Pareto-optimal plans that *do* work."**

### 🆕 What v5.0 Adds (Features 1, 2, 5)

**Feature 1: Causal Survival + Retention ROI Optimizer**
- Replaces flat 15% attrition with Weibull survival curves + Cox-style hazard ratios
- Factors: compa-ratio, tenure peak (12-24mo), performance, Ready Now status, Sales/Ops risk, gender x tech interaction, **attrition cascade (contagion)**
- Flags: `is_regrettable` (EE/GE + Ready Now), `compa_ratio`, `attrition_probability`
- **Retention Budget Optimizer:** Set bonus (₹L) and P(prevent) uplift → Get Expected Saves, Cost, ROI, Net Saved CoV. Turns retention from gut feel to ROI math.

**Feature 2: Talent Supply Chain + Cost of Vacancy**
- Hiring as a constrained queue: `max_hires = recruiters * reqs/recruiter * offer_accept_rate`
- Tracks `unfilled` roles, `effective_headcount` (ramped), and `CoV (Cost of Vacancy) = unfilled * revenue_per_FTE`
- Red breach banner when capacity exceeded. Bar chart: Hired vs Unfilled per month.
- Slider: `Revenue / FTE / Month (Lakhs)` — the number CFOs actually care about.

**Feature 5: Pareto Frontier Autonomous Architect**
- Instead of 3 fixed scenarios (Status Quo / Growth / Cost Opt), runs 50-500 autonomous trials
- Multi-objective: Min Cost, Max Headcount, Min Risk% (High/Critical)
- Non-dominated sorting → **Pareto Frontier** (black stars). Bubble color = Risk%
- Output: Top 5 Pareto plans with hires, recruiters, attrition_mult, final HC, cost, unfilled, CoV. One-click apply.

### Core Capabilities (still there, now better)
- **12-month Monte Carlo:** 1000 runs with fan charts P10/P50/P90 for headcount + cost + CoV
- **Tornado Sensitivity:** What actually moves final HC and cost
- **Restructuring Simulator:** Cut by LIFO/performance/cost/level with severance and break-even
- **AI Co-pilot:** Tool-calling agent for what-if, dept deep-dive, optimize, restructuring
- **Board Deck Mode:** 16:9 screenshot-ready charts

### 📊 Demo in 30 seconds
1. Use demo 500 employees (synthetic, no PII)
2. Set Recruiters = 2, see Headcount vs Capacity chart diverge + Unfilled stack rise
3. Scroll to Retention ROI: Set Target = Critical Only, Bonus = 2L → See positive ROI
4. Scroll to Pareto Frontier → Run Autonomous Search (200 trials) → See 73 optimal plans

### 🔧 Built With
- Python + Streamlit + Plotly + Pandas + NumPy
- No backend, no DB, no tracking. Privacy-first.

### 🚀 Run Locally
```bash
pip install streamlit plotly pandas numpy
streamlit run app.py
# open http://localhost:8501
```

### 📁 CSV Schema (optional upload)
Required: `level, department, tenure_months, performance_rating, salary_lakhs, promotion_readiness`
Optional: `age, gender, employee_id, revenue_per_fte_lakh`
Demo generator includes `compa_ratio, is_regrettable, attrition_probability` automatically.

### 🧠 18 Assumptions Documented in App
Now includes: survival baseline, Weibull tenure peak, compa-ratio <0.85 = +12% risk, cascade x1.4 if >2 exits in same dept same month, ramp productivity, CoV 2.5 months revenue, queueing for TA capacity.

### About Builder
**Sanuj Krishnan** — 14+ yrs HR leadership at Amazon, AWS, Swvl, Trianz. Built this to bridge people strategy and data science.

### License
MIT — Use, fork, commercialize. Star if useful.

---
**v5.0 Roadmap:** Feature 3 Promotion Chain Graph + Feature 4 Skills Ledger (Buy/Build/Borrow/Bot) coming next.
