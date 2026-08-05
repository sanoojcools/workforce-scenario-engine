# Business Case - Workforce Scenario Engine v5.0

### Problem
1. CHROs rebuild Excel headcount models for 2 days every time Finance asks what-if. Answers are single-point, no confidence.
2. Finance plans are unconstrained: "Hire 120 in 90 days" with 5 recruiters is mathematically impossible, but no tool shows it.
3. Retention budgets (₹ crores) are allocated with zero ROI math.
4. Scenario planning is manual: 3 scenarios vs 300 possible.

Result: Board loses confidence, CoV (Cost of Vacancy) bleeds revenue, regrettable attrition.

### Solution v5.0: Three Engines, Not One Calculator

**Engine 1 - Causal Survival Attrition (Feature 1):**
- Weibull + Cox hazard model, not flat 15%
- Includes compa-ratio, tenure curve, cascade contagion
- Output: Ranked flight risk + regrettable flag + retention ROI optimizer
- HR Value: "Spend ₹12L on 8 Critical regrettable saves ₹45L net, ROI 275%"

**Engine 2 - Talent Supply Chain (Feature 2):**
- Models hiring as M/M/c queue: max_hires = recruiters * reqs * accept_rate
- Tracks unfilled, effective HC (ramped), CoV
- Output: Breach alerts, Hires vs Unfilled bar, Headcount vs Capacity-Constrained trajectory
- CFO Value: "Aggressive plan leaves 42 roles unfilled = ₹3.2Cr CoV lost. Need 3 more recruiters."

**Engine 3 - Pareto Autonomous Architect (Feature 5):**
- Runs 50-500 trials across hires, attrition, promo, inflation, recruiters
- Multi-objective optimization: min Cost, max HC, min Risk%
- Output: Pareto frontier (black stars) - 73 optimal plans out of 200 in demo, top 5 table
- CEO Value: "Here are ALL efficient plans under ₹210Cr that hit 600 HC. Cheapest uses hiring freeze in Sales."

### ROI

**Time:** 95% saved: 300 scenarios in 2 min vs 3 days in Excel/Anaplan
**Money:**
- Avoid 1 regrettable exit: Save ₹38L CoV (2.5mo revenue) - retention optimizer finds them
- Avoid capacity breach: See CoV before it happens
- Better decisions: P10/P50/P90 bands, not point estimates

**Before v5.0:** "What's our final HC?" → Point estimate, no risk
**After v5.0:** "What's the probability we hit 550 HC under ₹160Cr with <30% risk and 5 recruiters?" → 68%, with 3 Pareto alternatives

### Differentiation vs Competitors
- Anaplan / Adaptive / Pigment: Finance-led, deterministic, no survival attrition, no TA capacity queue
- Visier / One Model: People analytics, descriptive, no Pareto optimization, no CoV
- ChartHop / Orgvue: Org charting, no Monte Carlo, no retention ROI, no autonomous architect
- This: HR-native, probabilistic, constrained, autonomous, open-source

### Moat
1. Proprietary survival params tuned on HR data
2. CoV + supply chain constraint logic (no one models recruiter capacity)
3. Pareto frontier from open-source optimization
4. Compounding: As users upload (opt-in), anonymized survival curves improve

### Built By
Sanuj Krishnan, CHRO Ex-Amazon/AWS, 14+ yrs. Bridges people strategy + data science.

### Ask
- Stars + feedback
- Design partners: 3 CHROs to test retention ROI with real (anonymized) data
- Next: Feature 3 Promotion Chain + Feature 4 Skills Ledger (Buy/Build/Borrow/Bot)

License: MIT
