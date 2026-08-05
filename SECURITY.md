# 🔒 Security & Privacy - Workforce Scenario Engine v5.0

### Philosophy: Privacy First, Demo Synthetic

**1. Demo Data Only**
- Default dataset is 100% synthetic: 500 employees (EMP_0001 etc), generated with numpy
- No real names, emails, or real salaries. Salary in Lakhs is randomized.
- Tenure, performance, readiness distributions are realistic but fake.

**2. Do NOT Upload Real PII**
- We explicitly warn in UI: "Do not upload real PII (names, emails)"
- If you must upload, anonymize first: replace names with IDs, emails removed.
- App expects: level, department, tenure_months, performance_rating, salary_lakhs, promotion_readiness
- Optional revenue_per_fte_lakh is aggregated, not personal.

**3. No Server-Side Storage**
- Streamlit is stateless: data lives in session memory, resets on refresh / browser close
- No database, no S3, no logging of uploaded CSVs
- We never store employee_id beyond session

**4. API Keys**
- KIMI_API_KEY / ANTHROPIC_API_KEY stored in Streamlit secrets (`.streamlit/secrets.toml`), never in code
- UI shows generic "AI co-pilot offline" if key missing, never leaks key names
- Code: `st.secrets.get("KIMI_API_KEY","")` with fallback check

**5. New v5.0 Data Added**
- Feature 1 adds `compa_ratio`, `is_regrettable`, `attrition_probability` - all derived, not PII
- Feature 2 adds `revenue_per_fte_lakh` - aggregated revenue assumption, not salary
- Feature 5 Pareto results are in-memory DataFrame, not persisted

**6. File Limits**
- `MAX_UPLOAD_MB = 10` enforced in `validate_csv()`
- Required cols check: level, department, tenure_months, performance_rating, salary_lakhs, promotion_readiness

**7. Open Source Transparency**
- All 18 assumptions documented in app expander with impact if false
- Monte Carlo seed centralized via RANDOM_SEED_BASE
- Code is MIT, auditable on GitHub

**8. What We DON'T Do**
- No tracking cookies
- No external analytics
- No email collection
- No employee data sent to LLM (LLM only gets aggregated baseline_summary, not individual rows)

**If you are a CHRO evaluating:** Run locally (`streamlit run app.py`) behind VPN for full control. Synthetic demo is safe for public Streamlit Cloud.

Report security issues: Open a GitHub issue (do not include real data).
