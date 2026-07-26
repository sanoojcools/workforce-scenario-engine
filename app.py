import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, warnings, re
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Workforce Scenario Engine v3.0", layout="wide")

# ========== API SETUP ==========
def get_kimi_client():
    try:
        from openai import OpenAI
        key = st.secrets.get("KIMI_API_KEY", "")
        if not key: return None, "KIMI_API_KEY not found"
        return OpenAI(api_key=key, base_url="https://api.moonshot.ai/v1"), None
    except Exception as e: return None, str(e)

def get_claude_client():
    try:
        import anthropic
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if not key: return None, "ANTHROPIC_API_KEY not found"
        return anthropic.Anthropic(api_key=key), None
    except Exception as e: return None, str(e)

def call_llm(messages, provider):
    if provider == "Kimi":
        client, err = get_kimi_client()
        if err: return None, err
        try:
            resp = client.chat.completions.create(model="kimi-k3", messages=messages, temperature=0.3, max_tokens=2000)
            return resp.choices[0].message.content, None
        except Exception as e: return None, str(e)
    elif provider == "Claude":
        client, err = get_claude_client()
        if err: return None, err
        try:
            msgs = [{"role":m["role"], "content":m["content"]} for m in messages if m["role"]!="system"]
            sys_msg = next((m["content"] for m in messages if m["role"]=="system"), "")
            resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=2000, temperature=0.3, system=sys_msg, messages=msgs)
            return resp.content[0].text, None
        except Exception as e: return None, str(e)
    return None, "Unknown provider"

# ========== ATTRITION MODEL ==========
def compute_attrition_probability(df):
    df = df.copy()
    risk = np.full(len(df), 0.15)
    perf_map = {"LE (Low)":0.25,"ME (Meets)":0.05,"EE (Exceeds)":-0.03,"GE (Greatly Exceeds)":-0.08}
    for perf, delta in perf_map.items():
        risk += np.where(df["performance_rating"]==perf, delta, 0)
    tenure = df["tenure_months"].astype(float)
    risk += np.where((tenure>=12)&(tenure<=24), 0.12, 0)
    risk += np.where(tenure<6, 0.08, 0)
    risk += np.where(tenure>60, -0.10, 0)
    dept = df["department"].astype(str).str.lower()
    risk += np.where(dept.str.contains("sales|operations"), 0.08, 0)
    risk += np.where(dept.str.contains("engineer|tech|rd"), 0.05, 0)
    level = df["level"].astype(str)
    risk += np.where(level.isin(["L4","L5"]), 0.06, 0)
    med = df.groupby("level")["salary_lakhs"].transform("median").replace(0, np.nan)
    ratio = df["salary_lakhs"]/med
    risk += np.where(ratio<0.85, 0.10, 0)
    risk += np.where(ratio>1.20, -0.08, 0)
    ready_map = {"Ready Now":0.07,"Ready 1-2Y":0.02,"Ready 2+Y":0.0,"Not Ready":0.0}
    for ready, delta in ready_map.items():
        risk += np.where(df["promotion_readiness"]==ready, delta, 0)
    gender = df["gender"].astype(str).str.upper()
    risk += np.where((gender=="F")&(dept.str.contains("engineer|tech|rd")), 0.03, 0)
    df["attrition_probability"] = np.clip(risk, 0.02, 0.85)
    conds = [df["attrition_probability"]<0.10, df["attrition_probability"]<0.20, df["attrition_probability"]<0.35, df["attrition_probability"]>=0.35]
    df["attrition_risk_bucket"] = np.select(conds, ["Low","Medium","High","Critical"], default="Medium")
    return df

# ========== DEMO DATA ==========
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
    return df

# ========== SIMULATION ENGINE ==========
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
                     cost_per_hire_lakhs, promotion_cycle_months, restructuring=None):
        current = self.baseline.copy()
        history = []
        backfill_queue = {}
        total_severance = 0.0
        total_restructure = 0
        hire_total = sum(hire_dist.values())
        if hire_total == 0:
            hire_dist = {"L3":0.41,"L4":0.33,"L5":0.21,"L6":0.05}
        else:
            hire_dist = {k:v/hire_total for k,v in hire_dist.items()}
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
            current["monthly_attrition_prob"] = current["attrition_probability"] / 12 * attrition_multiplier
            current["leaves_this_month"] = np.random.binomial(1, current["monthly_attrition_prob"])
            leavers = current[current["leaves_this_month"]==1].copy()
            current = current[current["leaves_this_month"]==0].copy()
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
            n_hire = planned_hires_per_month + backfill_queue.get(month,0)
            if n_hire > 0:
                hire_levels = np.random.choice(list(hire_dist.keys()), n_hire, p=list(hire_dist.values()))
                depts_pool = current["department"].unique() if len(current)>0 else ["Engineering","Finance","HR","Sales","Product","Operations"]
                new_hires = pd.DataFrame({"employee_id":[f"HIRE_{month}_{i}" for i in range(n_hire)],"level":hire_levels,"department":np.random.choice(depts_pool,n_hire),"tenure_months":np.random.randint(1,12,n_hire),"performance_rating":np.random.choice(["ME (Meets)","EE (Exceeds)","LE (Low)"],n_hire,p=[0.6,0.3,0.1]),"salary_lakhs":[{"L3":12,"L4":22,"L5":40,"L6":70}.get(l,12)*np.random.uniform(0.9,1.1) for l in hire_levels],"age":np.random.randint(24,34,n_hire),"gender":np.random.choice(["M","F","NB"],n_hire,p=[0.60,0.38,0.02]),"promotion_readiness":["Not Ready"]*n_hire,"attrition_probability":0.15,"attrition_risk_bucket":"Medium","manager_id":-1})
                current = pd.concat([current, new_hires], ignore_index=True)
            current["salary_lakhs"] *= (1 + salary_inflation/12/100)
            monthly_payroll = current["salary_lakhs"].sum() / 100
            hire_cost = n_hire * cost_per_hire_lakhs / 100
            total_cost = monthly_payroll + hire_cost + re_sev
            history.append({"month":month,"headcount":len(current),"monthly_cost_cr":round(total_cost,2),"payroll_cr":round(monthly_payroll,2),"hire_cost_cr":round(hire_cost,2),"severance_cr":round(re_sev,2),"attrition":len(leavers),"hires":n_hire,"promotions":n_promote,"restructure_exits":re_exits})
        return pd.DataFrame(history), current, total_severance, total_restructure

# ========== TORNADO ==========
def tornado_analysis(sim, base_params, var_name, var_range, restructuring=None):
    results = []
    for val in var_range:
        p = base_params.copy(); p[var_name] = val
        np.random.seed(42)
        hist, _, _, _ = sim.run_scenario(**p, restructuring=restructuring)
        results.append({var_name:val,"final_headcount":hist["headcount"].iloc[-1],"final_cost":hist["monthly_cost_cr"].iloc[-1]})
    return pd.DataFrame(results)

# ========== RULE-BASED INSIGHTS ==========
def generate_insights(baseline, hist, final_df, params, total_sev=0, total_re=0):
    insights = []
    net = hist["headcount"].iloc[-1] - len(baseline)
    if total_re > 0:
        insights.append(("🔧 Restructuring", str(total_re)+" cut, ₹"+str(round(total_sev,2))+"Cr severance. Net: "+str(net)+" heads.", "Critical"))
    if net > 100: insights.append(("🔥 Aggressive Growth", "Net +" + str(net) + " heads. Scale leadership bench.", "High"))
    elif net < 0: insights.append(("⚠️ Shrinking", "Net " + str(net) + " heads. Review critical roles.", "Critical"))
    else: insights.append(("📊 Moderate Growth", "Net +" + str(net) + " heads. Sustainable.", "Medium"))
    cg = (hist["monthly_cost_cr"].iloc[-1]/hist["monthly_cost_cr"].iloc[0]-1)*100
    sev = "High" if cg>30 else "Medium"
    insights.append(("💰 Cost", "Monthly cost up " + str(round(cg,1)) + "%.", sev))
    ta = hist["attrition"].sum(); ar = ta/len(baseline)*100
    sev = "Critical" if ar>25 else "High" if ar>15 else "Medium"
    insights.append(("🚪 Attrition", str(ta)+" exits ("+str(round(ar,1))+"% annualized).", sev))
    rn = len(baseline[baseline["promotion_readiness"]=="Ready Now"])
    tp = hist["promotions"].sum(); ratio = rn/max(tp,1)
    if ratio > 10: insights.append(("🎯 Chokepoint", str(rn)+" Ready → "+str(tp)+" promos ("+str(int(ratio))+":1). Flight risk elevated.", "Critical"))
    else: insights.append(("✅ Promotions", "Healthy "+str(round(ratio,1))+":1 ratio.", "Low"))
    dr = baseline.groupby("department").apply(lambda x:(x["attrition_risk_bucket"].isin(["High","Critical"])).mean()*100).sort_values(ascending=False)
    if len(dr)>0 and dr.iloc[0]>60:
        insights.append(("🏢 Dept Risk", str(dr.index[0])+": "+str(int(dr.iloc[0]))+"% High/Critical risk.", "Critical"))
    l5p = (final_df["level"].isin(["L5","L6","L7","L8"])).mean()*100
    ld = "Dilution risk" if l5p<25 else "Healthy"
    sev = "High" if l5p<25 else "Medium"
    insights.append(("🏗️ Leadership", "L5+: "+str(round(l5p,1))+"%. "+ld+".", sev))
    return insights

# ========== AI AGENT TOOLS ==========
def ai_run_scenario(sim, params, restructuring=None):
    np.random.seed(42)
    hist, final_df, sev, re = sim.run_scenario(**params, restructuring=restructuring)
    return {
        "final_headcount": int(hist["headcount"].iloc[-1]),
        "final_cost_cr": float(hist["monthly_cost_cr"].iloc[-1]),
        "total_attrition": int(hist["attrition"].sum()),
        "total_hires": int(hist["hires"].sum()),
        "total_promotions": int(hist["promotions"].sum()),
        "total_severance_cr": round(float(sev), 2),
        "total_restructure": int(re),
        "headcount_trajectory": hist["headcount"].tolist(),
        "cost_trajectory": hist["monthly_cost_cr"].tolist()
    }

def ai_analyze_department(df, dept_name):
    dept_df = df[df["department"] == dept_name]
    if len(dept_df) == 0: return {"error": "Department not found"}
    return {
        "employee_count": len(dept_df),
        "avg_tenure_months": round(float(dept_df["tenure_months"].mean()), 1),
        "high_critical_risk_pct": round(float((dept_df["attrition_risk_bucket"].isin(["High","Critical"])).mean()*100), 1),
        "avg_salary_lakhs": round(float(dept_df["salary_lakhs"].mean()), 2),
        "ready_now_count": int((dept_df["promotion_readiness"]=="Ready Now").sum()),
        "level_distribution": {k:int(v) for k,v in dept_df["level"].value_counts().to_dict().items()},
        "performance_distribution": {k:int(v) for k,v in dept_df["performance_rating"].value_counts().to_dict().items()}
    }

def ai_optimize(sim, params, target_headcount, max_cost, restructuring=None):
    best = None; best_score = float("inf")
    search_space = []
    for hires in range(0, 51, 5):
        for attr_mult in [0.7, 0.85, 1.0, 1.1, 1.3]:
            for infl in [0, 3, 6, 9, 12]:
                search_space.append({"planned_hires_per_month": hires, "attrition_multiplier": attr_mult, "salary_inflation": infl})
    for p in search_space:
        test_params = params.copy(); test_params.update(p)
        np.random.seed(42)
        hist, _, _, _ = sim.run_scenario(**test_params, restructuring=restructuring)
        hc = hist["headcount"].iloc[-1]; cost = hist["monthly_cost_cr"].iloc[-1]
        if cost <= max_cost:
            score = abs(hc - target_headcount)
            if score < best_score:
                best_score = score
                best = {"params": p, "headcount": int(hc), "cost": round(float(cost), 2)}
    return best or {"error": "No solution found"}

def ai_run_restructuring(sim, params, cut_count, criteria, protected_levels, protected_depts, severance_months, spread_months, backfill):
    restruct = {"enabled": True, "cut_count": cut_count, "criteria": criteria,
                "protected_levels": protected_levels, "protected_depts": protected_depts,
                "severance_months": severance_months, "spread_months": spread_months, "backfill": backfill}
    np.random.seed(42)
    hist, final_df, sev, re = sim.run_scenario(**params, restructuring=restruct)
    baseline_payroll = params.get("payroll_baseline", hist["payroll_cr"].iloc[0])
    monthly_savings = baseline_payroll - hist["payroll_cr"].iloc[-1]
    break_even = int(sev / max(monthly_savings, 0.01)) if monthly_savings > 0 else None
    return {
        "final_headcount": int(hist["headcount"].iloc[-1]),
        "final_cost_cr": float(hist["monthly_cost_cr"].iloc[-1]),
        "total_severance_cr": round(float(sev), 2),
        "positions_cut": int(re),
        "break_even_months": break_even,
        "natural_attrition": int(hist["attrition"].sum()),
        "headcount_trajectory": hist["headcount"].tolist()
    }

# ========== AI AGENT PROCESSOR ==========
SYSTEM_PROMPT = """You are the Workforce Scenario Engine AI — a strategic workforce planning advisor with access to simulation tools.

You can use these tools by outputting EXACTLY:
TOOL_CALL: {"tool": "TOOL_NAME", "params": {...}}

Available tools:
1. run_scenario — Run a workforce simulation
   Params: planned_hires_per_month (int), promotion_rate (float 0-0.3), salary_inflation (float 0-20), attrition_multiplier (float 0.5-1.5)
   Returns: final_headcount, final_cost_cr, total_attrition, total_hires, headcount_trajectory, cost_trajectory

2. analyze_department — Deep dive into a department
   Params: department_name (string)
   Returns: employee_count, avg_tenure, high_critical_risk_pct, avg_salary, ready_now_count, level_distribution, performance_distribution

3. optimize — Find best parameters to hit targets
   Params: target_headcount (int), max_cost (float in Cr)
   Returns: best_params, achieved_headcount, achieved_cost

4. run_restructuring — Model layoffs/restructuring
   Params: cut_count (int), criteria (string: lifo/performance/cost/level/random), protected_levels (list), protected_depts (list), severance_months (int), spread_months (int), backfill (bool)
   Returns: final_headcount, final_cost_cr, total_severance_cr, positions_cut, break_even_months, natural_attrition

RULES:
- If the user asks a "what if", use run_scenario with modified params.
- If they ask "why" or "analyze", use analyze_department.
- If they ask "best way" or "optimal", use optimize.
- If they ask about layoffs/restructuring, use run_restructuring.
- You may chain up to 3 tool calls. Output each on its own line.
- After tool results, provide a concise, strategic answer with specific numbers.
- Use bullet points. Bold key numbers. Keep under 200 words.
"""

def parse_tool_calls(text):
    calls = []
    for line in text.split("\n"):
        if line.strip().startswith("TOOL_CALL:"):
            try:
                json_str = line.strip().replace("TOOL_CALL:", "").strip()
                calls.append(json.loads(json_str))
            except: pass
    return calls

def execute_tool(tool_name, tool_params, sim, params, df, restructuring_cfg):
    if tool_name == "run_scenario":
        p = params.copy(); p.update(tool_params)
        return ai_run_scenario(sim, p, restructuring_cfg)
    elif tool_name == "analyze_department":
        return ai_analyze_department(df, tool_params.get("department_name", "Engineering"))
    elif tool_name == "optimize":
        return ai_optimize(sim, params, tool_params.get("target_headcount", 600), tool_params.get("max_cost", 250), restructuring_cfg)
    elif tool_name == "run_restructuring":
        return ai_run_restructuring(sim, params, **tool_params)
    return {"error": "Unknown tool"}

def process_ai_question(question, provider, sim, params, df, hist, final_df, restructuring_cfg):
    baseline_summary = {
        "total_employees": len(df),
        "monthly_payroll_cr": round(df["salary_lakhs"].sum()/100, 2),
        "high_critical_risk_count": int((df["attrition_risk_bucket"].isin(["High","Critical"])).sum()),
        "high_critical_risk_pct": round((df["attrition_risk_bucket"].isin(["High","Critical"])).mean()*100, 1),
        "ready_now_count": int((df["promotion_readiness"]=="Ready Now").sum()),
        "departments": {k:int(v) for k,v in df["department"].value_counts().to_dict().items()},
        "levels": {k:int(v) for k,v in df["level"].value_counts().to_dict().items()}
    }
    scenario_results = {
        "final_headcount": int(hist["headcount"].iloc[-1]),
        "final_cost_cr": float(hist["monthly_cost_cr"].iloc[-1]),
        "total_attrition": int(hist["attrition"].sum()),
        "total_hires": int(hist["hires"].sum()),
        "total_promotions": int(hist["promotions"].sum())
    }
    context = "BASELINE: " + json.dumps(baseline_summary) + "\nCURRENT_PARAMS: " + json.dumps({k:(float(v) if isinstance(v,(int,float)) else v) for k,v in params.items() if k!="hire_dist"}) + "\nSCENARIO_RESULTS: " + json.dumps(scenario_results) + "\nRESTRUCTURING: " + (json.dumps(restructuring_cfg) if restructuring_cfg else "None")

    if provider == "Stochastic (Rule-Based)":
        return stochastic_answer(question, sim, params, df, hist, final_df, restructuring_cfg)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": context + "\n\nQuestion: " + question}]
    response, err = call_llm(messages, provider)
    if err:
        return "⚠️ AI Error: " + err + "\n\nFalling back to rule-based...\n\n" + stochastic_answer(question, sim, params, df, hist, final_df, restructuring_cfg)

    tool_calls = parse_tool_calls(response)
    if tool_calls:
        results_text = []
        for tc in tool_calls:
            result = execute_tool(tc["tool"], tc.get("params", {}), sim, params, df, restructuring_cfg)
            results_text.append("Result of " + tc["tool"] + ": " + json.dumps(result))
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": "Tool results:\n" + "\n".join(results_text) + "\n\nNow provide your final strategic answer. Be concise, use bullets, bold key numbers."})
        final_response, err = call_llm(messages, provider)
        if err: return "⚠️ AI synthesis error: " + err + "\n\nTool results:\n" + "\n".join(results_text)
        return "🤖 **" + provider + " Agent**\n\n" + final_response
    return "🤖 **" + provider + " Agent**\n\n" + response

def stochastic_answer(question, sim, params, df, hist, final_df, restructuring_cfg):
    q = question.lower()
    baseline_hc = len(df)
    current_hc = int(hist["headcount"].iloc[-1])
    current_cost = float(hist["monthly_cost_cr"].iloc[-1])

    if "freeze" in q or ("hiring" in q and "0" in q) or ("stop" in q and "hire" in q):
        p = params.copy(); p["planned_hires_per_month"] = 0
        np.random.seed(42)
        h, f, _, _ = sim.run_scenario(**p, restructuring=restructuring_cfg)
        return "**🧊 Hiring Freeze Analysis**\n\nWith **0 planned hires/month**:\n- Final headcount: **" + str(h["headcount"].iloc[-1]) + "** (vs " + str(current_hc) + " current)\n- Monthly cost: **₹" + str(round(h["monthly_cost_cr"].iloc[-1],2)) + " Cr** (vs ₹" + str(round(current_cost,2)) + " Cr)\n- Natural attrition: **" + str(h["attrition"].sum()) + "** exits over 12 months\n- Net change: **" + str(h["headcount"].iloc[-1] - baseline_hc) + "** heads\n\n⚠️ **Risk:** Workforce shrinks by ~" + str(current_hc - h["headcount"].iloc[-1]) + " with no backfill."

    dept_match = re.search(r'(engineering|sales|finance|hr|product|operations)', q)
    if dept_match or ("department" in q and "risk" in q):
        dept = dept_match.group(1).capitalize() if dept_match else df["department"].value_counts().index[0]
        result = ai_analyze_department(df, dept)
        if "error" not in result:
            action = "Targeted retention for 12-24mo tenure band" if result["avg_tenure_months"] < 30 else "Address salary compression vs level median" if result["avg_salary_lakhs"] < 30 else "Manager quality intervention + growth conversations"
            return "**🏢 " + dept + " Department Analysis**\n\n- **Headcount:** " + str(result["employee_count"]) + " employees\n- **Flight Risk:** " + str(round(result["high_critical_risk_pct"],0)) + "% at High/Critical\n- **Avg Tenure:** " + str(round(result["avg_tenure_months"],0)) + " months\n- **Avg Salary:** ₹" + str(round(result["avg_salary_lakhs"],1)) + " Lakhs\n- **Ready Now:** " + str(result["ready_now_count"]) + " people\n\n💡 **Action:** " + action

    if any(w in q for w in ["optimal", "cheapest", "best plan", "grow", "target"]):
        target = 600; max_c = 250
        nm = re.search(r'(\d+)%', q)
        if nm: target = int(baseline_hc * (1 + int(nm.group(1))/100))
        nm2 = re.search(r'to (\d+)', q)
        if nm2: target = int(nm2.group(1))
        result = ai_optimize(sim, params, target, max_c, restructuring_cfg)
        if "error" not in result:
            p = result["params"]
            return "**🎯 Optimization: Reach " + str(target) + " employees under ₹" + str(max_c) + " Cr**\n\nBest parameters:\n- **Hires/month:** " + str(p["planned_hires_per_month"]) + "\n- **Attrition multiplier:** " + str(p["attrition_multiplier"]) + "x\n- **Salary inflation:** " + str(p["salary_inflation"]) + "%\n\n**Result:** " + str(result["headcount"]) + " employees at ₹" + str(result["cost"]) + " Cr/month\n\n📊 " + str(abs(result["headcount"] - target)) + " heads " + ("over" if result["headcount"] > target else "under") + " target."

    if any(w in q for w in ["layoff", "restructure", "cut", "reduce headcount", "downsize"]):
        return "**🔧 Restructuring Analysis**\n\nUse the **Restructuring panel** in the sidebar to model layoffs. Set:\n- Cut target (headcount or %)\n- Criteria: LIFO, Performance-based, Cost-based, or Level-based\n- Protected levels/departments\n- Severance months & spread\n\nThen ask me: *'What if I cut 50 people by performance with 3 months severance?'* and I'll run the simulation."

    if "compare" in q or "vs" in q or "versus" in q:
        return "**📊 Comparison Tip**\n\nI can compare scenarios, but you'll need to specify which two. Try:\n- *'Compare Status Quo vs Aggressive Growth'*\n- *'What's the cost difference between 10 hires and 20 hires?'*\n\nOr adjust the sliders and presets to see live comparison charts above."

    if "promotion" in q and ("backlog" in q or "choke" in q or "ready now" in q):
        rn = len(df[df["promotion_readiness"]=="Ready Now"])
        tp = hist["promotions"].sum()
        ratio = rn / max(tp, 1)
        needed = max(0, rn - tp)
        return "**🎯 Promotion Pipeline Analysis**\n\n- **Ready Now:** " + str(rn) + " employees\n- **Projected promotions:** " + str(tp) + " over 12 months\n- **Ratio:** " + str(round(ratio,1)) + ":1\n\nTo clear the backlog, you need **" + str(needed) + " additional promotions** or will face elevated flight risk among top performers.\n\n💡 **Action:** Increase promotion rate to " + str(round(rn/len(df)*100,1)) + "% or create parallel growth tracks."

    # Default fallback
    return "**🤖 Stochastic Agent**\n\nI analyzed your question: *\"" + question + "\"*\n\nBased on the current simulation:\n- **Headcount:** " + str(current_hc) + " (net " + str(current_hc - baseline_hc) + ")\n- **Cost:** ₹" + str(round(current_cost,2)) + " Cr/month\n- **Attrition:** " + str(hist["attrition"].sum()) + " projected exits\n- **Hires:** " + str(hist["hires"].sum()) + " total\n\nTry asking more specific questions like:\n- *'What if I freeze hiring?'*\n- *'Analyze Engineering department'*\n- *'Find optimal plan to grow 20%'*\n- *'Which department has highest flight risk?'*"

# ========== STREAMLIT UI ==========
st.title("🧮 Workforce Scenario Engine v3.0")
st.markdown("Agentic workforce planning with AI-powered simulation, analysis & strategic Q&A")

# Session state init
defaults = {
    "planned_hires": 8, "promotion_rate": 0.09, "salary_inflation": 6.0, "attrition_multiplier": 1.0,
    "ai_provider": "Stochastic (Rule-Based)", "scenario_preset": "Custom", "last_preset": "Custom",
    "hire_l3": 41, "hire_l4": 33, "hire_l5": 21, "hire_l6": 5,
    "promo_bump": 20, "backfill_delay": 1, "cost_per_hire": 3.0, "promotion_cycle": 3,
    "data_source": "demo", "chat_history": [],
    "restructure_enabled": False, "cut_count": 0, "cut_pct": 0,
    "restructure_criteria": "performance", "protected_levels": ["L3"],
    "protected_depts": [], "severance_months": 3, "spread_months": 1,
    "backfill_layoffs": False
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---- SIDEBAR ----
with st.sidebar:
    st.header("🤖 AI Insights")
    ai_provider = st.selectbox("Provider", ["Stochastic (Rule-Based)", "Kimi", "Claude"], key="ai_provider")
    if ai_provider != "Stochastic (Rule-Based)":
        client_test = get_kimi_client() if ai_provider == "Kimi" else get_claude_client()
        if client_test[1]:
            st.error("❌ " + client_test[1])
        else:
            st.success("✅ " + ai_provider + " connected")

    st.header("🎯 Scenario Preset")
    preset = st.selectbox("Preset", ["Custom", "Status Quo", "Aggressive Growth", "Cost Optimization"], key="scenario_preset")
    if preset != st.session_state.last_preset:
        if preset == "Status Quo":
            st.session_state.planned_hires = 8; st.session_state.promotion_rate = 0.09; st.session_state.salary_inflation = 6.0; st.session_state.attrition_multiplier = 1.0
        elif preset == "Aggressive Growth":
            st.session_state.planned_hires = 20; st.session_state.promotion_rate = 0.12; st.session_state.salary_inflation = 8.0; st.session_state.attrition_multiplier = 1.1
        elif preset == "Cost Optimization":
            st.session_state.planned_hires = 2; st.session_state.promotion_rate = 0.05; st.session_state.salary_inflation = 3.0; st.session_state.attrition_multiplier = 0.85
        st.session_state.last_preset = preset
        st.rerun()

    st.header("⚙️ Simulation Parameters")
    planned_hires = st.slider("Planned Hires / Month", 0, 50, st.session_state.planned_hires, key="planned_hires")
    promotion_rate = st.slider("Promotion Rate", 0.0, 0.30, st.session_state.promotion_rate, step=0.01, key="promotion_rate")
    salary_inflation = st.slider("Salary Inflation % (Annual)", 0.0, 20.0, st.session_state.salary_inflation, step=0.5, key="salary_inflation")
    attrition_multiplier = st.slider("Attrition Multiplier", 0.5, 1.5, st.session_state.attrition_multiplier, step=0.05, key="attrition_multiplier")

    with st.expander("🏗️ Hire Level Distribution", expanded=False):
        hire_l3 = st.slider("Hire % L3", 0, 100, st.session_state.hire_l3, key="hire_l3")
        hire_l4 = st.slider("Hire % L4", 0, 100, st.session_state.hire_l4, key="hire_l4")
        hire_l5 = st.slider("Hire % L5", 0, 100, st.session_state.hire_l5, key="hire_l5")
        hire_l6 = st.slider("Hire % L6", 0, 100, st.session_state.hire_l6, key="hire_l6")
        htot = hire_l3 + hire_l4 + hire_l5 + hire_l6
        if htot != 100 and htot > 0:
            st.warning("Sum = " + str(htot) + "%. Normalizing...")
            hire_l3 = int(hire_l3/htot*100); hire_l4 = int(hire_l4/htot*100)
            hire_l5 = int(hire_l5/htot*100); hire_l6 = 100 - hire_l3 - hire_l4 - hire_l5

    promo_bump = st.slider("Promotion Salary Bump %", 10, 50, st.session_state.promo_bump, key="promo_bump")
    backfill_delay = st.slider("Backfill Delay (months)", 0, 6, st.session_state.backfill_delay, key="backfill_delay")
    cost_per_hire = st.slider("Cost Per Hire (₹ Lakhs)", 0.0, 10.0, st.session_state.cost_per_hire, step=0.5, key="cost_per_hire")
    promotion_cycle = st.selectbox("Promotion Cycle (months)", [1,2,3,6,12], index=[1,2,3,6,12].index(st.session_state.promotion_cycle), key="promotion_cycle")

    # ---- RESTRUCTURING ----
    with st.expander("🔧 Restructuring / Layoffs", expanded=False):
        restructure_enabled = st.toggle("Enable Restructuring", key="restructure_enabled")
        if restructure_enabled:
            cut_type = st.radio("Cut by", ["Headcount", "Percentage"], key="cut_type")
            if cut_type == "Headcount":
                cut_count = st.number_input("Employees to cut", 0, 1000, st.session_state.cut_count, key="cut_count")
            else:
                cut_pct = st.slider("Percentage to cut", 0, 50, st.session_state.cut_pct, key="cut_pct")
                cut_count = int(500 * cut_pct / 100)  # Will be recalculated based on actual df size
            restructure_criteria = st.selectbox("Criteria", ["LIFO (Last In First Out)", "Performance (Lowest First)", "Cost (Highest Salary First)", "Level (Highest First)", "Random"], key="restructure_criteria")
            criteria_map = {"LIFO (Last In First Out)": "lifo", "Performance (Lowest First)": "performance", "Cost (Highest Salary First)": "cost", "Level (Highest First)": "level", "Random": "random"}
            protected_levels = st.multiselect("Protected Levels", ["L3","L4","L5","L6","L7","L8"], default=st.session_state.protected_levels, key="protected_levels")
            protected_depts = st.multiselect("Protected Departments", ["Engineering","Finance","HR","Sales","Product","Operations"], default=st.session_state.protected_depts, key="protected_depts")
            severance_months = st.slider("Severance (months salary)", 0, 12, st.session_state.severance_months, key="severance_months")
            spread_months = st.slider("Spread over (months)", 1, 6, st.session_state.spread_months, key="spread_months")
            backfill_layoffs = st.toggle("Backfill laid-off roles", key="backfill_layoffs")

    # ---- DATA SOURCE ----
    st.header("📁 Data Source")
    uploaded_file = st.file_uploader("Upload Workforce CSV", type=["csv"], key="workforce_uploader")
    if uploaded_file is not None:
        st.session_state.data_source = "upload"
        st.success("📄 Uploaded: " + uploaded_file.name)
    else:
        use_demo = st.checkbox("☑️ Use Demo Data (500 employees)", value=True, key="use_demo_checkbox")
        if use_demo:
            st.session_state.data_source = "demo"
            st.info("📊 Using demo data")
        else:
            st.session_state.data_source = "none"
            st.error("⚠️ Please upload a CSV or enable demo data")

# ---- LOAD DATA ----
if st.session_state.data_source == "upload" and uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = compute_attrition_probability(df)
        df["manager_id"] = df.get("manager_id", -1)
        st.sidebar.success("✅ Loaded " + str(len(df)) + " employees from upload")
    except Exception as e:
        st.error("❌ Error loading CSV: " + str(e))
        st.session_state.data_source = "demo"
        df = generate_demo_data()
        st.info("Fell back to demo data")
elif st.session_state.data_source == "demo":
    df = generate_demo_data()
    st.sidebar.success("✅ Loaded 500 demo employees")
else:
    st.error("👆 Please select a data source in the sidebar")
    st.stop()

# ---- RESTRUCTURING CONFIG ----
restruct_cfg = None
if st.session_state.get("restructure_enabled", False):
    restruct_cfg = {
        "enabled": True,
        "cut_count": st.session_state.get("cut_count", 0) if st.session_state.get("cut_type") == "Headcount" else int(len(df) * st.session_state.get("cut_pct", 0) / 100),
        "criteria": {"LIFO (Last In First Out)": "lifo", "Performance (Lowest First)": "performance", "Cost (Highest Salary First)": "cost", "Level (Highest First)": "level", "Random": "random"}.get(st.session_state.get("restructure_criteria", "performance"), "performance"),
        "protected_levels": st.session_state.get("protected_levels", ["L3"]),
        "protected_depts": st.session_state.get("protected_depts", []),
        "severance_months": st.session_state.get("severance_months", 3),
        "spread_months": st.session_state.get("spread_months", 1),
        "backfill": st.session_state.get("backfill_layoffs", False)
    }

# ---- BASELINE SUMMARY ----
st.subheader("📋 Baseline Workforce Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("👥 Total Employees", len(df))
c2.metric("💰 Monthly Payroll", "₹" + str(round(df["salary_lakhs"].sum()/100, 2)) + " Cr")
c3.metric("⚠️ High/Critical Risk", str((df["attrition_risk_bucket"].isin(["High","Critical"])).sum()), str(round((df["attrition_risk_bucket"].isin(["High","Critical"])).mean()*100, 0)) + "%")
c4.metric("🚀 Ready Now", str((df["promotion_readiness"]=="Ready Now").sum()))

c5, c6, c7, c8 = st.columns(4)
c5.metric("🏢 Largest Dept", df["department"].value_counts().index[0])
c6.metric("📅 Median Tenure", str(int(df["tenure_months"].median())) + " mo")
c7.metric("♀️ Female %", str(round((df["gender"]=="F").mean()*100, 0)) + "%")
c8.metric("🎂 Avg Age", str(int(df["age"].mean())) + " yrs")

# ---- ASSUMPTIONS ----
with st.expander("📌 Active Assumptions & Model Methodology", expanded=False):
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.markdown("**Hires/Month:** " + str(planned_hires))
    a2.markdown("**Promo Rate:** " + str(round(promotion_rate*100, 0)) + "%")
    a3.markdown("**Inflation:** " + str(salary_inflation) + "%")
    a4.markdown("**Attrition Mult:** " + str(attrition_multiplier) + "x")
    a5.markdown("**Backfill Delay:** " + str(backfill_delay) + "mo")
    st.divider()
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
        st.markdown("**" + icon_title + "** — " + desc)
        st.caption("⚠️ Impact if false: " + impact)
    st.info("📌 All projections are directional estimates. Validate against historical ground truth before board presentation.")

# ---- RUN SIMULATION ----
hire_dist = {"L3": hire_l3, "L4": hire_l4, "L5": hire_l5, "L6": hire_l6}
sim = WorkforceSimulator(df)
params = {
    "planned_hires_per_month": planned_hires, "promotion_rate": promotion_rate,
    "salary_inflation": salary_inflation, "attrition_multiplier": attrition_multiplier,
    "hire_dist": hire_dist, "promo_bump_pct": promo_bump, "backfill_delay": backfill_delay,
    "cost_per_hire_lakhs": cost_per_hire, "promotion_cycle_months": promotion_cycle
}

np.random.seed(42)
hist, final_df, total_sev, total_re = sim.run_scenario(**params, restructuring=restruct_cfg)

np.random.seed(42)
bp = params.copy(); bp["planned_hires_per_month"] = 0; bp["attrition_multiplier"] = 1.0; bp["salary_inflation"] = 0.0
hist_base, _, _, _ = sim.run_scenario(**bp, restructuring=None)

# ---- SCENARIO METRICS ----
st.subheader("📊 Scenario Metrics")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("👥 Final Headcount", hist["headcount"].iloc[-1], int(hist["headcount"].iloc[-1] - len(df)))
m2.metric("💰 Monthly Cost", "₹" + str(round(hist["monthly_cost_cr"].iloc[-1], 2)) + " Cr")
m3.metric("🚪 Total Attrition", hist["attrition"].sum())
m4.metric("📥 Total Hires", hist["hires"].sum())
m5.metric("📈 Promotions", hist["promotions"].sum())

if total_re > 0:
    st.warning("🔧 Restructuring Active: " + str(total_re) + " positions cut, ₹" + str(round(total_sev, 2)) + " Cr severance")

# ---- TRAJECTORY CHARTS ----
st.subheader("📈 Trajectory Charts")
col1, col2 = st.columns(2)
with col1:
    fig_hc = go.Figure()
    fig_hc.add_trace(go.Scatter(x=hist["month"], y=hist["headcount"], mode="lines+markers", name="Scenario", line=dict(color="#1f77b4", width=3)))
    fig_hc.add_trace(go.Scatter(x=hist_base["month"], y=hist_base["headcount"], mode="lines", name="Baseline (No hires)", line=dict(color="gray", dash="dash")))
    fig_hc.update_layout(title="Headcount Trajectory", xaxis_title="Month", yaxis_title="Headcount", height=350, template="plotly_white")
    st.plotly_chart(fig_hc, use_container_width=True)
with col2:
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(x=hist["month"], y=hist["monthly_cost_cr"], mode="lines+markers", name="Total Cost", line=dict(color="#ff7f0e", width=3)))
    fig_cost.add_trace(go.Scatter(x=hist["month"], y=hist["payroll_cr"], mode="lines", name="Payroll", line=dict(color="#2ca02c")))
    fig_cost.add_trace(go.Scatter(x=hist["month"], y=hist["hire_cost_cr"], mode="lines", name="Hire Cost", line=dict(color="#d62728")))
    if total_re > 0:
        fig_cost.add_trace(go.Scatter(x=hist["month"], y=hist["severance_cr"], mode="lines", name="Severance", line=dict(color="#9467bd", dash="dot")))
    fig_cost.update_layout(title="Cost Trajectory (₹ Cr)", xaxis_title="Month", yaxis_title="₹ Crores", height=350, template="plotly_white")
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
        r = tornado_analysis(sim, params, var, vals, restructuring=restruct_cfg)
        low = r["final_headcount"].min() - hist["headcount"].iloc[-1]
        high = r["final_headcount"].max() - hist["headcount"].iloc[-1]
        tornado_hc.append({"Variable": var.replace("_"," ").title(), "Low": low, "High": high})
    td_hc = pd.DataFrame(tornado_hc)
    fig_t1 = go.Figure()
    for _, row in td_hc.iterrows():
        fig_t1.add_trace(go.Bar(y=[row["Variable"]], x=[row["High"]], orientation="h", marker_color="#1f77b4", showlegend=False))
        fig_t1.add_trace(go.Bar(y=[row["Variable"]], x=[row["Low"]], orientation="h", marker_color="#d62728", showlegend=False))
    fig_t1.update_layout(barmode="overlay", xaxis_title="Impact on Final Headcount", height=300, template="plotly_white")
    st.plotly_chart(fig_t1, use_container_width=True)
with tc2:
    st.markdown("**Cost Sensitivity**")
    tornado_cost = []
    for var, vals in tornado_vars.items():
        r = tornado_analysis(sim, params, var, vals, restructuring=restruct_cfg)
        low = r["final_cost"].min() - hist["monthly_cost_cr"].iloc[-1]
        high = r["final_cost"].max() - hist["monthly_cost_cr"].iloc[-1]
        tornado_cost.append({"Variable": var.replace("_"," ").title(), "Low": low, "High": high})
    td_c = pd.DataFrame(tornado_cost)
    fig_t2 = go.Figure()
    for _, row in td_c.iterrows():
        fig_t2.add_trace(go.Bar(y=[row["Variable"]], x=[row["High"]], orientation="h", marker_color="#2ca02c", showlegend=False))
        fig_t2.add_trace(go.Bar(y=[row["Variable"]], x=[row["Low"]], orientation="h", marker_color="#ff7f0e", showlegend=False))
    fig_t2.update_layout(barmode="overlay", xaxis_title="Impact on Final Cost (₹ Cr)", height=300, template="plotly_white")
    st.plotly_chart(fig_t2, use_container_width=True)

# ---- COMPOSITION ----
st.subheader("📉 Workforce Composition")
co1, co2 = st.columns(2)
with co1:
    level_counts = final_df["level"].value_counts().reindex(["L3","L4","L5","L6","L7","L8"]).fillna(0)
    fig_donut = go.Figure(data=[go.Pie(labels=level_counts.index, values=level_counts.values, hole=0.4, marker_colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"])])
    fig_donut.update_layout(title="Final Level Distribution", height=350, template="plotly_white")
    st.plotly_chart(fig_donut, use_container_width=True)
with co2:
    fig_attr = go.Figure()
    fig_attr.add_trace(go.Bar(x=hist["month"], y=hist["attrition"], name="Natural Attrition", marker_color="#d62728"))
    if total_re > 0:
        fig_attr.add_trace(go.Bar(x=hist["month"], y=hist["restructure_exits"], name="Restructuring", marker_color="#9467bd"))
    fig_attr.add_trace(go.Scatter(x=hist["month"], y=hist["attrition"].cumsum() + hist["restructure_exits"].cumsum(), name="Cumulative", mode="lines+markers", yaxis="y2", line=dict(color="#ff7f0e", width=2)))
    fig_attr.update_layout(title="Attrition Curve", xaxis_title="Month", yaxis_title="Monthly Exits", yaxis2=dict(title="Cumulative", overlaying="y", side="right"), height=350, template="plotly_white", barmode="stack")
    st.plotly_chart(fig_attr, use_container_width=True)

# ---- AI INSIGHTS ----
st.subheader("🤖 Strategic Insights")
insights = generate_insights(df, hist, final_df, params, total_sev, total_re)
for title, text, severity in insights:
    color = {"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}.get(severity, "⚪")
    st.markdown(color + " **" + title + "** — " + text)

# ---- EXECUTIVE SUMMARY ----
st.subheader("📝 Executive Summary")
total_cost_delta = hist["monthly_cost_cr"].iloc[-1] - hist["monthly_cost_cr"].iloc[0]
st.markdown("Over 12 months, this scenario projects **" + str(hist["headcount"].iloc[-1]) + " employees** (net " + ("+" if hist["headcount"].iloc[-1] > len(df) else "") + str(hist["headcount"].iloc[-1] - len(df)) + "), with a monthly cost of **₹" + str(round(hist["monthly_cost_cr"].iloc[-1], 2)) + " Cr** (" + ("+" if total_cost_delta > 0 else "") + "₹" + str(round(total_cost_delta, 2)) + " Cr). **" + str(hist["attrition"].sum()) + "** natural exits expected against **" + str(hist["hires"].sum()) + "** hires (" + str(hist["promotions"].sum()) + " promotions). The biggest levers are hiring volume and attrition containment.")

# ========== CHAT PANEL ==========
st.subheader("💬 Ask the Workforce AI")

sample_questions = [
    "What if I freeze hiring for 3 months?",
    "Which department has the highest flight risk?",
    "Find the cheapest plan to grow 20%",
    "Should I hire L5s or promote L4s?",
    "What's the impact of cutting 10% by performance?",
    "Compare this vs Status Quo",
    "How do I clear the Ready Now backlog?",
    "What's my break-even on retention bonuses?"
]

cols = st.columns(4)
for i, q in enumerate(sample_questions):
    with cols[i % 4]:
        if st.button(q, key="chip_" + str(i), use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            st.session_state.pending_question = q
            st.rerun()

# Process pending question
if "pending_question" in st.session_state:
    q = st.session_state.pending_question
    del st.session_state.pending_question
    with st.spinner("🤖 " + ai_provider + " Agent is thinking..."):
        answer = process_ai_question(q, ai_provider, sim, params, df, hist, final_df, restruct_cfg)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a strategic question about your workforce..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.pending_question = prompt
    st.rerun()

# ---- DOWNLOADS ----
st.subheader("⬇️ Export")
dc1, dc2 = st.columns(2)
with dc1:
    csv = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Final Workforce CSV", csv, "workforce_projection.csv", "text/csv")
with dc2:
    summary = {
        "parameters": {k:(float(v) if isinstance(v,(int,float)) else v) for k,v in params.items() if k!="hire_dist"},
        "baseline_headcount": int(len(df)), "final_headcount": int(hist["headcount"].iloc[-1]),
        "total_attrition": int(hist["attrition"].sum()), "total_hires": int(hist["hires"].sum()),
        "total_promotions": int(hist["promotions"].sum()), "final_monthly_cost_cr": float(hist["monthly_cost_cr"].iloc[-1]),
        "restructuring": {"enabled": restruct_cfg is not None, "total_severance_cr": float(total_sev), "positions_cut": int(total_re)} if restruct_cfg else None
    }
    st.download_button("📥 Download Scenario JSON", json.dumps(summary, indent=2), "scenario_summary.json", "application/json")
