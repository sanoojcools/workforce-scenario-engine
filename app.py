import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, warnings, re
warnings.filterwarnings('ignore')
st.set_page_config(page_title='Workforce Scenario Engine v3.0', layout='wide')

# ========== API SETUP ==========
def get_kimi_client():
    try:
        from openai import OpenAI
        key = st.secrets.get('KIMI_API_KEY', '')
        if not key: return None, 'KIMI_API_KEY not found'
        return OpenAI(api_key=key, base_url='https://api.moonshot.ai/v1'), None
    except Exception as e: return None, str(e)

def get_claude_client():
    try:
        import anthropic
        key = st.secrets.get('ANTHROPIC_API_KEY', '')
        if not key: return None, 'ANTHROPIC_API_KEY not found'
        return anthropic.Anthropic(api_key=key), None
    except Exception as e: return None, str(e)

def call_llm(messages, provider):
    if provider == 'Kimi':
        client, err = get_kimi_client()
        if err: return None, err
        try:
            resp = client.chat.completions.create(model='kimi-k3', messages=messages, temperature=0.3, max_tokens=2000)
            return resp.choices[0].message.content, None
        except Exception as e: return None, str(e)
    elif provider == 'Claude':
        client, err = get_claude_client()
        if err: return None, err
        try:
            msgs = [{'role':m['role'], 'content':m['content']} for m in messages if m['role']!='system']
            sys_msg = next((m['content'] for m in messages if m['role']=='system'), '')
            resp = client.messages.create(model='claude-sonnet-4-6', max_tokens=2000, temperature=0.3, system=sys_msg, messages=msgs)
            return resp.content[0].text, None
        except Exception as e: return None, str(e)
    return None, 'Unknown provider'

# ========== ATTRITION MODEL ==========
def compute_attrition_probability(df):
    df = df.copy()
    risk = np.full(len(df), 0.15)
    perf_map = {'LE (Low)':0.25,'ME (Meets)':0.05,'EE (Exceeds)':-0.03,'GE (Greatly Exceeds)':-0.08}
    for perf, delta in perf_map.items():
        risk += np.where(df['performance_rating']==perf, delta, 0)
    tenure = df['tenure_months'].astype(float)
    risk += np.where((tenure>=12)&(tenure<=24), 0.12, 0)
    risk += np.where(tenure<6, 0.08, 0)
    risk += np.where(tenure>60, -0.10, 0)
    dept = df['department'].astype(str).str.lower()
    risk += np.where(dept.str.contains('sales|operations'), 0.08, 0)
    risk += np.where(dept.str.contains('engineer|tech|rd'), 0.05, 0)
    level = df['level'].astype(str)
    risk += np.where(level.isin(['L4','L5']), 0.06, 0)
    med = df.groupby('level')['salary_lakhs'].transform('median').replace(0, np.nan)
    ratio = df['salary_lakhs']/med
    risk += np.where(ratio<0.85, 0.10, 0)
    risk += np.where(ratio>1.20, -0.08, 0)
    ready_map = {'Ready Now':0.07,'Ready 1-2Y':0.02,'Ready 2+Y':0.0,'Not Ready':0.0}
    for ready, delta in ready_map.items():
        risk += np.where(df['promotion_readiness']==ready, delta, 0)
    gender = df['gender'].astype(str).str.upper()
    risk += np.where((gender=='F')&(dept.str.contains('engineer|tech|rd')), 0.03, 0)
    df['attrition_probability'] = np.clip(risk, 0.02, 0.85)
    conds = [df['attrition_probability']<0.10, df['attrition_probability']<0.20, df['attrition_probability']<0.35, df['attrition_probability']>=0.35]
    df['attrition_risk_bucket'] = np.select(conds, ['Low','Medium','High','Critical'], default='Medium')
    return df

# ========== DEMO DATA ==========
@st.cache_data
def generate_demo_data(n=500, random_seed=42):
    np.random.seed(random_seed)
    def norm_p(p):
        p = np.array(p, dtype=float)
        return p / p.sum()
    levels = np.random.choice(['L3','L4','L5','L6','L7','L8'], n, p=norm_p([0.36,0.30,0.17,0.10,0.06,0.02]))
    depts = np.random.choice(['Engineering','Finance','HR','Sales','Product','Operations'], n, p=norm_p([0.18,0.19,0.17,0.16,0.16,0.14]))
    tenure = np.clip(np.random.gamma(3,12,n),0,180).astype(int)
    perf = np.random.choice(['LE (Low)','ME (Meets)','EE (Exceeds)','GE (Greatly Exceeds)'], n, p=norm_p([0.12,0.54,0.28,0.07]))
    age = np.clip(np.random.normal(32,7,n).astype(int),22,60)
    gender = np.random.choice(['M','F','NB'], n, p=norm_p([0.60,0.38,0.02]))
    readiness = np.random.choice(['Not Ready','Ready Now','Ready 1-2Y','Ready 2+Y'], n, p=norm_p([0.15,0.24,0.37,0.25]))
    lb = {'L3':12,'L4':22,'L5':40,'L6':70,'L7':100,'L8':200}
    pm = {'LE (Low)':0.85,'ME (Meets)':1.0,'EE (Exceeds)':1.15,'GE (Greatly Exceeds)':1.30}
    salary = np.array([lb[l]*np.random.uniform(0.8,1.2)*pm[p] for l,p in zip(levels,perf)])
    df = pd.DataFrame({'employee_id':[f'EMP_{i+1:04d}' for i in range(n)],'level':levels,'department':depts,'tenure_months':tenure,'performance_rating':perf,'salary_lakhs':np.round(salary,2),'age':age,'gender':gender,'promotion_readiness':readiness})
    df = compute_attrition_probability(df)
    df['manager_id'] = -1
    return df

# ========== SIMULATION ENGINE ==========
class WorkforceSimulator:
    def __init__(self, baseline_df):
        self.baseline = baseline_df.copy()
        self.months = 12

    def _select_targets(self, df, cfg, n):
        el = df.copy()
        if cfg.get('protected_levels'): el = el[~el['level'].isin(cfg['protected_levels'])]
        if cfg.get('protected_depts'): el = el[~el['department'].isin(cfg['protected_depts'])]
        crit = cfg.get('criteria','random')
        if crit=='lifo': return el.sort_values('tenure_months',ascending=True).head(n)
        elif crit=='performance':
            el = el.copy(); el['_s']=el['performance_rating'].map({'LE (Low)':0,'ME (Meets)':1,'EE (Exceeds)':2,'GE (Greatly Exceeds)':3})
            return el.sort_values('_s',ascending=True).head(n)
        elif crit=='cost': return el.sort_values('salary_lakhs',ascending=False).head(n)
        elif crit=='level':
            el = el.copy(); el['_s']=el['level'].map({'L8':0,'L7':1,'L6':2,'L5':3,'L4':4,'L3':5})
            return el.sort_values('_s',ascending=True).head(n)
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
            hire_dist = {'L3':0.41,'L4':0.33,'L5':0.21,'L6':0.05}
        else:
            hire_dist = {k:v/hire_total for k,v in hire_dist.items()}
        for month in range(1, self.months + 1):
            # Restructuring
            re_exits = 0; re_sev = 0.0
            if restructuring and restructuring.get('enabled') and restructuring.get('cut_count',0)>0:
                spread = max(1, restructuring.get('spread_months',1))
                if month <= spread:
                    monthly_cut = restructuring['cut_count'] // spread
                    if month == spread: monthly_cut += restructuring['cut_count'] % spread
                    if monthly_cut > 0 and len(current) > monthly_cut:
                        targets = self._select_targets(current, restructuring, monthly_cut)
                        if len(targets) > 0:
                            re_exits = len(targets)
                            re_sev = targets['salary_lakhs'].sum() * restructuring.get('severance_months',0) / 100
                            total_severance += re_sev
                            total_restructure += re_exits
                            current = current[~current.index.isin(targets.index)]
                            if restructuring.get('backfill'):
                                backfill_queue[month+backfill_delay] = backfill_queue.get(month+backfill_delay,0) + re_exits
            # Attrition
            current['monthly_attrition_prob'] = current['attrition_probability'] / 12 * attrition_multiplier
            current['leaves_this_month'] = np.random.binomial(1, current['monthly_attrition_prob'])
            leavers = current[current['leaves_this_month']==1].copy()
            current = current[current['leaves_this_month']==0].copy()
            n_backfill = len(leavers)
            if n_backfill > 0 and backfill_delay >= 0:
                backfill_queue[month+backfill_delay] = backfill_queue.get(month+backfill_delay,0) + n_backfill
            # Promotions
            n_promote = 0
            if month % promotion_cycle_months == 0:
                eligible = current[(current['promotion_readiness'].isin(['Ready Now','Ready 1-2Y'])) & (current['performance_rating'].isin(['EE (Exceeds)','GE (Greatly Exceeds)'])) & (current['level']!='L8')]
                n_promote = int(len(eligible) * promotion_rate)
                if n_promote > 0 and len(eligible) > 0:
                    promote_idx = eligible.sample(min(n_promote,len(eligible)), random_state=month).index
                    level_order = ['L3','L4','L5','L6','L7','L8']
                    current.loc[promote_idx,'level'] = current.loc[promote_idx,'level'].apply(lambda x: level_order[min(level_order.index(x)+1,len(level_order)-1)])
                    current.loc[promote_idx,'salary_lakhs'] *= (1 + promo_bump_pct/100)
                    current.loc[promote_idx,'promotion_readiness'] = 'Not Ready'
            # Hiring
            n_hire = planned_hires_per_month + backfill_queue.get(month,0)
            if n_hire > 0:
                hire_levels = np.random.choice(list(hire_dist.keys()), n_hire, p=list(hire_dist.values()))
                depts_pool = current['department'].unique() if len(current)>0 else ['Engineering','Finance','HR','Sales','Product','Operations']
                new_hires = pd.DataFrame({'employee_id':[f'HIRE_{month}_{i}' for i in range(n_hire)],'level':hire_levels,'department':np.random.choice(depts_pool,n_hire),'tenure_months':np.random.randint(1,12,n_hire),'performance_rating':np.random.choice(['ME (Meets)','EE (Exceeds)','LE (Low)'],n_hire,p=[0.6,0.3,0.1]),'salary_lakhs':[{'L3':12,'L4':22,'L5':40,'L6':70}.get(l,12)*np.random.uniform(0.9,1.1) for l in hire_levels],'age':np.random.randint(24,34,n_hire),'gender':np.random.choice(['M','F','NB'],n_hire,p=[0.60,0.38,0.02]),'promotion_readiness':['Not Ready']*n_hire,'attrition_probability':0.15,'attrition_risk_bucket':'Medium','manager_id':-1})
                current = pd.concat([current, new_hires], ignore_index=True)
            current['salary_lakhs'] *= (1 + salary_inflation/12/100)
            monthly_payroll = current['salary_lakhs'].sum() / 100
            hire_cost = n_hire * cost_per_hire_lakhs / 100
            total_cost = monthly_payroll + hire_cost + re_sev
            history.append({'month':month,'headcount':len(current),'monthly_cost_cr':round(total_cost,2),'payroll_cr':round(monthly_payroll,2),'hire_cost_cr':round(hire_cost,2),'severance_cr':round(re_sev,2),'attrition':len(leavers),'hires':n_hire,'promotions':n_promote,'restructure_exits':re_exits})
        return pd.DataFrame(history), current, total_severance, total_restructure

# ========== RULE-BASED INSIGHTS ==========
def generate_insights(baseline, hist, final_df, params, total_sev=0, total_re=0):
    insights = []
    net = hist['headcount'].iloc[-1] - len(baseline)
    if total_re > 0:
        insights.append(('🔧 Restructuring', str(total_re)+' cut, ₹'+str(round(total_sev,2))+'Cr severance. Net: '+str(net)+' heads.', 'Critical'))
    if net > 100: insights.append(('🔥 Aggressive Growth', 'Net +' + str(net) + ' heads. Scale leadership bench.', 'High'))
    elif net < 0: insights.append(('⚠️ Shrinking', 'Net ' + str(net) + ' heads. Review critical roles.', 'Critical'))
    else: insights.append(('📊 Moderate Growth', 'Net +' + str(net) + ' heads. Sustainable.', 'Medium'))
    cg = (hist['monthly_cost_cr'].iloc[-1]/hist['monthly_cost_cr'].iloc[0]-1)*100
    sev = 'High' if cg>30 else 'Medium'
    insights.append(('💰 Cost', 'Monthly cost up ' + str(round(cg,1)) + '%.', sev))
    ta = hist['attrition'].sum(); ar = ta/len(baseline)*100
    sev = 'Critical' if ar>25 else 'High' if ar>15 else 'Medium'
    insights.append(('🚪 Attrition', str(ta)+' exits ('+str(round(ar,1))+'% annualized).', sev))
    rn = len(baseline[baseline['promotion_readiness']=='Ready Now'])
    tp = hist['promotions'].sum(); ratio = rn/max(tp,1)
    if ratio > 10: insights.append(('🎯 Chokepoint', str(rn)+' Ready → '+str(tp)+' promos ('+str(int(ratio))+':1). Flight risk elevated.', 'Critical'))
    else: insights.append(('✅ Promotions', 'Healthy '+str(round(ratio,1))+':1 ratio.', 'Low'))
    dr = baseline.groupby('department').apply(lambda x:(x['attrition_risk_bucket'].isin(['High','Critical'])).mean()*100).sort_values(ascending=False)
    if len(dr)>0 and dr.iloc[0]>60:
        insights.append(('🏢 Dept Risk', str(dr.index[0])+': '+str(int(dr.iloc[0]))+'% High/Critical risk.', 'Critical'))
    l5p = (final_df['level'].isin(['L5','L6','L7','L8'])).mean()*100
    ld = 'Dilution risk' if l5p<25 else 'Healthy'
    sev = 'High' if l5p<25 else 'Medium'
    insights.append(('🏗️ Leadership', 'L5+: '+str(round(l5p,1))+'%. '+ld+'.', sev))
    return insights
