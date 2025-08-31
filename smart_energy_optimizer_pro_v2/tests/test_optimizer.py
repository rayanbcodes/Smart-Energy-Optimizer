# tests/test_optimizer.py
import pandas as pd
from modules.optimizer import optimize_schedule_pulp, schedule_to_profile

def test_optimizer_basic():
    appliances = pd.DataFrame([
        {'name':'dishwasher','power_kw':1.2,'duration_hours':1,'flexible':1,'earliest_start':20,'latest_end':24},
        {'name':'washer','power_kw':0.6,'duration_hours':1,'flexible':1,'earliest_start':6,'latest_end':10}
    ])
    baseline = pd.DataFrame({'hour':list(range(24)),'kwh':[0.5]*24})
    prices = pd.DataFrame({'hour':list(range(24)),'price_per_kwh':[0.2]*24})
    sched = optimize_schedule_pulp(appliances, baseline, prices)
    profile = schedule_to_profile(baseline, appliances, sched)
    assert 'total_kwh' in profile.columns
    assert profile['total_kwh'].sum() > 0
