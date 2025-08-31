# modules/optimizer.py
from typing import Dict, List
import pulp
from modules.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

def optimize_schedule_pulp(appliances_df: pd.DataFrame, baseline_df: pd.DataFrame, prices_df: pd.DataFrame, max_simultaneous: int = None) -> Dict[str, List[int]]:
    hours = list(range(24))
    price_map = dict(zip(prices_df['hour'], prices_df['price_per_kwh']))
    baseline_map = dict(zip(baseline_df['hour'], baseline_df['kwh']))

    flex = appliances_df[appliances_df['flexible'] == 1].copy()
    problem = pulp.LpProblem("TOU_opt", pulp.LpMinimize)
    x = {}

    for _, row in flex.iterrows():
        name = row['name']
        estart = int(row['earliest_start'])
        lend = int(row['latest_end'])
        dur = int(row['duration_hours'])
        # Handle wrap-around windows (like 22 to 6)
        window = [t for t in hours if (estart <= lend and estart <= t < lend) or (estart > lend and (t >= estart or t < lend))]
        for t in window:
            x[(name, t)] = pulp.LpVariable(f"x_{name}_{t}", cat='Binary')

    # Objective
    cost_terms = []
    for t in hours:
        cost_terms.append(price_map.get(t, 0.0) * baseline_map.get(t, 0.0))
        for _, row in flex.iterrows():
            name = row['name']
            power = float(row['power_kw'])
            if (name, t) in x:
                cost_terms.append(price_map.get(t, 0.0) * power * x[(name, t)])
    problem += pulp.lpSum(cost_terms)

    # Duration constraints
    for _, row in flex.iterrows():
        name = row['name']
        dur = int(row['duration_hours'])
        vars_for_name = [x[(name, t)] for t in hours if (name, t) in x]
        if vars_for_name:
            problem += pulp.lpSum(vars_for_name) == dur

    if max_simultaneous:
        # prevent more than max_simultaneous devices ON at same hour, weighted by power>threshold
        for t in hours:
            vars_at_t = [x[(name,t)] for (name,t) in x if t==t]
            if vars_at_t:
                problem += pulp.lpSum(vars_at_t) <= max_simultaneous

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    schedule = {}
    for _, row in flex.iterrows():
        name = row['name']
        on_hours = [t for t in hours if (name, t) in x and pulp.value(x[(name, t)]) > 0.5]
        schedule[name] = sorted(on_hours)
    logger.info(f"Optimized schedule (pulp): {schedule}")
    return schedule

def schedule_to_profile(baseline_df, appliances_df, schedule):
    hours = list(range(24))
    base_map = dict(zip(baseline_df['hour'], baseline_df['kwh']))
    flex_profile = [0.0]*24
    for _, row in appliances_df.iterrows():
        name = row['name']
        power = float(row['power_kw'])
        if row['flexible'] == 1:
            hours_on = schedule.get(name, [])
            for h in hours_on:
                flex_profile[h] += power
        else:
            # distribute proportionally across the day
            for h in hours:
                flex_profile[h] += power * (row['duration_hours']/24.0)
    out = pd.DataFrame({'hour': hours, 'baseline_kwh':[base_map.get(h,0.0) for h in hours], 'flexible_kwh': flex_profile})
    out['total_kwh'] = out['baseline_kwh'] + out['flexible_kwh']
    return out
