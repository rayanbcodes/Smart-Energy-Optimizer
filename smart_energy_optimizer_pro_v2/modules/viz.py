# modules/viz.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_price_curve(prices_df):
    fig = px.line(prices_df, x='hour', y='price_per_kwh', title='TOU Price Curve', markers=True)
    fig.update_xaxes(dtick=1)
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    return fig

def plot_hourly_comparison(baseline_profile, optimized_profile, title="Hourly kWh Comparison"):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=baseline_profile['hour'], y=baseline_profile['baseline_kwh'], name='Baseline kWh'))
    fig.add_trace(go.Bar(x=baseline_profile['hour'], y=baseline_profile['flexible_kwh'], name='Flexible kWh (Baseline)'))
    fig.add_trace(go.Bar(x=optimized_profile['hour'], y=optimized_profile['flexible_kwh'], name='Flexible kWh (Optimized)', marker=dict(opacity=0.7)))
    fig.update_layout(barmode='group', title=title)
    return fig
