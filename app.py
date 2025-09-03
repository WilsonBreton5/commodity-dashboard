from dash import Dash, dcc, html, Input, Output, callback
import numpy as np
import pandas as pd
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

# ------------------------------------------------------------
# 1. Create the Dash app and expose server for Render
# ------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server  # 👈 required for Render
import logging
logging.basicConfig(level=logging.DEBUG)

# ------------------------------------------------------------
# 2. Commodity tickers (Yahoo Finance symbols)
# ------------------------------------------------------------
COMMODITIES = {
    "Copper": "HG=F",
    "Aluminum": "ALI=F",
    "Nickel": "NID=F"
}

# ------------------------------------------------------------
# 3. Layout
# ------------------------------------------------------------
app.layout = html.Div([
    html.H2("Commodity Risk Metrics Dashboard", style={"textAlign": "center"}),

    dcc.Dropdown(
        id="commodity-dropdown",
        options=[{"label": name, "value": ticker} for name, ticker in COMMODITIES.items()],
        value="HG=F",  # Default to Copper
        clearable=False,
        style={"width": "300px", "margin": "auto"}
    ),

    html.Div(dbc.Spinner(size="lg", color="black", type="border"), id="commodity-graph"),
])

# ------------------------------------------------------------
# 4. Callback to generate figure
# ------------------------------------------------------------
@app.callback(
    Output("commodity-graph", "children"),
    Input("commodity-dropdown", "value")
)
def calc_commodity_risk(ticker):
    # Fetch ~5 years of daily data
    df = yf.download(ticker, period="5y", interval="1d").reset_index()

    # Prepare data
    df = df[["Date", "Close"]].rename(columns={"Close": "Value"})
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    df = df[df['Value'] > 0]
    df.index = pd.to_numeric(df.index, errors="coerce")

    # Risk metric calculation
    df['MA'] = df['Value'].rolling(200, min_periods=1).mean().dropna()
    df['Preavg'] = (np.log(df['Value']) - np.log(df['MA'])) * df.index**0.395

    # Normalize 0-1
    df['avg'] = (df['Preavg'] - df['Preavg'].cummin()) / \
                (df['Preavg'].cummax() - df['Preavg'].cummin())

    df = df[df.index > 100]  # drop early noise

    # Create plot
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Value'],
                             name='Price', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['avg'],
                             name='Risk', line=dict(color='white')),
                             secondary_y=True)

    # Green (buy zones)
    opacity = 0.2
    for i in range(5, 0, -1):
        opacity += 0.05
        fig.add_hrect(y0=i*0.1, y1=((i-1)*0.1), line_width=0,
                      fillcolor='green', opacity=opacity, secondary_y=True)

    # Red (sell zones)
    opacity = 0.2
    for i in range(6, 10):
        opacity += 0.1
        fig.add_hrect(y0=i*0.1, y1=((i+1)*0.1), line_width=0,
                      fillcolor='red', opacity=opacity, secondary_y=True)

    # Layout settings
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Price ($USD)", type="log", showgrid=False)
    fig.update_yaxes(title="Risk", type="linear", secondary_y=True,
                     showgrid=True, tick0=0.0, dtick=0.1, range=[0, 1])
    fig.update_layout(template="plotly_dark",
                      title={"text": f"{ticker} – Current Risk: {df['avg'].iloc[-1]:.2f}"})

    return dcc.Graph(figure=fig, responsive=True, style={'height': '90vh'})

# ------------------------------------------------------------
# 5. Run locally
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
