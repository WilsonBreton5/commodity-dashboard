from dash import Dash, dcc, html, Input, Output
import numpy as np
import pandas as pd
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import os

# ------------------------------------------------------------
# 1. Create the Dash app and expose server for Render
# ------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Breton Wilson-Commodity Risk Dashboard"
server = app.server
import logging
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------
# 2. Commodity tickers (Yahoo Finance symbols)
# ------------------------------------------------------------
COMMODITIES = {

    "Copper": "HG=F",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Bitcoin": "BTC-USD"
}

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------------------------------------------------
# 3. Layout
# ------------------------------------------------------------
app.layout = html.Div([
    html.H2("Commodity Risk Metrics Dashboard", style={"textAlign": "center"}),

    dcc.Dropdown(
        id="commodity-dropdown",
        options=[{"label": name, "value": ticker} for name, ticker in COMMODITIES.items()],
        value="HG=F",
        clearable=False,
        style={
            "width": "300px",
            "margin": "auto",
            "color": "black", 
            }
    ),

    html.Div(dbc.Spinner(size="lg", color="black", type="border"), id="commodity-graph"),
])

# ------------------------------------------------------------
# 4. Helper: fetch and cache data
# ------------------------------------------------------------
def fetch_commodity_data(ticker):
    cache_file = os.path.join(CACHE_DIR, f"{ticker}.csv")
    try:
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, parse_dates=["Date"])
            logging.info(f"Loaded {ticker} data from cache.")
        else:
            df = yf.download(ticker, period="max", interval="1d")
            if df.empty:
                raise ValueError("Yahoo returned empty data")
            
            # Flatten columns if multi-index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join(col).strip() for col in df.columns.values]
            
            df = df.reset_index()
            df.to_csv(cache_file, index=False)
            logging.info(f"Downloaded {ticker} data and cached.")

        # Choose 'Close' column safely
        close_col = None
        for col in df.columns:
            if 'Close' in col:
                close_col = col
                break
        if close_col is None:
            raise ValueError(f"No 'Close' column found in {ticker} data")

        df = df[['Date', close_col]].rename(columns={close_col: 'Value'})

        # Ensure numeric
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['Value'])
        df = df[df['Value'] > 0].reset_index(drop=True)

        return df

    except Exception as e:
        logging.error(f"Error fetching {ticker}: {e}")
        return pd.DataFrame(columns=["Date", "Value"])



# ------------------------------------------------------------
# 5. Callback to generate figure
# ------------------------------------------------------------
@app.callback(
    Output("commodity-graph", "children"),
    Input("commodity-dropdown", "value")
)
def calc_commodity_risk(ticker):
    # Fetch cached data
    df = fetch_commodity_data(ticker)
    
    if df.empty:
        return html.Div(f"No data available for {ticker}", style={"color": "red", "textAlign": "center"})
    
    # Ensure 'Value' is numeric and positive
    df = df[df['Value'] > 0].sort_values('Date').reset_index(drop=True)
    if df.empty:
        return html.Div(f"No positive price data for {ticker}", style={"color": "red", "textAlign": "center"})
    
    # Compute moving average safely
    window = min(200, len(df))
    if len(df) >= 1:
        df['MA'] = df['Value'].rolling(window, min_periods=1).mean()
    else:
        return html.Div(f"Not enough data to compute moving average for {ticker}", style={"color": "red", "textAlign": "center"})
    
    # Drop NaNs safely
    if 'MA' in df.columns:
        df = df.dropna(subset=['MA']).reset_index(drop=True)
    else:
        return html.Div(f"MA column not computed for {ticker}", style={"color": "red", "textAlign": "center"})
    
    if df.empty:
        return html.Div(f"Not enough data to compute metrics for {ticker}", style={"color": "red", "textAlign": "center"})
    
    # Compute risk metric
    df['Preavg'] = (np.log(df['Value']) - np.log(df['MA'])) * (np.arange(len(df)) ** 0.395)
    df['avg'] = (df['Preavg'] - df['Preavg'].cummin()) / (df['Preavg'].cummax() - df['Preavg'].cummin())
    
    # Drop early noise if data is long enough
    df = df[df.index > 100] if len(df) > 100 else df
    
    # Create figure
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Value'], name='Price', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['avg'], name='Risk', line=dict(color='white')), secondary_y=True)
    
    # Green/red zones
    opacity = 0.2
    for i in range(5, 0, -1):
        opacity += 0.05
        fig.add_hrect(y0=i*0.1, y1=((i-1)*0.1), line_width=0, fillcolor='green', opacity=opacity, secondary_y=True)
    opacity = 0.2
    for i in range(6, 10):
        opacity += 0.1
        fig.add_hrect(y0=i*0.1, y1=((i+1)*0.1), line_width=0, fillcolor='red', opacity=opacity, secondary_y=True)
    
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Price ($USD)", type="log", showgrid=False)
    fig.update_yaxes(title="Risk", type="linear", secondary_y=True, tick0=0.0, dtick=0.1, range=[0,1])
    
    current_risk = df['avg'].iloc[-1] if not df['avg'].empty else float('nan')
    fig.update_layout(template="plotly_dark", title={"text": f"{ticker} – Current Risk: {current_risk:.2f}"})
    
    return dcc.Graph(figure=fig, responsive=True, style={'height': '90vh'})





# ------------------------------------------------------------
# 6. Run locally
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
