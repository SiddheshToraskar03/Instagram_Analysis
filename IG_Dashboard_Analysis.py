import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents Tkinter errors
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go  # For embedding the heatmap
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load dataset
data = pd.read_csv("instagram_analysis_with_dates.csv", encoding='latin1', nrows=5000)
data = data.dropna()

# Convert Date column (if present)
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])

# Keep only numeric data for correlation heatmap
numeric_data = data.select_dtypes(include=[np.number])

# Dash App
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("📊 Instagram Engagement Dashboard", style={'textAlign': 'center'}),

    # Filters Section
    html.Div([
        dcc.DatePickerRange(
            id='date-picker',
            start_date=data['Date'].min() if 'Date' in data.columns else None,
            end_date=data['Date'].max() if 'Date' in data.columns else None,
            display_format='YYYY-MM-DD',
            style={'marginRight': '20px'}
        ),
        
        dcc.Dropdown(
            id='metric-dropdown',
            options=[{'label': metric, 'value': metric} for metric in ['Likes', 'Comments', 'Shares', 'Saves']],
            value='Likes',
            clearable=False,
            style={'width': '200px', 'marginRight': '20px'}
        ),

        dcc.Dropdown(
            id='hashtag-filter',
            options=[{'label': 'With Hashtags', 'value': 'Hashtags'}, {'label': 'No Hashtags', 'value': 'No Hashtags'}],
            value='Hashtags',
            clearable=False,
            style={'width': '200px'}
        )
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),

    # Tabs Section
    dcc.Tabs([
        dcc.Tab(label='📈 Overview', children=[
            dcc.Graph(id='scatter-plot'),
            dcc.Graph(id='engagement-bar-chart')
        ]),

        dcc.Tab(label='📊 Advanced Analysis', children=[
            dcc.Graph(id='engagement-time-series'),
            dcc.Graph(id='hashtag-impact-bar'),
            dcc.Graph(id='top-engaging-posts'),
            dcc.Graph(id='engagement-distribution')
        ]),

        # New Tab for Correlation Matrix
        dcc.Tab(label='📊 Correlation Matrix', children=[
            dcc.Graph(id='correlation-matrix')
        ])
    ])
])

# Apply Filters
def filter_data(selected_metric, start_date, end_date, hashtag_filter):
    df = data.copy()
    
    if 'Date' in df.columns:
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    if 'From Hashtags' in df.columns:
        if hashtag_filter == 'Hashtags':
            df = df[df['From Hashtags'] > 0]
        elif hashtag_filter == 'No Hashtags':
            df = df[df['From Hashtags'] == 0]

    # Avoid ValueError: Take only available rows if dataset is smaller than 200
    sample_size = min(200, len(df))
    return df.sample(sample_size, replace=False) if sample_size > 0 else df

@app.callback(
    Output('scatter-plot', 'figure'),
    Output('engagement-bar-chart', 'figure'),
    Output('engagement-time-series', 'figure'),
    Output('hashtag-impact-bar', 'figure'),
    Output('top-engaging-posts', 'figure'),
    Output('engagement-distribution', 'figure'),
    Output('correlation-matrix', 'figure'),
    Input('metric-dropdown', 'value'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('hashtag-filter', 'value')
)
def update_dashboard(selected_metric, start_date, end_date, hashtag_filter):
    df_filtered = filter_data(selected_metric, start_date, end_date, hashtag_filter)

    if df_filtered.empty:
        return px.scatter(title="No Data Available"), px.bar(title="No Data Available"), \
               px.line(title="No Data Available"), px.bar(title="No Data Available"), \
               px.bar(title="No Data Available"), px.histogram(title="No Data Available"), \
               go.Figure()

    # Scatter Plot (Engagement vs Impressions)
    scatter_fig = px.scatter(
        df_filtered, x='Impressions', y=selected_metric, size=selected_metric,
        title=f"{selected_metric} vs Impressions"
    )
    
    # Engagement Bar Chart (Caption vs Engagement)
    bar_fig = px.bar(
        df_filtered, x='Caption', y=selected_metric,
        title=f"{selected_metric} by Post Caption"
    )
    
    # Engagement Over Time
    if 'Date' in data.columns:
        df_time = df_filtered.groupby('Date')[selected_metric].sum().reset_index()
        time_series_fig = px.line(df_time, x='Date', y=selected_metric, title=f"{selected_metric} Over Time")
    else:
        time_series_fig = px.line(title="No Date Column Found")
    
    # Hashtag Effectiveness
    if 'From Hashtags' in data.columns:
        hashtag_fig = px.bar(
            df_filtered, x='From Hashtags', y=selected_metric,
            title=f"Effect of Hashtags on {selected_metric}"
        )
    else:
        hashtag_fig = px.bar(title="No Hashtag Data Available")

    # Top Engaging Posts
    top_posts_fig = px.bar(
        df_filtered.sort_values(by=selected_metric, ascending=False).head(10),
        x='Caption', y=selected_metric, title=f"Top 10 Posts by {selected_metric}"
    )

    # Engagement Distribution
    engagement_hist_fig = px.histogram(
        df_filtered, x=selected_metric, title=f"{selected_metric} Distribution"
    )

    # Correlation Matrix Heatmap
    corr_matrix = numeric_data.corr()
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='YlGnBu',
        colorbar=dict(title='Correlation')
    ))
    heatmap_fig.update_layout(title="Correlation Matrix Heatmap", xaxis_title="Features", yaxis_title="Features")

    return scatter_fig, bar_fig, time_series_fig, hashtag_fig, top_posts_fig, engagement_hist_fig, heatmap_fig

if __name__ == '__main__':
    app.run_server(debug=True)
