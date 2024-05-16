from datetime import datetime
import streamlit as st
import pandas as pd
import pandas as pd
from hopsworks_connections import pull_data, pull_model
import pytz
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_score
import matplotlib.patches as mpatches
import requests


### streamlit run src/dashboard.py

st.set_page_config(layout="wide")

##### Make sidebar
progress_bar = st.sidebar.header('Project Explanation and Video ---> https://www.davydsadovskyy.com/#/projects')
progress_bar = st.sidebar.header('⚙️ Hopsworks Feature Store Data Retrieval')

progress_bar = st.sidebar.progress(0)
N_STEPS = 2

# Get the current time in EST
current_date_utc = pd.to_datetime(datetime.utcnow(), utc=True).floor('T')
est = pytz.timezone('America/New_York')
current_date_est = current_date_utc.tz_convert(est)
formatted_date = current_date_est.strftime('%m-%d-%Y, %I:%M %p')

# Get current ETH price
response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd')
data = response.json()
current_eth_price = data['ethereum']['usd']

##### Display the title and the date header
# st.subheader(f'{formatted_date} EST')
# st.subheader(f'Current ETH Price: {current_eth_price}')
col1, col2 = st.columns(2)
with col1:
    st.subheader(f'{formatted_date} EST')
with col2:
    st.subheader(f'Current ETH Price: {current_eth_price}')
st.title('Ethereum Returns Forecasting')


@st.cache_data
def get_prediction_data():
    predictions_df = pull_data('eth_ohlc_predictions', 2, 'eth_ohlc_predictions_view', 2)
    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date
    return predictions_df

@st.cache_data
def get_model_specs():
    model_specs = pull_model('catboost_eth_returns', 1)[1]
    timestamp = model_specs.created
    date_time = datetime.fromtimestamp(timestamp / 1000) - timedelta(days=1)
    formatted_date = date_time.strftime('%m/%d/%Y')
    return formatted_date

predictions_df = get_prediction_data()
with st.spinner(text="Fetching model predictions from the store"):
    st.sidebar.write('✅ Model predictions arrived')
    progress_bar.progress(1/N_STEPS)

model_specs = get_model_specs()
with st.spinner(text="Getting Model Information"):
    st.sidebar.write('✅ Model meta data arrived')
    progress_bar.progress(2/N_STEPS)

##### write predictions
# last_date = predictions_df['date'].iloc[-1].strftime('%m/%d/%Y')
last_date = (pd.to_datetime(predictions_df['date'].iloc[-1]) + pd.Timedelta(days=1)).strftime('%m/%d/%Y')
# st.markdown(f"### CatBoost Model Predictions for {last_date}: {predictions_df['predicted_probability'].iloc[-1]} Probability of 1.0% Returns")
st.markdown(
    f"### CatBoost Model Predictions for {last_date}: <span style='color:green;'>{predictions_df['predicted_probability'].iloc[-1]*100:.2f}% Chance of 1.0% Returns</span>",
    unsafe_allow_html=True
)
st.write("")
st.write("")
st.write("")
st.write("")


def make_price_and_probability_plot(data, last_n_days):

    df = data.copy()

    df['avg_high_close'] = (df['close']+df['high']) / 2
    last_n_day_prediction_df = df.tail(last_n_days)

    colors = np.where(last_n_day_prediction_df['tmw_percent_increase_to_avg_high_low'] > 0.01, 'green', 'red')
    colors[-1] = 'grey'  # Set the last element to blue

    fig, ax = plt.subplots(figsize=(10, 5))

    line1, = plt.plot(last_n_day_prediction_df['date'], last_n_day_prediction_df['close'], label='Close', marker='o', linestyle='-')
    line2, = plt.plot(last_n_day_prediction_df['date'], last_n_day_prediction_df['avg_high_close'], label='(High + Close) / 2', marker='o', linestyle='--')
    
    # Overlay points with specific colors
    for i in range(len(last_n_day_prediction_df)):
        plt.scatter(last_n_day_prediction_df['date'].iloc[i], last_n_day_prediction_df['close'].iloc[i], color=colors[i], s=40, zorder=3)

    # Custom legend entries
    green_patch = mpatches.Patch(color='green', label='More than 1.0% Returns Next Day')
    red_patch = mpatches.Patch(color='red', label='Less than 1.0% Returns Next Day')
    grey_patch = mpatches.Patch(color='grey', label='Next Day Value not yet Known')

    # reference values for extending the y axis
    difference = last_n_day_prediction_df['avg_high_close'].max() - last_n_day_prediction_df['close'].min()
    y_max = last_n_day_prediction_df['avg_high_close'].max() + 0.3 * difference  # Extend 30% higher
    y_min = last_n_day_prediction_df['close'].min() - 0.1 * difference  # Extend 10% lower

    # plt.title(f"Ethereum Prices and Prediction for 1.0% Return Between Close and Next Day's (High+Close)/2")
    plt.title(f"Last {last_n_days} Days of Ethereum Prices and Prediction for 1.0% Return Between\nClose and Next Day's (High+Close)/2", fontsize=16, loc='left')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend(handles=[line1, line2, green_patch, red_patch, grey_patch], loc='upper left')
    plt.grid(True)
    ax.set_ylim(y_min, y_max)  # Set the custom y-axis limits
    ax.grid(True)
    ax2 = ax.twinx()
    ax2.bar(last_n_day_prediction_df['date'], last_n_day_prediction_df['predicted_probability'], color='grey', alpha=0.3, width=.5)
    ax2.set_ylabel('Probability of 1.0% return')
    ax2.set_ylim(0, 1)  # Set the custom y-axis limits
    plt.tight_layout()
    plt.show()

    return fig

fig1 = make_price_and_probability_plot(predictions_df, 30)
st.pyplot(fig1)


def make_precision_score_plot(data, date_range_str):

    df = data.copy()
    # Assume thresholds and predictions_df_no_NA are predefined
    thresholds = np.arange(0.35, 0.75, 0.005)
    tmw_0_0_percent_increase_binary = (df['tmw_percent_increase_to_avg_high_low'] >= 0).astype(int)
    tmw_1_0_percent_increase_binary = df['tmw_1_0_percent_increase_binary']

    precision_scores_1_0 = []
    precision_scores_0_0 = []
    count_1s_1_0 = []
    count_1s_0_0 = []
    count_0s_1_0 = []
    count_0s_0_0 = []

    proportion_1_0 = tmw_1_0_percent_increase_binary.value_counts(normalize=True).get(1, 0)
    proportion_0_0 = tmw_0_0_percent_increase_binary.value_counts(normalize=True).get(1, 0)

    count_1_0 = tmw_1_0_percent_increase_binary.value_counts().get(1, 0)
    count_0_0 = tmw_0_0_percent_increase_binary.value_counts().get(1, 0)

    for threshold in thresholds:
        pred_at_threshold = (df['predicted_probability'] >= threshold).astype(int)
        precision_scores_1_0.append(precision_score(tmw_1_0_percent_increase_binary, pred_at_threshold, zero_division=0))
        precision_scores_0_0.append(precision_score(tmw_0_0_percent_increase_binary, pred_at_threshold, zero_division=0))
        count_1s_1_0.append(pred_at_threshold.sum())
        count_0s_1_0.append(len(pred_at_threshold) - pred_at_threshold.sum())
        count_1s_0_0.append(pred_at_threshold.sum())
        count_0s_0_0.append(len(pred_at_threshold) - pred_at_threshold.sum())

    fig = go.Figure()

    # Add traces for precision scores
    fig.add_trace(go.Scatter(
        x=thresholds, y=precision_scores_1_0, mode='lines+markers',
        name='CatBoost > 1.0% Predicted Returns',
        text=[f"1.0% Return<br>Threshold = {threshold.round(3)}<br>Precision Score = {score:.4f}<br>1's predicted: {c1}<br>0's predicted: {c0}"
            for score, threshold, c1, c0 in zip(precision_scores_1_0, thresholds, count_1s_1_0, count_0s_1_0)],
        hoverinfo='text',
        marker=dict(color='blue'),
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=thresholds, y=precision_scores_0_0, mode='lines+markers',
        name='Positive Percent Returns',
        text=[f"0.0% Return<br>Threshold = {threshold.round(3)}<br>Precision Score = {score:.4f}<br>1's predicted: {c1}<br>0's predicted: {c0}"
            for score, threshold, c1, c0 in zip(precision_scores_0_0, thresholds, count_1s_0_0, count_0s_0_0)],
        hoverinfo='text',
        marker=dict(color='rgb(204,173,0)'),
        line=dict(color='rgb(204,173,0)')
    ))

    fig.add_trace(go.Scatter(
        x=thresholds, y=[proportion_1_0] * len(thresholds),
        mode='lines', name='Proportion of 1.0% Return Days',
        line=dict(dash='dash', color='blue'),
        hoverinfo='text',
        text=[f"Proportion of 1.0% Return Days: {proportion_1_0:.4f}<br>Number of 1.0% return days: {count_1_0}"] * len(thresholds)
    ))

    fig.add_trace(go.Scatter(
        x=thresholds, y=[proportion_0_0] * len(thresholds),
        mode='lines', name='Proportion of 0.0% Return Days',
        line=dict(dash='dash', color='rgb(204,173,0)'),
        hoverinfo='text',
        text=[f"Proportion of 0.0% Return Days: {proportion_0_0:.4f}<br>Number of 0.0% Return Days: {count_0_0}"] * len(thresholds)
    ))

    fig.update_layout(
        # title='Precision Scores by Predicted Model Probability Threshold<br>Precision - Of all Days Predicted to have 1.0% Returns Next Day, How Many Actually Did?',
        title=f'<span style="font-size: 23px;">{date_range_str} Precision Scores by Predicted Model Probability Threshold</span><br><span style="font-size: 16px;">Of all Days Predicted to have 1.0% Returns at a Threshold, How Many Actually Did?</span>',
        xaxis_title='Threshold',
        yaxis_title='Precision Score',
        legend_title='Return Type'
    )

    return fig

def plot_and_select_date_range(data):
    df = data.copy()

    # Adjust the marker size and use container width
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['close'], mode='lines+markers',
        hoverinfo='text',
        marker=dict(size=4),  # Smaller marker size
        text=[f"Date: {date}<br>Close Price: {price}" for date, price in zip(df['date'], df['close'])]
    ))
    fig.update_layout(
        title='<span style="font-size: 23px;">Ethereum Price Over Days Where Backtested Predictions Are Available</span>',
        xaxis_title='Date',
        yaxis_title='Close Price',
        xaxis_tickangle=-45,
        hovermode='closest'
    )

    # Match the graph width to the slider by setting use_container_width to True
    st.plotly_chart(fig, use_container_width=True)

    min_date, max_date = df['date'].min(), df['date'].max()
    date_range = st.slider(
        "Select Date Range",
        value=(min_date, max_date),
        format="MM/DD/YY",
        key='date_slider'
    )

    filtered_data = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
    date_range_str = f"{date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}"

    fig = make_precision_score_plot(filtered_data, date_range_str)
    st.plotly_chart(fig, use_container_width=True)

def plot_and_select_date_range(data):
    df = data.copy()

    # Adjust the marker size and use container width
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['close'], mode='lines+markers',
        hoverinfo='text',
        marker=dict(size=4),  # Smaller marker size
        text=[f"Date: {date}<br>Close Price: {price}" for date, price in zip(df['date'], df['close'])]
    ))
    fig.update_layout(
        title='<span style="font-size: 23px;">Ethereum Price Over Days Where Backtested Predictions Are Available</span>',
        xaxis_title='Date',
        yaxis_title='Close Price',
        xaxis_tickangle=-45,
        hovermode='closest'
    )

    # Match the graph width to the input fields by setting use_container_width to True
    st.plotly_chart(fig, use_container_width=True)

    min_date, max_date = df['date'].min(), df['date'].max()

    # Input fields for start and end dates
    start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    # Ensure start_date is before end_date
    if start_date > end_date:
        st.error("Error: End Date must fall after Start Date.")
        return

    filtered_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

    fig = make_precision_score_plot(filtered_data, date_range_str)
    st.plotly_chart(fig, use_container_width=True)

st.title('Historical Model Performance')
st.markdown(
    f"#### This CatBoost Model Was Trained on Data from 2015-11-15 to {model_specs}<br>Backtested Predictions are Available From 2018-08-11 to {predictions_df[:-1]['date'].max()}",
    unsafe_allow_html=True
)
plot_and_select_date_range(predictions_df[:-1])

