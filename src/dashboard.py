from datetime import datetime
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import pandas as pd
from models_and_validation import get_precision_score_via_backtest
from sklearn.ensemble import RandomForestClassifier
from hopsworks_connections import pull_data, pull_model
from matplotlib import pyplot as plt


st.set_page_config(layout="wide")

current_date = pd.to_datetime(datetime.utcnow(), utc=True).floor('H')
st.title(f'Ethereum Returns Forcasting $$$$')
st.header(f'{current_date} UTC')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 4


##### Get the full data from hopsworks, which already includesd the predicttion for tomorrow
eth_ohlc = pull_data('transformed_ethereum_ohlc', 1, 'view1', 1)
eth_ohlc['date'] = pd.to_datetime(eth_ohlc['date'])
eth_ohlc.set_index('date', inplace=True)

print(eth_ohlc.tail())


##### Plot ethereum close price time series, and color code our predictions #####
end_date = pd.to_datetime(datetime.utcnow(), utc=True).floor('H')
start_date = end_date - pd.Timedelta(weeks=2)
filtered_data = eth_ohlc.loc[start_date:end_date]

# color = 'red' if prediction[0] == 0 else 'green'
# plt.figure(figsize=(10, 5))
# # Plot all data points in grey
# plt.plot(filtered_data.index, filtered_data['close'], marker='o', linestyle='-', color='grey', label='Training Data')
# # Highlight the most recent data point with the predictive color
# last_date = filtered_data.index[-1]
# last_close = filtered_data['close'].iloc[-1]
# plt.plot(last_date, last_close, marker='o', markersize=10, color=color, label='Lower Predicted Close Price Tomorrow' if color == 'red' else 'Higher Predicted Close Price Tomorrow')
# plt.title('Ethereum Close Price Over the Last Two Weeks')
# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.grid(True)
# plt.xticks(rotation=45)
# # Add a legend to the plot
# plt.legend()
# plt.tight_layout()
# plt.show()
# st.pyplot(plt)
