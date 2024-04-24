from datetime import datetime
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import pandas as pd
from hopsworks_connections import pull_data
from matplotlib import pyplot as plt
import pytz


### streamlit run src/dashboard.py


st.set_page_config(layout="wide")


# Convert UTC to EST
current_date_utc = pd.to_datetime(datetime.utcnow(), utc=True).floor('T')
est = pytz.timezone('America/New_York')
current_date_est = current_date_utc.tz_convert(est)
# Format the date string as MM-DD-YYYY, HH:MM AM/PM
formatted_date = current_date_est.strftime('%m-%d-%Y, %I:%M %p')

# Display the title and the date header
st.title('Ethereum Returns Forecasting')
st.header(f'{formatted_date} EST')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 1

with st.spinner(text="Fetching model predictions from the store"):
    predictions_df = pull_data('eth_ohlc_predictions', 1, 'eth_ohlc_predictions_view', 1)
    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date
    st.sidebar.write('✅ Model predictions arrived')
    progress_bar.progress(1/N_STEPS)


##### write predictions
# last_date = predictions_df['date'].iloc[-1].strftime('%m/%d/%Y')
last_date = (pd.to_datetime(predictions_df['date'].iloc[-1]) + pd.Timedelta(days=1)).strftime('%m/%d/%Y')
st.markdown(f"### Random Forest Model Predictions for {last_date}:")
st.markdown(f"#### 1.0% returns - {predictions_df['pred_tmw_1_0_percent_increase_binary'].iloc[-1]}", unsafe_allow_html=True)

##### Plot ethereum close price time series, and color code our predictions #####
# Create a shifted column for 'tmw_avg_high_close' to represent the previous day's values
predictions_df['prev_tmw_avg_high_close'] = predictions_df['tmw_avg_high_close'].shift(1)
# Filter the DataFrame for the last two weeks
last_two_weeks_data = predictions_df.tail(14)

fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(last_two_weeks_data['date'], last_two_weeks_data['close'], label='Close', marker='o', linestyle='-')
plt.plot(last_two_weeks_data['date'], last_two_weeks_data['prev_tmw_avg_high_close'], label='(High + Close) / 2', marker='o', linestyle='--')
plt.title('Ethereum Prices Over Past 2 Weeks')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
st.pyplot(fig)