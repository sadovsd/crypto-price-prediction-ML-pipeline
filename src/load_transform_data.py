from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
try:
    # for when its called as a module from a jupyter notebook
    from .paths import RAW_DATA_DIR, HISTORICAL_OHLC
except ImportError:
    # for when its run on its own
    from paths import RAW_DATA_DIR, HISTORICAL_OHLC
import time
import pandas as pd
import pandas_ta as ta

# RAW_DATA_DIR = '../data/raw'
# HISTORICAL_OHLC = '../data/raw/historical_ohlc'

def download_historical_ohlc_from_coinlore():
    chrome_options = Options()
    prefs = {"download.default_directory": str(RAW_DATA_DIR)}
    chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get('https://www.coinlore.com/coin/ethereum/historical-data')

        # Wait for the download button to be clickable and click it
        WebDriverWait(driver, 4).until(
            EC.element_to_be_clickable((By.XPATH, '//button[@onclick="tableToCSV()"]'))
        ).click()

        time.sleep(3)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()


def transform_ohlc_to_features_target():

    ##### Load the raw OHLC data and clean out the $ signs and shit #####
    df = pd.read_csv(HISTORICAL_OHLC / 'ethereum.csv', parse_dates=True)
    def convert_value(value):
        """
        Converts a string value to a float. Removes $ signs, and converts
        billion (bn), million (m), and thousand (K) values to their numeric equivalents.
        """
        value = value.replace('$', '')  # Remove $ sign to simplify processing
        if value[-1].lower() == 'm':
            return float(value[:-1]) * 1_000_000
        elif value[-1].lower() == 'b':
            return float(value[:-1]) * 1_000_000_000
        elif value[-1].lower() == 'k':
            return float(value[:-1]) * 1_000
        elif value[-2:].lower() == 'bn':  # Handle 'bn' for billions
            return float(value[:-2]) * 1_000_000_000
        else:
            return float(value)
    columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
    for column in columns_to_convert:
        df[column] = df[column].apply(lambda x: convert_value(x) if isinstance(x, str) else x)
    df['Volume(ETH)'] = pd.to_numeric(df['Volume(ETH)'].apply(lambda x: convert_value(x) if isinstance(x, str) else x), errors='coerce')
    df = df.iloc[::-1]
    df["tommorow_price"] = df["Close"].shift(-1)
    df["target"] = (df["tommorow_price"] > df["Close"]).astype(int)

    ##### add new columns that will be part of design matrix #####
    horizons = [2,5,10,25,50,100] 
    predictors = []
    for horizon in horizons:
        # Exponential Moving Average (EMA)
        ema_col = f"ema_{horizon}"
        df[ema_col] = df["Close"] / ta.ema(df["Close"], length=horizon)
        predictors.append(ema_col)
        # Relative Strength Index (RSI)
        rsi_col = f"rsi_{horizon}"
        df[rsi_col] = ta.rsi(df["Close"], length=horizon)
        predictors.append(rsi_col)
        # Simple Moving Average (SMA)
        sma_col = f"sma_{horizon}"
        df[sma_col] = df["Close"] / ta.sma(df["Close"], length=horizon)
        predictors.append(sma_col)
    df = df.dropna()

    # Hopsworks feature store only accepts lowercase feature names
    df.rename(columns={'Market Cap' : 'market_cap'}, inplace=True)
    df.rename(columns={'Volume(ETH)' : 'volume_eth'}, inplace=True)
    df.rename(columns={col: col.lower() for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']}, inplace=True)

    # Hopsworks needs event_time feature to be datetime type, not string
    df['date'] = pd.to_datetime(df['date'])
    
    return df
