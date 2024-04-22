import hopsworks
from dotenv import load_dotenv
import os
from pathlib import Path
import joblib
from datetime import datetime, timedelta


load_dotenv('../.env')
HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
PROJECT_NAME = 'ethereum_returns'

# def pull_data2():
#     connection = hsfs.connection()
#     fs = connection.get_feature_store(name='ethereum_returns_featurestore')
#     fg = fs.get_feature_group('ethereum_ohlc_feature_group', version=1)

def pull_data(feature_group_name, feature_group_version, feature_view_name, feature_view_version):
    
    project = hopsworks.login(
        project=PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY
    )

    feature_store = project.get_feature_store()

    feature_group = feature_store.get_or_create_feature_group(
        name=feature_group_name,
        version=feature_group_version,
        description="Historical daily ethereum OHLC time series",
        primary_key = ['date'],
        event_time='date'
    )

    # create feature view if it doesn't exist yet
    try:
        feature_store.create_feature_view(
            name=feature_view_name,
            version=feature_view_version,
            query=feature_group.select_all()
        )
    except:
        print('Feature view already existed. Skip creation.')
    
    feature_view = feature_store.get_feature_view(
        name=feature_view_name,
        version=feature_view_version
    )

    # even when the second object is defined, it is still noneType... In hopsworks docs tho,
    # they define like this: feature_df, label_df
    data, _ = feature_view.training_data(
        description='Transformed Ethereum OHLC'
    )
    data = data.sort_values(by='date', ascending=True)

    # data = feature_view.create_training_data()

    return data

def pull_last_row(feature_group_name, feature_group_version, feature_view_name, feature_view_version):
    project = hopsworks.login(
        project=PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY
    )

    feature_store = project.get_feature_store()

    feature_group = feature_store.get_or_create_feature_group(
        name=feature_group_name,
        version=feature_group_version,
        description="Historical daily ethereum OHLC time series",
        primary_key = ['date'],
        event_time='date'
    )

    # create feature view if it doesn't exist yet
    try:
        feature_store.create_feature_view(
            name=feature_view_name,
            version=feature_view_version,
            query=feature_group.select_all()
        )
    except:
        print('Feature view already existed. Skip creation.')
    
    feature_view = feature_store.get_feature_view(
        name=feature_view_name,
        version=feature_view_version
    )

    # Assume current_date is today's date, you can also set it as datetime.now()
    current_date = datetime.now()

    # Calculate the start and end times for "yesterday"
    fetch_data_from = current_date - timedelta(days=1)
    fetch_data_to = current_date - timedelta(days=1)

    # Set start_time to the beginning of yesterday (00:00)
    start_time = datetime(fetch_data_from.year, fetch_data_from.month, fetch_data_from.day, 0, 0, 0)

    # Set end_time to the end of yesterday (23:59:59)
    end_time = datetime(fetch_data_to.year, fetch_data_to.month, fetch_data_to.day, 23, 59, 59)

    # Fetch data using the feature store's get_batch_data method, with precise start and end times
    data = feature_view.get_batch_data(start_time=start_time, end_time=end_time)

    return data

def pull_model(model_name, model_version):
    project = hopsworks.login(
        project=PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY
    )

    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=model_name,
        version=model_version
    )  

    model_dir = model.download()
    model = joblib.load(Path(model_dir)  / f'{model_name}.pkl')

    return model

def upload_data(df, feature_group_name, feature_group_version):

    project = hopsworks.login(
        project=PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY
    )

    feature_store = project.get_feature_store()

    feature_group = feature_store.get_or_create_feature_group(
        name=feature_group_name,
        version=feature_group_version,
        # description=feature_group_description,
        primary_key = ['date'],
        event_time='date'
    )

    feature_group.insert(df, write_options={"wait_for_job": False})

def upload_model(model_name, model_description, metric_type, metric_value):

    # hopsworks models are automatically assigned version numbers so we don't have to define that here
    project = hopsworks.login(
        project=PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY
    )
    model_registry = project.get_model_registry()

    model = model_registry.sklearn.create_model(
        name=model_name,
        metrics={metric_type: metric_value}, # for comparing test errors over time, on streamlit dashboard
        description=model_description
    )

    # pip install hsml --upgrade
    # model.save(MODELS_DIR / 'model.pkl')
    # model.save(str(MODELS_DIR / 'model.pkl'))
    model.save(f'../models/ohlc/{model_name}.pkl')
