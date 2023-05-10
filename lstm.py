import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
import seaborn as sns
import warnings

from lstm_class import DeepModelTS

warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
from kairos import Kairos
import pytz

def get_millisecond_from_date_time(date_time):
    tz = pytz.timezone('Asia/Kolkata')
    dt_with_tz = tz.localize(date_time, is_dst=None)
    seconds = (dt_with_tz - datetime(1970, 1, 1, tzinfo=pytz.UTC)).total_seconds()
    return seconds * 1000

def ml_insert():
    # Data collection
    try:
        start_date_time = datetime.strptime("04-04-2021 00:00:00", "%d-%m-%Y %H:%M:%S")
        end_date_time = datetime.strptime("11-04-2022 08:00:00", "%d-%m-%Y %H:%M:%S")
        # end = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        # end_date_time = datetime(year=int(end.split(' ')[0].split('-')[0]), month=int(end.split(' ')[0].split('-')[1]),
        #                          day=int(end.split(' ')[0].split('-')[2]),hour=int(end.split(' ')[1].split(':')[0]),
        #                          minute=int(end.split(' ')[1].split(':')[1]),second=int(end.split(' ')[1].split(':')[2]))

        start_millisecond = get_millisecond_from_date_time(start_date_time)
        end_millisecond = get_millisecond_from_date_time(end_date_time)

        kairos = Kairos()

        kairos.set_metrics_tags({"category_3": "device_instance_188", "category_5": "tag_293",
                             "category_1": "industry_3_client_1107", "category_2": "gateway_instance_2"})
        kairos.set_metrics_name("ilens.live_data.raw")
        kairos.set_start_end_date(True, start_millisecond, end_millisecond)

        kairos.set_metrics_aggregators([
        {
            "name": "filter",
            "filter_op": "lte",
            "threshold": "0"
        },
        {
            "name": "first",
            "sampling": {"value": "15", "unit": 'minutes'},
            "align_start_time": True
        },
        {
            "name": "diff"
        },
        {
            "name": "filter",
            "filter_op": "lte",
            "threshold": "0"
        }
            ])

        response = []
        try:
            response = kairos.get_kairos_data()

        except Exception:
            pass
    except Exception:
        traceback.print_exc()

    try:
        # Data Preparation

        df = pd.DataFrame(response[0]['values'], columns=["Epoch Time", "Value"])
        df['Epoch Time'] = pd.to_datetime(df['Epoch Time'], unit='ms').dt.tz_localize('GMT').dt.tz_convert(
        'Asia/Kolkata').dt.tz_localize(None)
        df.columns
        print("=" * 50)
        print("First Five Rows ", "\n")
        print(df.head(2), "\n")

        print("=" * 50)
        print("Information About Dataset", "\n")
        print(df.info(), "\n")

        print("=" * 50)
        print("Describe the Dataset ", "\n")
        print(df.describe(), "\n")

        print("=" * 50)
        print("Null Values t ", "\n")
        print(df.isnull().sum(), "\n")

        dataset = df
        dataset["Month"] = pd.to_datetime(df["Epoch Time"]).dt.month
        dataset["Year"] = pd.to_datetime(df["Epoch Time"]).dt.year
        dataset["Date"] = pd.to_datetime(df["Epoch Time"]).dt.date
        # dataset["Time"] = pd.to_datetime(df["Epoch Time"], errors='coerce').dt.time
        dataset['Time'] = pd.to_datetime(df['Epoch Time']).dt.strftime('%H:%M:%S')
        dataset["Week"] = pd.to_datetime(df["Epoch Time"]).dt.week
        dataset["Day"] = pd.to_datetime(df["Epoch Time"]).dt.day_name()
        dataset = df.set_index("Epoch Time")
        dataset.index = pd.to_datetime(dataset.index)
        deep_learner = DeepModelTS(
            data=df,
            Y_var='Value',
            lag=6,
            LSTM_layer_depth=50,
            epochs=10,
            batch_size=256,
            train_test_split=0.15
        )

        # Fitting the model
        model = deep_learner.LSTModel()

        deep_learner = DeepModelTS(
            data=df,
            Y_var='Value',
            lag=24,  # 24 past hours are used
            LSTM_layer_depth=50,
            epochs=10,
            batch_size=256,
            train_test_split=0.15
        )
        model = deep_learner.LSTModel()

        # Defining the lag that we used for training of the model
        lag_model = 24
        # Getting the last period
        ts = df['Value'].tail(lag_model).values.tolist()
        # Creating the X matrix for the model
        X, _ = deep_learner.create_X_Y(ts, lag=lag_model)
        # Getting the forecast
        yhat = model.predict(X)

        yhat = deep_learner.predict()
        # Constructing the forecast dataframe
        fc = df.tail(len(yhat)).copy()
        fc.reset_index(inplace=True)
        fc['forecast'] = yhat
        # Ploting the forecasts
        plt.figure(figsize=(12, 8))
        for dtype in ['Value', 'forecast']:
            plt.plot(
                'Epoch Time',
                dtype,
                data=fc,
                label=dtype,
                alpha=0.8
            )
        # plt.legend()
        # plt.grid()
        plt.show()

        # Creating the model using full data and forecasting n steps ahead
        deep_learner = DeepModelTS(
        data = df,
        Y_var = 'Value',
        lag = 48,
        LSTM_layer_depth = 64,
        epochs = 10,
        train_test_split = 0
        )
        # Fitting the model
        deep_learner.LSTModel()
        # Forecasting n steps ahead
        n_ahead = 672
        yhat = deep_learner.predict_n_ahead(n_ahead)
        yhat = [y[0][0] for y in yhat]

        last = df['Epoch Time'].get(df.index[-1])
        lastone = last
        ff = []
        d = pd.DataFrame()
        for day in range(0, 672):
            last = last + pd.to_timedelta(15, unit='minutes')
            ff.append(last)
        d['Epoch Time'] = ff
        d['Value'] = yhat

        ss = d['Epoch Time'].tolist()
        d.index = d["Epoch Time"]
        df2 = d.drop(["Epoch Time"], axis=1)
        ss = df2.index.astype(str).tolist()
        cc = [datetime(year=int(x.split(" ")[0].split("-")[0]), month=int(x.split(" ")[0].split("-")[1]),
                       day=int(x.split(" ")[0].split("-")[2]), hour=int(x.split(" ")[1].split(":")[0]),
                       minute=int(x.split(" ")[1].split(":")[1]),
                       second=int(x.split(" ")[1].split(":")[2])
                       ) for x in ss]
        #
        dd = [get_millisecond_from_date_time(x) for x in cc]

        d['Epoch Time'] = dd
        temp_1 = d.copy(deep=True)
        # del temp_1['Difference'] , temp_1['Value']
        columns_titles = ['Epoch Time', 'Value']
        temp_1 = temp_1.reindex(columns=columns_titles)
        tag_1 = temp_1.values.tolist()


        pridction = [{"name": "ilens.live_data.raw", "datapoints": tag_1, 'tags': {"category_3": "device_instance_188", "category_5": "tag_11585",
                                                                                   "category_1": "industry_3_client_1107", "category_2": "gateway_instance_78"}} ]
        kairos.update_kairos_data(True, pridction)
        # diffe = [{"name": "ilens.live_data.raw", "datapoints": tag_2, 'tags': {"category_3":"device_instance_486","category_5":"tag_10075",
        #                                                                        "category_1": "industry_3_client_1107", "category_2": "gateway_instance_78"}} ]
        # kairos.update_kairos_data(True, diffe)
        #
        # raw = [{"name": "ilens.live_data.raw", "datapoints": tag_3, 'tags': {"category_3":"device_instance_499","category_5":"tag_10076" ,
        #                                                                      "category_1": "industry_3_client_1107", "category_2": "gateway_instance_88"}} ]
        # kairos.update_kairos_data(True, raw)
    except:
        traceback.print_exc()


if __name__=="__main__":
    ml_insert()