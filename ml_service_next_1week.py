import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from xgboost import plot_importance, plot_tree
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
    start_date_time = datetime.strptime("01-01-2020 00:00:00", "%d-%m-%Y %H:%M:%S")
    end = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    end_date_time = datetime(year=int(end.split(' ')[0].split('-')[0]), month=int(end.split(' ')[0].split('-')[1]),
                             day=int(end.split(' ')[0].split('-')[2]),hour=int(end.split(' ')[1].split(':')[0]),
                             minute=int(end.split(' ')[1].split(':')[1]),second=int(end.split(' ')[1].split(':')[2]))

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

    df = pd.DataFrame(response[1]['values'], columns=["Epoch Time", "Value"])
    df['Epoch Time'] = pd.to_datetime(df['Epoch Time'], unit='ms').dt.tz_localize('GMT').dt.tz_convert(
        'Asia/Kolkata').dt.tz_localize(None)
    df.columns
    print(df.info())
    print('skewness before removing outliers---->',df.skew())  # (-1.96 to +1.96)
    print('kurtosis before removing outliers---->',st.kurtosis(df['Value']))  # (-3 to +3)

    def remove_outlier(df_in, col_name):
        q1 = df_in[col_name].quantile(0.25)
        q3 = df_in[col_name].quantile(0.75)
        iqr = q3 - q1  # Interquartile range
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
        return df_out

    df2 = remove_outlier(df, 'Value')
    # df2 = df[(df['Value'] > (np.mean(df['Value']) - (3 * np.std(df['Value'])))) & (
    #             df['Value'] < (np.mean(df['Value']) + 0.34 * np.std(df['Value'])))]
    print('shape after removing outliers---->',df2.shape)
    print('skewness after removing outliers---->',df2.skew())
    print('kurtosis after removing outliers---->',st.kurtosis(df2['Value']))

    last = df2['Epoch Time'].get(df2.index[-1])
    lastone=last
    ff = []
    d = pd.DataFrame()
    for day in range(0, 672):
        last = last + pd.to_timedelta(15, unit='minutes')
        ff.append(last)
    d['Epoch Time'] = ff
    d['Value'] =0

    df2["Epoch Time"] = pd.to_datetime(df2["Epoch Time"])
    df2.index = df2["Epoch Time"]
    df2.drop('Epoch Time', axis=1, inplace=True)


    d["Epoch Time"] = pd.to_datetime(d["Epoch Time"])
    d.index = d["Epoch Time"]
    d.drop('Epoch Time', axis=1, inplace=True)

    train = df2
    test=d

    def create_features(df, label=None):
        """
        Creates time series features from datetime index
        """
        df['date'] = df.index
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.weekofyear

        X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
                'dayofyear', 'dayofmonth', 'weekofyear']]
        if label:
            y = df[label]
            return X, y
        return X

    X_train, y_train = create_features(train, label='Value')
    X_test, y_test = create_features(test, label='Value')

    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train)],
            early_stopping_rounds=50,
            verbose=False)  # Change verbose to True if you want to see it train

    test['MW_Prediction'] = reg.predict(X_test)
    # pjme_all = pd.concat([test, train], sort=False)
    # plot
    pjme_all = pd.concat([test, train], sort=False)
    _ = pjme_all[['Value', 'MW_Prediction']].plot(figsize=(15, 5))
    plt.show()
    print('RMSE',np.sqrt(mean_squared_error(y_true=test['Value'],
                   y_pred=test['MW_Prediction'])))

    print('MAE',mean_absolute_error(y_true=test['Value'],
                   y_pred=test['MW_Prediction']))

    from sklearn.metrics import r2_score
    print(r2_score(y_true=test['Value'],
             y_pred=test['MW_Prediction']))

    def mean_absolute_percentage_error(y_true, y_pred):
        """Calculates MAPE given y_true and y_pred"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mean_absolute_percentage_error(y_true=test['Value'],y_pred=test['MW_Prediction'])
    pjme_all = pd.concat([test, train], sort=False)
    _ = pjme_all[['Value', 'MW_Prediction']].plot(figsize=(15, 5))
    test['MW_Prediction'].plot(figsize=(15, 5))

    plt.show()

    consum_test_pre_diff = test[["MW_Prediction"]]
    print('Predicated------>',consum_test_pre_diff)
    ss = consum_test_pre_diff.index.astype(str).tolist()

    cc = [datetime(year=int(x.split(" ")[0].split("-")[0]),month=int(x.split(" ")[0].split("-")[1]),day=int(x.split(" ")[0].split("-")[2]),hour=int(x.split(" ")[1].split(":")[0]),minute =int(x.split(" ")[1].split(":")[1]),
    second = int(x.split(" ")[1].split(":")[2])
        )for x in ss]
    #
    dd = [get_millisecond_from_date_time(x) for x in cc]

    consum_test_pre_diff['timestamp'] = dd
    temp_1 = consum_test_pre_diff.copy(deep=True)
    # del temp_1['Difference'] , temp_1['Value']
    columns_titles = ['timestamp','MW_Prediction']
    temp_1=temp_1.reindex(columns=columns_titles)
    tag_1 = temp_1.values.tolist()
    print('-------')
    # pridction = [{"name": "ilens.live_data.raw", "datapoints": tag_1, 'tags': {"category_3": "device_instance_486", "category_5": "tag_10074",
    #                                                                            "category_1": "industry_3_client_1107", "category_2": "gateway_instance_78"}} ]
    # kairos.update_kairos_data(True, pridction)
    # diffe = [{"name": "ilens.live_data.raw", "datapoints": tag_2, 'tags': {"category_3":"device_instance_188","category_5":"tag_10075",
    #                                                                        "category_1": "industry_3_client_1107", "category_2": "gateway_instance_2"}} ]
    # kairos.update_kairos_data(True, diffe)
    #
    # raw = [{"name": "ilens.live_data.raw", "datapoints": tag_3, 'tags': {"category_3":"device_instance_188","category_5":"tag_10076" ,
    #                                                                      "category_1": "industry_3_client_1107", "category_2": "gateway_instance_2"}} ]
    # kairos.update_kairos_data(True, raw)

ml_insert()