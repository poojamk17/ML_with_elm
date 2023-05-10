import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
from kairos import Kairos
import pytz
from sklearn.ensemble import IsolationForest
from scipy import stats

def get_millisecond_from_date_time(date_time):
    tz = pytz.timezone('Asia/Kolkata')
    dt_with_tz = tz.localize(date_time, is_dst=None)
    seconds = (dt_with_tz - datetime(1970, 1, 1, tzinfo=pytz.UTC)).total_seconds()
    return seconds * 1000

def ml_insert():
    start_date_time = datetime.strptime("01-04-2020 00:00:00", "%d-%m-%Y %H:%M:%S")
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

    df = pd.DataFrame(response[0]['values'], columns=["Epoch Time", "Value"])
    df['Epoch Time'] = pd.to_datetime(df['Epoch Time'], unit='ms').dt.tz_localize('GMT').dt.tz_convert(
        'Asia/Kolkata').dt.tz_localize(None)
    df.columns
    # adding holidays
    hol = pd.read_csv("elm_holidays_2020.csv", sep=',')
    hol['DATE'] = pd.to_datetime(hol['DATE'])
    hol["Drop_me"] = hol["DATE"].dt.strftime("%m-%d")
    hol_kar = hol[hol['KARNATAKA'] == 'H']
    hol_kar.reset_index(inplace=True)

    print(df.info())
    print('skewness before removing outliers---->',df.skew())  # (-1.96 to +1.96)
    print('kurtosis before removing outliers---->',st.kurtosis(df['Value']))  # (-3 to +3)
    #
    # def remove_outlier(df_in, col_name):
    #     q1 = df_in[col_name].quantile(0.25)
    #     q3 = df_in[col_name].quantile(0.75)
    #     iqr = q3 - q1  # Interquartile range
    #     fence_low = q1 - 1.5 * iqr
    #     fence_high = q3 + 1.5 * iqr
    #     df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    #     return df_out
    #
    # df2 = remove_outlier(df, 'Value')

    outliers_fraction = 0.03

    model = IsolationForest().fit(df['Value'].values.reshape(-1, 1))
    scores_pred = model.decision_function(df['Value'].values.reshape(-1, 1))
    threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
    print(threshold)
    labels = [('anomaly' if x < threshold else 'normal') for x in scores_pred]
    df['Anomaly_iforest'] = labels
    df2 = df[df['Anomaly_iforest'] == 'normal']

    # df2 = df[(df['Value'] > (np.mean(df['Value']) - (3 * np.std(df['Value'])))) & (
    #             df['Value'] < (np.mean(df['Value']) + 0.34 * np.std(df['Value'])))]
    print('shape after removing outliers---->',df2.shape)
    print('skewness after removing outliers---->',df2.skew())
    print('kurtosis after removing outliers---->',st.kurtosis(df2['Value']))

    last = df2['Epoch Time'].get(df2.index[-1])
    lastone = last
    ff = []
    d = pd.DataFrame()
    for day in range(0, 672):
        last = last + pd.to_timedelta(15, unit='minutes')
        ff.append(last)
    d['Epoch Time'] = ff
    d['Value'] = 0
    df2 = df2.append(d, ignore_index=True)

    df2["Hour"] = df2["Epoch Time"].dt.hour
    df2["Minute"] = df2["Epoch Time"].dt.minute

    df2["Day"] = df2["Epoch Time"].dt.dayofweek
    df2["Month"] = df2["Epoch Time"].dt.month
    df2["Year"] = df2["Epoch Time"].dt.year
    df2["Q"] = df2["Epoch Time"].dt.quarter
    df2["Dayofyear"] = df2["Epoch Time"].dt.dayofyear
    df2["Dayofmonth"] = df2["Epoch Time"].dt.day
    df2["Weekofyear"] = df2["Epoch Time"].dt.weekofyear

    df2["Drop_me"] = df2["Epoch Time"].dt.strftime("%m-%d")

    df2.index = df2["Epoch Time"]
    df2 = df2.drop(["Epoch Time"], axis=1)
    df2.head()

    # Be ready for some bruteforce if functions
    def feature_holidays(row):
        for i in range(hol_kar.shape[0]):
            if row["Drop_me"] == hol_kar['Drop_me'][i]:
                return hol_kar['HOLIDAYS'][i]
        return 'Other'

    def feature_worktime(row):
        if (row["Hour"] > 7) & (row["Hour"] < 19):
            return "Worktime"
        return "NonWorkTime"

    def feature_peak(row):
        if (row["Hour"] > 7) & (row["Hour"] <= 18):
            return "Peak"
        return "NonPeak"

    def feature_weekend(row):
        if row["Day"] == 5 or row["Day"] == 6:
            return "Weekend"
        return "NonWeekend"

    df2["Holiday"] = df2.apply(lambda row: feature_holidays(row), axis=1)
    df2["Work"] = df2.apply(lambda row: feature_worktime(row), axis=1)
    df2["Peak"] = df2.apply(lambda row: feature_peak(row), axis=1)
    df2["Weekend"] = df2.apply(lambda row: feature_weekend(row), axis=1)

    print('value_count    ', df2['Holiday'].value_counts())

    df2 = df2.drop(["Drop_me"], axis=1)
    df2 = df2.drop(['Anomaly_iforest'], axis=1)
    dummies = pd.get_dummies(df2[["Holiday", "Peak", "Work", "Weekend"]], prefix="Dummy")
    df2 = df2.join(dummies, lsuffix="_left")
    df2 = df2.drop(df2[["Holiday", "Peak", "Work", "Weekend"]], axis=1)

    def lag_features(lag_dataset, period_list):

        temp_data = lag_dataset["Value"]

        for period in period_list:
            lag_dataset["lag_consumption_{}".format(period)] = temp_data.shift(period)
        #         lag_dataset["mean_rolling_{}".format(period)] = temp_data.rolling(period).mean()
        #         lag_dataset["max_rolling_{}".format(period)] = temp_data.rolling(period).max()
        #         lag_dataset["min_rolling_{}".format(period)] = temp_data.rolling(period).min()

        for column in lag_dataset.columns[20:]:
            lag_dataset[column] = lag_dataset[column].fillna(lag_dataset.groupby("Minute")["Value"].transform("mean"))

        return lag_dataset

    df2 = lag_features(df2, [672])

    def lag_features1(lag_dataset, period_list):

        temp_data = lag_dataset["lag_consumption_672"]

        for period in period_list:
            #         lag_dataset["lag_consumption_{}".format(period)] = temp_data.shift(period)
            lag_dataset["mean_rolling_{}".format(period)] = temp_data.rolling(period).mean()
            lag_dataset["max_rolling_{}".format(period)] = temp_data.rolling(period).max()
            lag_dataset["min_rolling_{}".format(period)] = temp_data.rolling(period).min()

        for column in lag_dataset.columns[20:]:
            lag_dataset[column] = lag_dataset[column].fillna(lag_dataset.groupby("Minute")["Value"].transform("mean"))

        return lag_dataset

    df2 = lag_features1(df2, [672])

    split_date = lastone
    elm_train = df2.loc[df2.index <= split_date]
    elm_test = df2.loc[df2.index > split_date]

    test = elm_test.copy()
    # Train - Test
    X_train = elm_train.drop("Value", axis=1)
    y_train = elm_train["Value"]
    X_test = elm_test.drop("Value", axis=1)
    y_test = elm_test["Value"]

    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train)],
            early_stopping_rounds=50,
            verbose=False)  # Change verbose to True if you want to see it train

    elm_test['Prediction'] = reg.predict(X_test)
    pjme_all = pd.concat([elm_test, elm_train], sort=False)

    print('rmse   ',np.sqrt(mean_squared_error(y_true=test['Value'],
                   y_pred=elm_test['Prediction'])))
    # removing -ve value
    elm_neg = elm_test[elm_test['Prediction'] < 0]

    print('mae   ',mean_absolute_error(y_true=test['Value'],
                   y_pred=elm_test['Prediction']))

    for i in range(len(elm_neg)):
        elm_neg['Prediction'][i] = elm_train[
            (elm_train['Day'] == elm_neg['Day'][i]) & (elm_train['Hour'] == elm_neg['Hour'][i]) & (
                        elm_train['Minute'] == elm_neg['Minute'][i])].mean()['Value']
    elm_test.update(elm_neg)

    consum_test_pre_diff = elm_test[["Prediction"]]
    print(consum_test_pre_diff['Prediction'].sort_values()[:40])

    # plot
    pjme_all = pd.concat([elm_test, elm_train], sort=False)
    _ = pjme_all[['Value', 'Prediction']].plot(figsize=(15, 5))
    elm_test['Prediction'].plot(figsize=(15, 5))
    plt.show()

    print('Predicated------>',consum_test_pre_diff)

    ss = consum_test_pre_diff.index.astype(str).tolist()

    cc = [datetime(year=int(x.split(" ")[0].split("-")[0]),month=int(x.split(" ")[0].split("-")[1]),day=int(x.split(" ")[0].split("-")[2]),hour=int(x.split(" ")[1].split(":")[0]),minute =int(x.split(" ")[1].split(":")[1]),
    second = int(x.split(" ")[1].split(":")[2])
        )for x in ss]

    dd = [get_millisecond_from_date_time(x) for x in cc]

    consum_test_pre_diff['timestamp'] = dd
    temp_1 = consum_test_pre_diff.copy(deep=True)
    # del temp_1['Difference'] , temp_1['Value']
    columns_titles = ['timestamp','Prediction']
    temp_1=temp_1.reindex(columns=columns_titles)
    tag_1 = temp_1.values.tolist()
    print('-------')
    pridction = [{"name": "ilens.live_data.raw", "datapoints": tag_1, 'tags': {"category_3": "device_instance_188", "category_5": "tag_10085",
                                                                               "category_1": "industry_3_client_1107", "category_2": "gateway_instance_78"}} ]
    kairos.update_kairos_data(True, pridction)
    # diffe = [{"name": "ilens.live_data.raw", "datapoints": tag_2, 'tags': {"category_3":"device_instance_499","category_5":"tag_10075",
    #                                                                        "category_1": "industry_3_client_1107", "category_2": "gateway_instance_88"}} ]
    # kairos.update_kairos_data(True, diffe)
    #
    # raw = [{"name": "ilens.live_data.raw", "datapoints": tag_3, 'tags': {"category_3":"device_instance_499","category_5":"tag_10076" ,
    #                                                                      "category_1": "industry_3_client_1107", "category_2": "gateway_instance_88"}} ]
    # kairos.update_kairos_data(True, raw)

ml_insert()