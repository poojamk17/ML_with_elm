import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
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

    kairos.set_metrics_tags({"category_3": "device_instance_173", "category_5": "tag_293",
                             "category_1": "industry_3_client_1107", "category_2": "gateway_instance_34"})
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
    print(df.info())
    print('outlier count',df[df['Value'] > 100000].count())
    df.drop(df[df['Value'] > 100000].index, inplace=True)
    print('skewness before removing outliers---->',df.skew())  # (-1.96 to +1.96)
    print('kurtosis before removing outliers---->',st.kurtosis(df['Value']))  # (-3 to +3)

    df2 = df[(df['Value'] > (np.mean(df['Value']) - (3 * np.std(df['Value'])))) & (
                df['Value'] < (np.mean(df['Value']) + 1.85 * np.std(df['Value'])))]
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
    print('shape after removing outliers---->',df2.shape)
    print('skewness after removing outliers---->',df2.skew())
    print('kurtosis after removing outliers---->',st.kurtosis(df2['Value']))
    import seaborn as sns
    sns.boxplot(df2['Value'])
    plt.show()
    sns.distplot(df2['Value'], bins=5)
    plt.show()
    df2["Epoch Time"] = pd.to_datetime(df2["Epoch Time"])


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
    print(df2.head())

    def feature_holidays(row):
        if row["Drop_me"] == "01-01":
            return "New Year"
        if row["Drop_me"] == "04-07":
            return "Ind Day"
        if row["Drop_me"] == "11-28":
            return "Thanksgiving"
        if row["Drop_me"] == "12-25":
            return "Christmas"
        return 'Other'

    def feature_worktime(row):
        if row["Hour"] > 7 & row["Hour"] <= 17:
            return "Worktime"
        return "NonWorkTime"

    def feature_peak(row):
        if row["Hour"] > 7 & row["Hour"] <= 22:
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
    df2 = df2.drop(["Drop_me"], axis=1)

    dummies = pd.get_dummies(df2[["Holiday", "Peak", "Work", "Weekend"]], prefix="Dummy")
    df2 = df2.join(dummies, lsuffix="_left")
    df2 = df2.drop(df2[["Holiday", "Peak", "Work", "Weekend"]], axis=1)
    df2.head()
    df2.index


    train_test_date = datetime.today() - timedelta(days=7)
    consum_test = df2[df2.index > train_test_date].copy()
    consum_train = df2[df2.index <= train_test_date].copy()

    raw_val = response[0]['values']

    print('consum_test info---->',consum_test.info())
    print('consum_train info---->',consum_train.info())

    def lag_features(lag_dataset, period_list):
        temp_data = lag_dataset["Value"]

        for period in period_list:
            lag_dataset["lag_consumption_{}".format(period)] = temp_data.shift(period)
            lag_dataset["mean_rolling_{}".format(period)] = temp_data.rolling(period).mean()
            lag_dataset["max_rolling_{}".format(period)] = temp_data.rolling(period).max()
            lag_dataset["min_rolling_{}".format(period)] = temp_data.rolling(period).min()

        for column in lag_dataset.columns[20:]:
            lag_dataset[column] = lag_dataset[column].fillna(lag_dataset.groupby("Minute")["Value"].transform("mean"))

        return lag_dataset

    consum_train = lag_features(consum_train, [7])
    consum_test = lag_features(consum_test, [7])
    consum_train.columns

    X_train = consum_train.drop("Value", axis=1)
    y_train = consum_train["Value"]
    X_test = consum_test.drop("Value", axis=1)
    y_test = consum_test["Value"]

    model_lgbm = LGBMRegressor(objective='rmse', n_estimators=3000, learning_rate=0.01, num_leaves=36,
                               min_child_samples=15,
                               n_jobs=-1, random_state=None, max_depth=3, reg_lambda=0.0, reg_alpha=0.0,
                               min_split_gain=0.0)
    eval_set_ALLRESTS = [(X_train, y_train), (X_test, y_test)]
    model_lgbm.fit(X_train, y_train, eval_set=eval_set_ALLRESTS, eval_metric='rmse', early_stopping_rounds=15,
                   verbose=20)

    consum_test["Prediction"] = model_lgbm.predict(X_test)
    print('Actually value and predicated value',consum_test[["Value", "Prediction"]])

    mean_sq = mean_squared_error(y_test, model_lgbm.predict(X_test))
    rmse = np.sqrt(mean_sq)

    mean_abs_sq = mean_absolute_error(y_test, model_lgbm.predict(X_test))

    print("Root Mean Squared Error : {}".format(rmse))
    print("Mean Absolute Error : {}".format(mean_abs_sq))

    from sklearn.metrics import r2_score
    print('r2_score',r2_score(y_test,model_lgbm.predict(X_test)))

    consum_test["Difference"] = np.abs(consum_test["Value"] - consum_test["Prediction"])
    print('sorting',consum_test["Difference"].sort_values(ascending=False)[:100])

    worst_days = consum_test.groupby(['Year', 'Month', 'Dayofmonth']).mean()[['Value', 'Prediction', 'Difference']]
    print('worst days---->',worst_days.sort_values(by="Difference", ascending=False)[:80])

    print('best days----->',worst_days.sort_values(by="Difference", ascending=True)[:50])
    consum_test['Difference'].sort_values(ascending=True)
    diff_10000_above = consum_test[(consum_test['Difference'] < 16000) & (consum_test['Difference'] > 10000)][
        ['Difference']]
    diff_5000_10000 = consum_test[(consum_test['Difference'] < 10000) & (consum_test['Difference'] > 5000)][['Difference']]
    diff_1000_5000 = consum_test[(consum_test['Difference'] < 5000) & (consum_test['Difference'] > 1000)][['Difference']]
    diff_500_1000 = consum_test[(consum_test['Difference'] < 1000) & (consum_test['Difference'] > 500)][['Difference']]
    diff_100_500 = consum_test[(consum_test['Difference'] < 500) & (consum_test['Difference'] > 100)][['Difference']]
    diff_50_100 = consum_test[(consum_test['Difference'] < 100) & (consum_test['Difference'] > 50)][['Difference']]
    diff_50_below = consum_test[(consum_test['Difference'] < 50)][['Difference']]
    diff_50_below['Difference'].sort_values(ascending=True)

    consum_test[['Value', 'Prediction', 'Difference']]
    consum_test_pre_diff = consum_test[["Value","Prediction", "Difference"]]
    print('Actually Predicated Difference------>',consum_test_pre_diff)
    ss = consum_test_pre_diff.index.astype(str).tolist()

    cc = [datetime(year=int(x.split(" ")[0].split("-")[0]),month=int(x.split(" ")[0].split("-")[1]),day=int(x.split(" ")[0].split("-")[2]),hour=int(x.split(" ")[1].split(":")[0]),minute =int(x.split(" ")[1].split(":")[1]),
    second = int(x.split(" ")[1].split(":")[2])
        )for x in ss]

    dd = [get_millisecond_from_date_time(x) for x in cc]
    #
    consum_test_pre_diff['timestamp'] = dd
    temp_1 = consum_test_pre_diff.copy()
    del temp_1['Difference'] , temp_1['Value']
    columns_titles = ['timestamp','Prediction']
    temp_1=temp_1.reindex(columns=columns_titles)
    temp_2 = consum_test_pre_diff.copy()
    del temp_2['Prediction'], temp_2['Value']
    columns_titles = ['timestamp','Difference']
    temp_2= temp_2.reindex(columns=columns_titles)
    temp_3 = consum_test_pre_diff.copy()
    del temp_3['Prediction'], temp_3['Difference']
    columns_titles = ['timestamp','Value']
    temp_3= temp_3.reindex(columns=columns_titles)
    tag_1 = temp_1.values.tolist()
    tag_2 = temp_2.values.tolist()
    tag_3 = temp_3.values.tolist()

    # pridction = [{"name": "ilens.live_data.raw", "datapoints": tag_1, 'tags': {"category_3": "device_instance_173", "category_5": "tag_10074",
    #                                                                            "category_1": "industry_3_client_1107", "category_2": "gateway_instance_34"}} ]
    # kairos.update_kairos_data(True, pridction)
    # diffe = [{"name": "ilens.live_data.raw", "datapoints": tag_2, 'tags': {"category_3":"device_instance_173","category_5":"tag_10075",
    #                                                                        "category_1": "industry_3_client_1107", "category_2": "gateway_instance_34"}} ]
    # kairos.update_kairos_data(True, diffe)
    #
    # raw = [{"name": "ilens.live_data.raw", "datapoints": tag_3, 'tags': {"category_3":"device_instance_173","category_5":"tag_10076" ,
    #                                                                      "category_1": "industry_3_client_1107", "category_2": "gateway_instance_34"}} ]
    # kairos.update_kairos_data(True, raw)

ml_insert()