import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
import warnings
warnings.filterwarnings('ignore')
from fbprophet import Prophet
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
    start_date_time = datetime.strptime("01-11-2019 00:00:00", "%d-%m-%Y %H:%M:%S")
    end_date_time = datetime.strptime("23-12-2020 09:00:00", "%d-%m-%Y %H:%M:%S")
    # end = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    # end_date_time = datetime(year=int(end.split(' ')[0].split('-')[0]), month=int(end.split(' ')[0].split('-')[1]),
    #                          day=int(end.split(' ')[0].split('-')[2]),hour=int(end.split(' ')[1].split(':')[0]),
    #                          minute=int(end.split(' ')[1].split(':')[1]),second=int(end.split(' ')[1].split(':')[2]))

    start_millisecond = get_millisecond_from_date_time(start_date_time)
    end_millisecond = get_millisecond_from_date_time(end_date_time)

    kairos = Kairos()

    kairos.set_metrics_tags({"category_3": "device_instance_486", "category_5": "tag_293",
                             "category_1": "industry_3_client_1107", "category_2": "gateway_instance_78"})
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
    print('shape after removing outliers---->',df2.shape)
    print('skewness after removing outliers---->',df2.skew())
    print('kurtosis after removing outliers---->',st.kurtosis(df2['Value']))

    import seaborn as sns
    sns.boxplot(df2['Value'])
    plt.show()
    print(df2[df2['Value']<1000])

    df2["Hour"] = df2["Epoch Time"].dt.hour
    df2["Minute"] = df2["Epoch Time"].dt.minute
    df2["Day"] = df2["Epoch Time"].dt.dayofweek
    df2["Month"] = df2["Epoch Time"].dt.month
    df2["Year"] = df2["Epoch Time"].dt.year
    df2["Q"] = df2["Epoch Time"].dt.quarter
    df2["Dayofyear"] = df2["Epoch Time"].dt.dayofyear
    df2["Dayofmonth"] = df2["Epoch Time"].dt.day
    df2["Weekofyear"] = df2["Epoch Time"].dt.weekofyear


    last = df2['Epoch Time'].get(df2.index[-1])
    lastone = last
    ff = []
    d = pd.DataFrame()
    for day in range(0, 672):
        last = last + pd.to_timedelta(15, unit='minutes')
        ff.append(last)
    d['Epoch Time'] = ff
    d['Value'] = 0

    d.index = d["Epoch Time"]
    d.drop('Epoch Time', axis=1, inplace=True)

    df2.index = df2["Epoch Time"]
    df2 = df2.drop(["Epoch Time"], axis=1)
    df2.head()

    elm_train = df2
    elm_test = d
    test = elm_test.copy()
    # Format data for prophet model using ds and y
    elm_train.reset_index() \
        .rename(columns={'Datetime': 'ds',
                         'Value': 'y'})
    # Setup and train model and fit
    model = Prophet()

    model.fit(elm_train.reset_index() \
              .rename(columns={'Epoch Time': 'ds',
                               'Value': 'y'}))
    # Predict on training set with model
    elm_test_fcst = model.predict(df=elm_test.reset_index() \
                                   .rename(columns={'Epoch Time': 'ds'}))
    print(elm_test_fcst.head())

    # Plot the forecast
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    fig = model.plot(elm_test_fcst,ax=ax)
    # Plot the components of the model
    fig = model.plot_components(elm_test_fcst)
    plt.show()
    # # removing -ve value
    elm_neg = elm_test_fcst[elm_test_fcst['yhat'] < 0]

    elm_neg["Hour"] = elm_test_fcst["ds"].dt.hour
    elm_neg["Minute"] = elm_test_fcst["ds"].dt.minute
    elm_neg["Day"] = elm_test_fcst["ds"].dt.dayofweek
    elm_neg["Month"] = elm_test_fcst["ds"].dt.month
    elm_neg["Year"] = elm_test_fcst["ds"].dt.year
    elm_neg["Q"] = elm_test_fcst["ds"].dt.quarter
    elm_neg["Dayofyear"] = elm_test_fcst["ds"].dt.dayofyear
    elm_neg["Dayofmonth"] = elm_test_fcst["ds"].dt.day
    elm_neg["Weekofyear"] = elm_test_fcst["ds"].dt.weekofyear

    elm_neg.index = elm_neg["ds"]
    elm_test_fcst.index = elm_test_fcst['ds']
    for i in range(len(elm_neg)):
        elm_neg['yhat'][i] = elm_train[
            (elm_train['Day'] == elm_neg['Day'][i]) &
            (elm_train['Hour'] == elm_neg['Hour'][i]) & (
                        elm_train['Minute'] == elm_neg['Minute'][i])].mean()['Value']
    elm_test_fcst.update(elm_neg)

    # Plot the forecast
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    fig = model.plot(elm_test_fcst, ax=ax)
    plt.show()
    print('rmse   ', np.sqrt(mean_squared_error(y_true=test['Value'],
                                                y_pred=elm_test_fcst['yhat'])))
    print('mae   ', mean_absolute_error(y_true=test['Value'],
                                        y_pred=elm_test_fcst['yhat']))
    consum_test_pre_diff = elm_test_fcst[['ds',"yhat"]]
    consum_test_pre_diff.index = consum_test_pre_diff["ds"]
    print(consum_test_pre_diff['yhat'].sort_values()[:40])

    elm_test_fcst['yhat'].plot(figsize=(15, 5))
    plt.show()

    print('Predicated------>',consum_test_pre_diff)
    ss = consum_test_pre_diff.index.astype(str).tolist()

    cc = [datetime(year=int(x.split(" ")[0].split("-")[0]),month=int(x.split(" ")[0].split("-")[1]),day=int(x.split(" ")[0].split("-")[2]),hour=int(x.split(" ")[1].split(":")[0]),minute =int(x.split(" ")[1].split(":")[1]),
    second = int(x.split(" ")[1].split(":")[2])
        )for x in ss]
    #
    dd = [get_millisecond_from_date_time(x) for x in cc]

    consum_test_pre_diff['ds'] = dd
    temp_1 = consum_test_pre_diff.copy(deep=True)
    # del temp_1['Difference'] , temp_1['Value']
    columns_titles = ['ds','yhat']
    temp_1=temp_1.reindex(columns=columns_titles)
    tag_1 = temp_1.values.tolist()
    print('-------')
    # pridction = [{"name": "ilens.live_data.raw", "datapoints": tag_1, 'tags': {"category_3": "device_instance_486", "category_5": "tag_10089",
    #                                                                            "category_1": "industry_3_client_1107", "category_2": "gateway_instance_78"}} ]
    # kairos.update_kairos_data(True, pridction)

ml_insert()