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
    start_date_time = datetime.strptime("01-04-2020 00:00:00", "%d-%m-%Y %H:%M:%S")
    end_date_time = datetime.strptime("28-03-2022 08:00:00", "%d-%m-%Y %H:%M:%S")
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

    df2['Value'] = np.log(df2['Value'])
    # Works with a ds and y column names
    df2.rename(columns={'Epoch Time': 'ds', 'Value': 'y'}, inplace=True)

    # Create Future Dates of 7 days
    m = Prophet()
    m.add_country_holidays(country_name='IN')

    # Python
    def nfl_sunday(ds):
        date = pd.to_datetime(ds)
        if date.weekday() == 6 or date.weekday() == 5:
            return 1
        else:
            return 0

    df2['sat_sunday'] = df2['ds'].apply(nfl_sunday)

    m.add_regressor('sat_sunday')

    m.fit(df2)
    future_dates = m.make_future_dataframe(periods=672, freq='15 min')
    future_dates['sat_sunday'] = future_dates['ds'].apply(nfl_sunday)
    forecast = m.predict(future_dates)
    fig = m.plot_components(forecast)
    forecast.plot(x='ds', y='yhat')
    plt.show()

    forecast['exp_yhat'] = np.exp(forecast['yhat'])
    print(forecast.head())
    future=forecast[forecast.shape[0]-672:]
    future.plot(x='ds', y='exp_yhat')
    plt.show()

    consum_test_pre_diff = future[['ds',"exp_yhat"]]
    consum_test_pre_diff.index = consum_test_pre_diff["ds"]

    print('Predicated------>',consum_test_pre_diff)
    ss = consum_test_pre_diff.ds.astype(str).tolist()

    cc = [datetime(year=int(x.split(" ")[0].split("-")[0]),month=int(x.split(" ")[0].split("-")[1]),day=int(x.split(" ")[0].split("-")[2]),hour=int(x.split(" ")[1].split(":")[0]),minute =int(x.split(" ")[1].split(":")[1]),
    second = int(x.split(" ")[1].split(":")[2])
        )for x in ss]

    dd = [get_millisecond_from_date_time(x) for x in cc]

    consum_test_pre_diff['ds'] = dd
    temp_1 = consum_test_pre_diff.copy(deep=True)
    # del temp_1['Difference'] , temp_1['Value']
    columns_titles = ['ds','exp_yhat']
    temp_1=temp_1.reindex(columns=columns_titles)
    tag_1 = temp_1.values.tolist()
    print('-------')
    # pridction = [{"name": "ilens.live_data.raw", "datapoints": tag_1, 'tags': {"category_3": "device_instance_188", "category_5": "tag_10089",
    #                                                                            "category_1": "industry_3_client_1107", "category_2": "gateway_instance_2"}} ]
    # kairos.update_kairos_data(True, pridction)

ml_insert()