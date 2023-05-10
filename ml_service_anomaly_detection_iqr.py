import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
from kairos import Kairos
import pytz

def get_millisecond_from_date_time(date_time):
    tz = pytz.timezone('Asia/Kolkata')
    dt_with_tz = tz.localize(date_time, is_dst=None)
    seconds = (dt_with_tz - datetime(1970, 1, 1, tzinfo=pytz.UTC)).total_seconds()
    return seconds * 1000

def ml_insert():
    start_date_time = datetime.strptime("01-03-2020 00:00:00", "%d-%m-%Y %H:%M:%S")
    end_date_time = datetime.strptime("01-04-2020 00:00:00", "%d-%m-%Y %H:%M:%S")

    start_millisecond = get_millisecond_from_date_time(start_date_time)
    end_millisecond = get_millisecond_from_date_time(end_date_time)

    kairos = Kairos()

    kairos.set_metrics_tags({"category_3": "device_instance_188", "category_5": "tag_288",
                             "category_1": "industry_3_client_1107", "category_2": "gateway_instance_2"})
    kairos.set_metrics_name("ilens.live_data.raw")
    kairos.set_start_end_date(True, start_millisecond, end_millisecond)

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
    print(df.describe())

    # IQR METHOD

    def remove_outlier(df_in, col_name):
        q1 = df_in[col_name].quantile(0.25)
        q3 = df_in[col_name].quantile(0.75)
        print('q1  ', q1, '       ', 'q3  ', q3)
        iqr = q3 - q1  # Interquartile range
        print('iqr  ', iqr)
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        print('upperbound: ', fence_high, 'lowerbound: ', fence_low)
        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
        return df_out

    df2 = remove_outlier(df, 'Value')
    df['Anomaly_iqr'] = ~df['Epoch Time'].isin(df2['Epoch Time']) | ~df['Value'].isin(df2['Value'])

    print('Anomaly_iqr',df['Anomaly_iqr'].value_counts())

    df.index = df["Epoch Time"]
    df = df.drop(["Epoch Time"], axis=1)
    df.head()

    print(df[df['Anomaly_iqr']==True])
    df3=df[df['Anomaly_iqr']==True][["Value"]]
    ss = df3.index.astype(str).tolist()

    cc = [datetime(year=int(x.split(" ")[0].split("-")[0]), month=int(x.split(" ")[0].split("-")[1]),
                   day=int(x.split(" ")[0].split("-")[2]), hour=int(x.split(" ")[1].split(":")[0]),
                   minute=int(x.split(" ")[1].split(":")[1]),
                   second=int(x.split(" ")[1].split(":")[2])
                   ) for x in ss]
    #
    dd = [get_millisecond_from_date_time(x) for x in cc]

    df3['timestamp'] = dd
    temp_1 = df3.copy(deep=True)
    columns_titles = ['timestamp', 'Value']
    temp_1 = temp_1.reindex(columns=columns_titles)
    tag_1 = temp_1.values.tolist()

    print('-------')
    # anomaly = [{"name": "ilens.live_data.raw", "datapoints": tag_1, 'tags': {"category_3": "device_instance_188", "category_5": "tag_10088",
    #                                                                            "category_1": "industry_3_client_1107", "category_2": "gateway_instance_2"}} ]
    # kairos.update_kairos_data(True, anomaly)

ml_insert()