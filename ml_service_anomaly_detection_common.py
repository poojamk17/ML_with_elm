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
    start_date_time = datetime.strptime("01-03-2022 00:00:00", "%d-%m-%Y %H:%M:%S")
    end_date_time = datetime.strptime("01-04-2022 00:00:00", "%d-%m-%Y %H:%M:%S")

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

    # Gaussian Mixture Method
    #
    # from sklearn.mixture import GaussianMixture
    # gm = GaussianMixture(random_state=0)
    # gm.fit(df['Value'].values.reshape(-1, 1))
    #
    # densities = gm.score_samples(df['Value'].values.reshape(-1, 1))
    # density_threshold = np.percentile(densities, 1)
    #
    # gm_result = [-1 if i < density_threshold else 0 for i in densities]
    # gm_result_df = pd.DataFrame()
    # gm_result_df['Epoch Time'] = df['Epoch Time']
    # gm_result_df['Value'] = df['Value']
    # gm_result_df['anomaly'] = ['anomaly' if i == -1 else 'normal' for i in gm_result]
    # df['Anomaly_gm'] = gm_result_df['anomaly']
    #
    # # Isolation Forest Method
    #
    # from sklearn.ensemble import IsolationForest
    # from scipy import stats
    #
    # outliers_fraction = 0.02
    #
    # model = IsolationForest().fit(df['Value'].values.reshape(-1, 1))
    # scores_pred = model.decision_function(df['Value'].values.reshape(-1, 1))
    # threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
    # print(threshold)
    # labels = [('anomaly' if x < threshold else 'normal') for x in scores_pred]
    # df['Anomaly_iforest'] = labels

    # print('Anomaly_iforest',df['Anomaly_iforest'].value_counts())
    # print('Anomaly_gm',df['Anomaly_gm'].value_counts())
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
    anomaly = [{"name": "ilens.live_data.raw", "datapoints": tag_1, 'tags': {"category_3": "device_instance_188", "category_5": "tag_10088",
                                                                               "category_1": "industry_3_client_1107", "category_2": "gateway_instance_2"}} ]
    kairos.update_kairos_data(True, anomaly)
    # diffe = [{"name": "ilens.live_data.raw", "datapoints": tag_2, 'tags': {"category_3":"device_instance_499","category_5":"tag_10075",
    #                                                                        "category_1": "industry_3_client_1107", "category_2": "gateway_instance_88"}} ]
    # kairos.update_kairos_data(True, diffe)
    #
    # raw = [{"name": "ilens.live_data.raw", "datapoints": tag_3, 'tags': {"category_3":"device_instance_499","category_5":"tag_10076" ,
    #                                                                      "category_1": "industry_3_client_1107", "category_2": "gateway_instance_88"}} ]
    # kairos.update_kairos_data(True, raw)

ml_insert()