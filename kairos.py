import json

import requests

from app_config import AppConfig

app_conf = AppConfig()
kairos_server = app_conf.get_kairos_host()


# class to get datas from kairosdb
def create_obj_update_query_with_array_of_object(name, timestamp, value, tags):
    return {"name": name, "timestamp": timestamp, "value": value, "tags": tags}


class Kairos:
    def __init__(self):
        self.query = {"plugins": [], "cache_time": 0, "metrics": []}
        self.update_query_with_single_object = {"name": "", "datapoints": [], "tags": {}}
        dict_metrics = {"tags": {}, "name": "",
                        "group_by": []}
        dict_metrics["group_by"].append({"name": "tag",
                                         "tags": ["category_1", "category_2", "category_3", "category_4", "category_5",
                                                  "category_6"]})
        self.query["metrics"].append(dict_metrics)
        self.response = []

    # method to set metrics tags
    # value must be dict
    def set_metrics_tags(self, value):
        self.query["metrics"][0]["tags"] = value

    # method to set metrics name
    # value must be string
    def set_metrics_name(self, value):
        self.query["metrics"][0]["name"] = value

    # method to set metrics aggregators
    # value must be list of dicts
    def set_metrics_aggregators(self, value):
        self.query["metrics"][0]["aggregators"] = value

    # method to set absolute or relative start and end date
    # assign True to is_absolute if having absolute start and end date else False if having relative start and end date
    def set_start_end_date(self, is_absolute, start='', end=''):
        if is_absolute:
            if start != "":
                self.query["start_absolute"] = start
            if end != "":
                self.query["end_absolute"] = end
            self.delete_start_end_date_from_query("relative")
        else:
            if start != "":
                self.query["start_relative"] = start
            if end != "":
                self.query["end_relative"] = end
            self.delete_start_end_date_from_query("absolute")

    def set_relative_time(self, start_time, unit="minutes"):
        self.query["start_relative"] = {"value": start_time, "unit": unit}

    # method to return the data from kairosdb based on query
    def get_kairos_data(self):
        self.response = requests.post(kairos_server + "/api/v1/datapoints/query", data=json.dumps(self.query))
        return self.response.json()["queries"][0]["results"] if "queries" in self.response.json() else [
            {'tags': {}, 'name': 'ilens.live_data.raw', 'values': []}]

    # method to delete absolute or relative start and end date from query
    # type - absolute or relative
    def delete_start_end_date_from_query(self, type):
        if "start_" + type in self.query:
            del self.query["start_" + type]
        if "end_" + type in self.query:
            del self.query["end_" + type]

    # method to update the data to kairosdb based on query
    def update_kairos_data(self, is_multi_data_update=False, update_query_with_array_of_object=[]):
        return requests.post(kairos_server + "/api/v1/datapoints", data=json.dumps(
            self.update_query_with_single_object if is_multi_data_update is False else update_query_with_array_of_object))

    # method to set update query name
    # value must be string
    def set_update_query_name(self, value):
        self.update_query_with_single_object["name"] = value

    # method to set update query datapoints
    # value must be array of arrays
    def set_update_query_datapoints(self, value):
        self.update_query_with_single_object["datapoints"].append(value)

    # method to set update query tags
    # value must be object
    def set_update_query_tags(self, value):
        self.update_query_with_single_object["tags"] = value
