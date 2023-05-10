import yaml


class AppConfig:
    def __init__(self):
        config_path = "config.yml"
        with open(config_path, 'r') as ymlFile:
            self.cfg = yaml.load(ymlFile, Loader=yaml.FullLoader)

    def get_mongo_host(self):
        return self.cfg['mongo_db']['mongo_connection']

    def get_mongo_uri(self, db_name):
        return self.cfg['mongo_db']['mongo_connection'] + '/' + db_name

    def get_kairos_host(self):
        return self.cfg['kairos_db']['url']

