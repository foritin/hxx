import os
from pathlib import Path


class PathManager:

    @classmethod
    def get_root_path(cls):
        return Path(__file__).parent

    @classmethod
    def get_resource_path(cls):
        path = os.path.join(cls.get_root_path(), "resource")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @classmethod
    def get_stock_data_path(cls):
        path = os.path.join(cls.get_resource_path(), "data", "stock")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @classmethod
    def get_cn_stock_data_path(cls):
        path = os.path.join(cls.get_stock_data_path(), "cn")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @classmethod
    def get_fund_etf_data_path(cls):
        path = os.path.join(cls.get_resource_path(), "data", "fund_etf")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @classmethod
    def get_cn_fund_etf_data_path(cls):
        path = os.path.join(cls.get_fund_etf_data_path(), "cn")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @classmethod
    def get_configure_path(cls):
        path = os.path.join(cls.get_resource_path(), "configure")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @classmethod
    def get_alphalab_path(cls):
        path = os.path.join(cls.get_configure_path(), "alphalab")
        if not os.path.exists(path):
            os.makedirs(path)
        return path
