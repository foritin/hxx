import yaml
from pathlib import Path


class BaseTools:

    @staticmethod
    def load_yaml_config(file_path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)


class PathTools:

    @staticmethod
    def combine_path(base, *args) -> Path:
        return Path(base).joinpath(*args)

    @staticmethod
    def get_root_path():
        return Path(__file__).parent.parent

    @staticmethod
    def get_data_path():
        return PathTools.get_root_path().joinpath("resource", "data")

    @staticmethod
    def get_strategy_path():
        return PathTools.get_root_path().joinpath("strategy")
