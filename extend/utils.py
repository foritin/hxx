import yaml
from pathlib import Path


class BaseTools:

    @staticmethod
    def load_yaml_config(file_path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)


class PathTools:

    @classmethod
    def combine_path(cls, base, *args) -> Path:
        return Path(base).joinpath(*args)

    @classmethod
    def get_root_path(cls):
        return Path(__file__).parent.parent

    @classmethod
    def get_data_path(cls):
        return PathTools.get_root_path().joinpath("resource", "data")

    @classmethod
    def get_strategy_path(cls):
        return PathTools.get_root_path().joinpath("strategy")
    
    @classmethod
    def get_log_path(cls):
        return PathTools.get_root_path().joinpath("output", "logs")
        
