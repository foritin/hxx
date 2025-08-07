from pathlib import Path




class PathTools:


    @staticmethod
    def combine_path(base, *args):
        return Path(base).joinpath(*args)

    @staticmethod
    def get_data_path():
        return Path(__file__).parent.parent.joinpath("resource", "data")
    
    