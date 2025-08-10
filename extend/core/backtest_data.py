from typing import Optional, Dict
import polars as pl

from extend.core.backtest_logger import BacktestLogger
from extend.data.data_source import MultiDataSource
from extend.utils import PathTools
from extend.data.data_loader import load_local_data


class BacktestDataManager:


    def __init__(self, logger=None):
        self.logger: Optional[BacktestLogger] = logger
        self.multi_data_source = MultiDataSource()
        self.data_dict = {}
    


    def initialize_trade_config(self, trade_config: Dict):
        self.multi_data_source.initialize_trade_config(trade_config=trade_config)
        return self


    def log(self, message):
        if self.logger:
            self.logger.log_message(message)
        else:
            print(message)
    

    def fetch_data(self, symbol_configs: dict, use_cache=True, save_data=True):

        self.log("\nFetching data...")

        data_dict = {}
        local_data_path = PathTools.get_data_path()
        for symbol, config in symbol_configs.items():
            # 构造数据获取参数
            data_params = {
                "symbol": symbol,
                "data_type": config.get("data_type"),
                "start_date": config.get("start_date"),
                "end_date": config.get("end_date"),
                "use_cache": use_cache,
                "save_data": save_data
            }
            # 获取数据
            
            kline_interval = config.get("interval")
            if config.get("source_type") == "csv":
                file_path = PathTools.combine_path(local_data_path, data_params["data_type"], symbol, kline_interval, f"{symbol}.csv")
                df = load_local_data(file_path, start_date=data_params["start_date"], end_date=data_params["end_date"])
                key = f"{symbol}_{kline_interval}"
                data_dict[key] = df
            
            self.data_dict = data_dict
            return data_dict
    

    def create_data_sources(self, symbol_configs: dict, data_dict):
        for i, (symbol, config) in enumerate(symbol_configs.items()):
            
            kline_interval = config["interval"]

            key = f"{symbol}_{kline_interval}"
            for key in data_dict:
                data = data_dict[key]
                self.multi_data_source.add_data_source(symbol=symbol, kline_interval=kline_interval, data=data)
                self.log(f"Added data source for {symbol} {kline_interval}")
        return self.multi_data_source
    
    def align_data(self, align=True, fill_method="ffill"):

        if align and len(self.multi_data_source) > 1:
            self.log("\nAligning data...")
            self.multi_data_source.align_data(align_index=True, fill_method=fill_method)
        
        return self.multi_data_source


    def get_data_source(self):
        return self.multi_data_source