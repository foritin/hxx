from typing import Optional
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
    

    def log(self, message):
        if self.logger:
            self.logger.log_message(message)
        else:
            print(message)
    

    def fetch_data(self, symbols_and_periods, symbol_configs: dict, base_config: dict):

        self.log("\nFetching data...")

        data_dict = {}
        local_data_path = PathTools.get_data_path()
        for symbol, config in symbol_configs.items():
            # 构造数据获取参数
            data_params = {
                "symbol": symbol,
                "data_type": config.get("data_type"),
                "start_date": config.get("start_date"),
                "end_date": config.get("end_data"),
                "use_cache": base_config.get("use_cache", True),
                "save_data": base_config.get("save_data", True),
            }
            # 获取数据
            for period_config in config.get("periods", []):
                kline_period = period_config.get("kline_period", "1h")
                if config.get("source_type") == "csv":
                    file_path = PathTools.combine_path(local_data_path, data_params["data_type"], symbol, kline_period, f"{symbol}.csv")
                    df = load_local_data(file_path, start_date=data_params["start_date"], end_date=data_params["end_data"])
                    key = f"{symbol}_{kline_period}"
                    data_dict[key] = df
            
            self.data_dict = data_dict
            return data_dict
    

    def create_data_sources(self, symbols_and_periods, data_dict):
        for i, item in enumerate(symbols_and_periods):
            symbol = item["symbol"]
            kline_period = item["kline_period"]

            key = f"{symbol}_{kline_period}"
            for key in data_dict:
                data = data_dict[key]
                self.multi_data_source.add_data_source(symbol, kline_period, data)
                self.log(f"Added data source for {symbol} {kline_period}")
        return self.multi_data_source
    
    def align_data(self, align=True, fill_method="ffill"):

        if align and len(self.multi_data_source) > 1:
            self.log("\nAligning data...")
            self.multi_data_source.align_data(align_index=True, fill_method=fill_method)
        
        return self.multi_data_source


    def get_data_source(self):
        return self.multi_data_source