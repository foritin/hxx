from typing import Optional, List, Dict
import polars as pl

from extend.core.backtest_logger import BacktestLogger
from extend.data.data_source import MultiDataSource
from extend.core.backtest_config import TradeConfig, SingleSymbolConfig, SourceType
from extend.utils import PathTools
from extend.data.data_loader import load_local_data


class BacktestDataManager:

    def __init__(self, trade_config: TradeConfig, logger: Optional[BacktestLogger] = None):
        self.logger: Optional[BacktestLogger] = logger
        self.trade_config: TradeConfig = trade_config
        self.multi_data_source = MultiDataSource()
        self.data_dict = {}

    def log(self, message):
        if self.logger:
            self.logger.log_message(message)
        else:
            print(message)

    def build_datasources(self, symbol_configs: List[SingleSymbolConfig]) -> MultiDataSource:
        """Build data sources for backtest"""
        local_data_path = PathTools.get_data_path()

        for symbol_config in symbol_configs:
            symbol = symbol_config.symbol
            symbol_type = symbol_config.symbol_type
            start_date = symbol_config.start_date
            end_date = symbol_config.end_date
            periods = symbol_config.periods
            source_type = symbol_config.source_type
            for period in periods:
                period_str = period.value
                if source_type == SourceType.CSV:
                    file_path = PathTools.combine_path(local_data_path, symbol_type, symbol, period_str, f"{symbol}.csv")
                    df = load_local_data(file_path, start_date=start_date, end_date=end_date)
                    self.multi_data_source.add_data_source(symbol=symbol, kline_period=period_str, data=df)
        # 为每个datasource分配资金
        self.multi_data_source.allocate_fund(self.trade_config)
        return self.multi_data_source

    def align_data(self, align=True, fill_method="ffill"):

        if align and len(self.multi_data_source) > 1:
            self.log("\nAligning data...")
            self.multi_data_source.align_data(align_index=True, fill_method=fill_method)

        return self.multi_data_source

    def get_data_source(self):
        return self.multi_data_source
