import polars as pl
import numpy as np
from extend.data.data_source import MultiDataSource


class BacktestResultCalculator:

    def __init__(self, logger=None):
        self.logger = logger
        self.results = {}

    def log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def calculate_performance(self, results):
        pass

    def calculate_results(self, multi_data_source: MultiDataSource):
        results = {}

        for i, ds in enumerate(multi_data_source.data_sources):
            trades = ds.trades

            if not trades:
                self.log(f"datasource {i} has no trades")
                continue
            trade_df = pl.DataFrame(trades)
            days_count = ds.data.select((pl.col("datetime").max() - pl.col("datetime").min()).dt.total_days().alias("total_days")).item()
            win_trades = trade_df.filter(pl.col("pnl") > 0).shape[0]
            loss_trades = trade_df.filter(pl.col("pnl") < 0).shape[0]
            win_rate = win_trades / (win_trades + loss_trades) if (win_trades + loss_trades) > 0 else 0
            
            total_pnl = trade_df.select(pl.col("pnl").sum()).item()
            
            print()
