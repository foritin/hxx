import time
from typing import Union, Optional
from pathlib import Path
from extend.core.backtest_logger import BacktestLogger
from extend.core.backtest_data import BacktestDataManager
from extend.core.backtest_report import BacktestReportGenerator
from extend.core.backtest_result import BacktestResultCalculator
from extend.api.strategy_api import StrategyAPI, create_strategy_api
from extend.data.data_source import MultiDataSource
from extend.utils import BaseTools
from extend.core.backtest_config import StrategyConfig


class MultiSourceBacktester:

    def __init__(self, config_file_path: Union[str, Path]):

        yaml_config = BaseTools.load_yaml_config(config_file_path)
        self.strategy_config = StrategyConfig(**yaml_config)

        self.base_config = self.strategy_config.base_config
        self.trade_config = self.strategy_config.trade_config
        self.symbol_configs = self.strategy_config.symbol_configs

        self.initialize_flag = False
        self._last_multi_data_source: Optional[MultiDataSource] = None

        # 回测结果
        self.results = {}

        # 日志管理器
        self.logger: Optional[BacktestLogger] = None

        # 数据管理器
        self.data_manager: Optional[BacktestDataManager] = None

        # 回测结果计算器
        self.result_calculator: Optional[BacktestResultCalculator] = None

        # 报告生成器
        self.report_generator: Optional[BacktestReportGenerator] = None

        # 参数优化标志
        self._in_optimization_mode = False

    def initialize(self):

        # 初始化日志管理器
        self.logger = BacktestLogger(debug_mode=self.base_config.debug)
        # 初始化数据管理器
        self.data_manager = BacktestDataManager(logger=self.logger, trade_config=self.trade_config)
        self.report_generator = BacktestReportGenerator(logger=self.logger)
        self.result_calculator = BacktestResultCalculator(logger=self.logger)
        self.logger.log_message("Backtest engine initialized.")
        self.initialize_flag = True

    def run_backtest(self, strategy_func):
        assert self.logger and self.data_manager and self.result_calculator and self.report_generator, "Backtest engine not initialized."
        # 数据初始化
        self.logger.log_message("Fetching data for backtest...")
        multi_data_source = self.data_manager.build_datasources(symbol_configs=self.symbol_configs).align_data(
            align_index=self.base_config.align_data,
            fill_method=self.base_config.fill_method,
        )
        if len(multi_data_source) == 0:
            self.logger.log_message("No data fetched for backtest.")
            return {}
        self.logger.log_message("Running backtest...")
        min_length = min([len(source.data) for source in multi_data_source.data_sources if not source.data.is_empty()])
        self.logger.log_message(f"Backtest data length: {min_length}")

        # 记录回测开始时间
        start_time = time.time()

        # 策略上下文
        strategy_params = self.strategy_config.strategy_params.model_dump() if self.strategy_config.strategy_params else {}
        context = {"data": multi_data_source, "log": self.logger.log_message, "params": strategy_params}
        api = create_strategy_api(context)

        if hasattr(strategy_func, "initialize"):
            self.logger.log_message("Initializing strategy...")
            strategy_func.initialize(api)

        # loop strategy
        for i in range(min_length):
            for ds in multi_data_source.data_sources:
                if not ds.data.is_empty() and i < len(ds.data):
                    ds.current_idx = i
                    row = ds.data.row(i, named=True)
                    ds.current_price = row["close"]
                    ds.current_datetime = row["datetime"]
                    ds._process_pending_orders(log_callback=self.logger.log_message)
            # 显示进度条（仅在非优化模式下显示）
            # if not self._in_optimization_mode:
            #     current_time = time.time()
            #     if current_time - progress_last_update >= progress_update_interval:
            #         progress_last_update = current_time
            #         progress = float(i + 1) / min_length
            #         filled_length = int(progress_bar_length * progress)
            #         bar = "█" * filled_length + "-" * (progress_bar_length - filled_length)

            #         # 添加每分钟处理的K线数
            #         elapsed = current_time - start_time
            #         if elapsed > 0:
            #             bars_per_minute = (i + 1) / elapsed * 60
            #             estimated_time = (min_length - i - 1) * elapsed / (i + 1)

            #             # 清空当前行并显示进度条
            #             print(f"\r回测进度: |{bar}| {progress*100:.1f}% ({i+1}/{min_length}) [{bars_per_minute:.0f}K线/分钟] [剩余: {estimated_time:.1f}秒]", end="", flush=True)

            # 调试信息（仅在debug=True时显示详细日志）
            if self.base_config.debug and i % 100 == 0:
                self.logger.log_message(f"处理第 {i}/{min_length} 条数据")
                for j, ds in enumerate(multi_data_source.data_sources):
                    if not ds.data.is_empty() and i < len(ds.data):
                        self.logger.log_message(f"数据源 #{j}: 时间={ds.current_datetime}, 价格={ds.current_price:.2f}, 持仓={ds.current_pos}")

            # 运行策略
            strategy_func(api)
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.log_message(f"回测完成，耗时: {elapsed_time:.2f} 秒")
        # 计算回测结果
        results = self.result_calculator.calculate_results(multi_data_source)

        self._last_multi_data_source = multi_data_source

    # def run_backtest(self, strategy_func, strategy_params=None):

    #     debug_mode = self.base_config.get("debug", False)
    #     self.logger.set_debug_mode(debug_mode)

    #     if not self.symbols_and_periods:
    #         self.logger.log_message("No symbols and periods configured.Please add_symbol_config first")
    #         return

    #     self.logger.prepare_log_file(self.symbols_and_periods)
    #     if hasattr(self, "_data_preloaded") and self._data_preloaded:  # type: ignore
    #         self.logger.log_message("Use preloaded data for backtest...")
    #         data_dict = self._preloaded_data  # type: ignore
    #         multi_data_source = self._preloaded_multi_data_source  # type: ignore
    #     else:
    #         self.logger.log_message("Fetching data for backtest...")
    #         data_dict = self.data_manager.fetch_data(self.symbols_and_periods, self.symbol_configs, self.base_config)
    #         multi_data_source = self.data_manager.create_data_sources(self.symbols_and_periods, data_dict)

    #         # 数据对齐
    #         multi_data_source = self.data_manager.align_data(
    #             align=self.base_config.get("align_data", True),
    #             fill_method=self.base_config.get("fill_method", "ffill"),
    #         )

    #     if len(multi_data_source) == 0:
    #         self.logger.log_message("No data fetched for backtest.")
    #         return {}

    #     # 运行回测
    #     self.logger.log_message("Running backtest...")
    #     min_length = min([len(source.data) for source in multi_data_source.data_sources if not source.data.is_empty()])
    #     self.logger.log_message(f"Backtest data length: {min_length}")

    #     # 记录回测开始时间
    #     start_time = time.time()

    #     # 策略上下文
    #     context = {"data": multi_data_source, "log": self.logger.log_message, "params": strategy_params or {}}
    #     api = create_strategy_api(context)

    #     if hasattr(strategy_func, "initialize"):
    #         self.logger.log_message("Initializing strategy...")
    #         strategy_func.initialize(api)

    #     progress_last_update = time.time()
    #     progress_update_interval = 0.5
    #     progress_bar_length = 50

    #     # loop strategy
    #     for i in range(min_length):
    #         for ds in multi_data_source.data_sources:
    #             if not ds.data.is_empty() and i < len(ds.data):
    #                 ds.current_idx = i
    #                 row = ds.data.row(i, named=True)
    #                 ds.current_price = row["close"]
    #                 ds.current_datetime = row["datetime"]
    #                 ds._process_pending_orders(log_callback=self.logger.log_message)
    #         # 显示进度条（仅在非优化模式下显示）
    #         if not self._in_optimization_mode:
    #             current_time = time.time()
    #             if current_time - progress_last_update >= progress_update_interval:
    #                 progress_last_update = current_time
    #                 progress = float(i + 1) / min_length
    #                 filled_length = int(progress_bar_length * progress)
    #                 bar = "█" * filled_length + "-" * (progress_bar_length - filled_length)

    #                 # 添加每分钟处理的K线数
    #                 elapsed = current_time - start_time
    #                 if elapsed > 0:
    #                     bars_per_minute = (i + 1) / elapsed * 60
    #                     estimated_time = (min_length - i - 1) * elapsed / (i + 1)

    #                     # 清空当前行并显示进度条
    #                     print(f"\r回测进度: |{bar}| {progress*100:.1f}% ({i+1}/{min_length}) [{bars_per_minute:.0f}K线/分钟] [剩余: {estimated_time:.1f}秒]", end="", flush=True)

    #         # 调试信息（仅在debug=True时显示详细日志）
    #         if debug_mode and i % 100 == 0:
    #             self.logger.log_message(f"处理第 {i}/{min_length} 条数据")
    #             for j, ds in enumerate(multi_data_source.data_sources):
    #                 if not ds.data.is_empty() and i < len(ds.data):
    #                     self.logger.log_message(f"数据源 #{j}: 时间={ds.current_datetime}, 价格={ds.current_price:.2f}, 持仓={ds.current_pos}")

    #         # 运行策略
    #         strategy_func(api)
