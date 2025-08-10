import time
from typing import Optional, Dict
from extend.core.backtest_logger import BacktestLogger
from extend.core.backtest_data import BacktestDataManager
from extend.api.strategy_api import StrategyAPI, create_strategy_api

class MultiSourceBacktester:

    def __init__(self, base_config: Dict):
        # 默认基础配置
        self.default_base_config = {
            # API数据参数
            'use_api_data': True,
            'use_cache': True,
            'save_data': True,

            'initial_capital': 1000000,
            'commission': 0.0003,
            'margin_rate': 0.1,
            'contract_multiplier': 10,
            
            # 是否获取多个品种和周期的数据
            'fetch_multiple': True,
            
            # 是否对齐数据
            'align_data': True,
            'fill_method': 'ffill',
            
            # 调试模式
            'debug': False
        }
        
        # 使用传入的基础配置或默认配置
        self.default_base_config.update(base_config)
        self.base_config = self.default_base_config.copy()
        
        
        
        # 品种特定配置字典
        self.symbol_configs = {}
        
        # 回测结果
        self.results = {}
        
        # 日志管理器
        self.debug_mode = self.base_config.get('debug', False)
        self.logger = BacktestLogger(debug_mode=self.debug_mode)
        
        # 数据管理器
        self.data_manager = BacktestDataManager(self.logger)\
            .initialize_trade_config({
                "initial_captial": self.base_config['initial_capital'],
                "commission": self.base_config['commission'],
                "margin_rate": self.base_config['margin_rate'],
                "contract_multiplier": self.base_config['contract_multiplier']
                }
                )


        self._in_optimization_mode = False


    def add_symbol_config(self, symbol, config):
        """
        添加品种特定配置和周期
        
        Args:
            symbol (str): 品种代码，如'rb888'
            config (dict): 品种特定配置，可以包含以下额外参数：
                - periods (list): 周期配置列表，每个元素是一个字典，包含 'kline_period' 和 'adjust_type'
                  例如：[{'kline_period': '1h', 'adjust_type': '1'}, {'kline_period': 'D', 'adjust_type': '0'}]
                - kline_period (str): 单个K线周期，如'1h', 'D'（如果不提供periods）
                - adjust_type (str): 单个复权类型，'0'表示不复权，'1'表示后复权（如果不提供periods）
        """
        # 保存品种配置
        base_symbol_config = config.copy()
        
        self.symbol_configs[symbol] = base_symbol_config

        return self
    

    def set_optimization_mode(self, enable=True):
        self._in_optimization_mode = enable
        return self

    def run_backtest(self, strategy_func, strategy_params=None):

        if not self.symbol_configs:
            self.logger.log_message("No symbols and periods configured.Please add_symbol_config first")
            return
        
        self.logger.prepare_log_file(self.symbol_configs)
        if hasattr(self, "_data_preloaded") and self._data_preloaded: # type: ignore
            self.logger.log_message("Use preloaded data for backtest...")
            data_dict = self._preloaded_data  # type: ignore
            multi_data_source = self._preloaded_multi_data_source  # type: ignore
        else:
            self.logger.log_message("Fetching data for backtest...")
            data_dict = self.data_manager.fetch_data(self.symbol_configs, use_cache=self.base_config.get("use_cache", True), save_data=self.base_config.get("save_data", True))
            multi_data_source = self.data_manager.create_data_sources(self.symbol_configs, data_dict)

            # 数据对齐
            multi_data_source = self.data_manager.align_data(
                align=self.base_config.get("align_data", True),
                fill_method=self.base_config.get("fill_method", "ffill"),
            )
        
        if len(multi_data_source) == 0:
            self.logger.log_message("No data fetched for backtest.")
            return {}
        
        # 运行回测
        self.logger.log_message("Running backtest...")
        min_length = min([len(source.data) for source in multi_data_source.data_sources if not source.data.is_empty()])
        self.logger.log_message(f"Backtest data length: {min_length}")

        # 记录回测开始时间
        start_time = time.time()

        # 策略上下文
        context = {
            "data": multi_data_source,
            "log": self.logger.log_message,
            "params": strategy_params or {}
        }
        api = create_strategy_api(context)

        if hasattr(strategy_func, "initialize"):
            self.logger.log_message("Initializing strategy...")
            strategy_func.initialize(api)
        
        progress_last_update = time.time()
        progress_update_interval = 0.5
        progress_bar_length = 50

        # loop strategy
        for i in range(min_length):
            for ds in multi_data_source.data_sources:
                if not ds.data.is_empty() and i < len(ds.data):
                    ds.current_idx = i
                    ds.current_price = ds.data["close"][i]
                    ds.current_datetime = ds.data["datetime"][i]

                    ds._process_pending_orders(log_callback=self.logger.log_message)

            if not self._in_optimization_mode:
                current_time = time.time()
                if current_time - progress_last_update >= progress_update_interval:
                    progress_last_update = current_time
                    progress = float(i + 1) / min_length
                    filled_length = int(progress_bar_length * progress)
                    bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
                    
                    # 添加每分钟处理的K线数
                    elapsed = current_time - start_time
                    if elapsed > 0:
                        bars_per_minute = (i + 1) / elapsed * 60
                        estimated_time = (min_length - i - 1) * elapsed / (i + 1)
                        
                        # 清空当前行并显示进度条
                        print(f"\r回测进度: |{bar}| {progress*100:.1f}% ({i+1}/{min_length}) [{bars_per_minute:.0f}K线/分钟] [剩余: {estimated_time:.1f}秒]", end='', flush=True)
            
            # 调试信息（仅在debug=True时显示详细日志）
            if self.debug_mode and i % 100 == 0:
                self.logger.log_message(f"处理第 {i}/{min_length} 条数据")
                for j, ds in enumerate(multi_data_source.data_sources):
                    if not ds.data.is_empty() and i < len(ds.data):
                        self.logger.log_message(f"数据源 #{j}: 时间={ds.current_datetime}, 价格={ds.current_price:.2f}, 持仓={ds.current_pos}")
            
            # 运行策略
            strategy_func(api)
        
        # 完成进度条（仅在非优化模式下显示）
        if not self._in_optimization_mode:
            print(f"\r回测进度: |{'█' * progress_bar_length}| 100.0% ({min_length}/{min_length}) [完成]", flush=True)
            print()  # 添加一个换行
        
        # 记录回测结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.log_message(f"回测完成，耗时: {elapsed_time:.2f}秒")