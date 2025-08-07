import time
from extend.core.backtest_logger import BacktestLogger
from extend.core.backtest_data import BacktestDataManager
from extend.api.strategy_api import StrategyAPI, create_strategy_api

class MultiSourceBacktester:

    def __init__(self, base_config=None):
        # 默认基础配置
        self.default_base_config = {
            # API数据参数
            'use_api_data': True,
            'use_cache': True,
            'save_data': True,
            'data_type': "csv",
            
            # 是否获取多个品种和周期的数据
            'fetch_multiple': True,
            
            # 是否对齐数据
            'align_data': True,
            'fill_method': 'ffill',
            
            # 调试模式
            'debug': False
        }
        
        # 使用传入的基础配置或默认配置
        self.base_config = base_config or self.default_base_config.copy()
        
        # 默认品种配置
        self.default_symbol_config = {
            'initial_capital': 100000.0,
            'commission': 0.0003,
            'margin_rate': 0.1,
            'contract_multiplier': 10
        }
        
        # 品种特定配置字典
        self.symbol_configs = {}
        
        # 品种和周期列表
        self.symbols_and_periods = []
        
        # 回测结果
        self.results = {}
        
        # 日志管理器
        debug_mode = self.base_config.get('debug', False)
        self.logger = BacktestLogger(debug_mode=debug_mode)
        
        # 数据管理器
        self.data_manager = BacktestDataManager(self.logger)


    def set_base_config(self, config):
        self.base_config.update(config)
        debug_mode = self.base_config.get('debug', False)
        self.logger.set_debug_mode(debug_mode)
        return self

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
        self.symbol_configs[symbol] = config.copy()
        
        # 处理周期信息
        if 'periods' in config:
            # 如果提供了多个周期配置
            for period_config in config['periods']:
                self.symbols_and_periods.append({
                    "symbol": symbol,
                    "kline_period": period_config['kline_period'],
                    "adjust_type": period_config.get('adjust_type', '1')
                })
        elif 'kline_period' in config:
            # 如果只提供了单个周期
            self.symbols_and_periods.append({
                "symbol": symbol,
                "kline_period": config['kline_period'],
                "adjust_type": config.get('adjust_type', '1')
            })
        
        return self
    

    def run_backtest(self, strategy_func, strategy_params=None):

        debug_mode = self.base_config.get("debug", False)
        self.logger.set_debug_mode(debug_mode)

        if not self.symbols_and_periods:
            self.logger.log_message("No symbols and periods configured.Please add_symbol_config first")
            return
        
        self.logger.prepare_log_file(self.symbols_and_periods)
        if hasattr(self, "_data_preloaded") and self._data_preloaded: # type: ignore
            self.logger.log_message("Use preloaded data for backtest...")
            data_dict = self._preloaded_data  # type: ignore
            multi_data_source = self._preloaded_multi_data_source  # type: ignore
        else:
            self.logger.log_message("Fetching data for backtest...")
            data_dict = self.data_manager.fetch_data(self.symbols_and_periods, self.symbol_configs, self.base_config)
            multi_data_source = self.data_manager.create_data_sources(self.symbols_and_periods, data_dict)

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
        min_length = min([len(source) for source in multi_data_source.data_sources if not source.is_empty()])
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
        progress_updaate_interval = 0.5
        progress_bar_length = 50

        # loop strategy
        for i in range(min_length):
            for ds in multi_data_source.data_sources:
                if not ds.data.is_empty() and i < len(ds.data):
                    ds.current_idx = i
                    row = ds.data.row(i)
                    