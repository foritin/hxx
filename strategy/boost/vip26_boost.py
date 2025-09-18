from extend.core.backtest_engine import MultiSourceBacktester
from extend.utils import PathTools

strategy_path = PathTools.get_strategy_path()
config_file_path = PathTools.combine_path(strategy_path, "trends", "vip26.yaml")

# 创建多数据源回测器
backtester = MultiSourceBacktester(config_file_path=config_file_path)

backtester.initialize()

# backtester.run_backtest(strategy_module_path="strategy.trends.vip26.Strategy")
results = backtester.run_backtest_optimization(strategy_module_path="strategy.trends.vip26.Strategy", max_processes=16)
# print()