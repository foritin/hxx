import itertools
import time
from multiprocessing import Pool, shared_memory
import numpy as np
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


class SharedMemoryDataFrame:
    """自定义DataFrame类，直接使用共享内存而不复制数据"""
    
    def __init__(self, data_dict, length):
        self.data_dict = data_dict
        self.length = length
        self.columns = list(data_dict.keys())
    
    def __len__(self):
        return self.length
    
    def row(self, index, named=True):
        """模拟polars DataFrame的row方法"""
        if named:
            return {col: self.data_dict[col][index] for col in self.columns}
        else:
            return [self.data_dict[col][index] for col in self.columns]
    
    def is_empty(self):
        return self.length == 0
    
    def __getitem__(self, key):
        """支持多种索引方式"""
        if isinstance(key, str):
            # 列访问: data["close"]
            return self.data_dict[key]
        elif isinstance(key, tuple) and len(key) == 2:
            # 位置访问: data[i, "close"]
            row_idx, col_name = key
            return self.data_dict[col_name][row_idx]
        elif isinstance(key, int):
            # 行访问: data[i] (返回该行的Series)
            return {col: self.data_dict[col][key] for col in self.columns}
        else:
            raise ValueError(f"Unsupported key type: {type(key)}")
    
    def to_pandas(self):
        """转换为pandas DataFrame（如果需要）"""
        import pandas as pd
        return pd.DataFrame(self.data_dict)
    
    def to_numpy(self):
        """转换为numpy数组"""
        import numpy as np
        return np.column_stack([self.data_dict[col] for col in self.columns])
    
    def slice(self, start, length):
        """切片操作，模拟polars的slice方法"""
        sliced_data = {}
        for col in self.columns:
            sliced_data[col] = self.data_dict[col][start:start+length]
        return SharedMemoryDataFrame(sliced_data, length)


class MultiSourceBacktester:

    def __init__(self, config_file_path: Union[str, Path]):

        yaml_config = BaseTools.load_yaml_config(config_file_path)
        self.strategy_config = StrategyConfig(**yaml_config)

        self.base_config = self.strategy_config.base_config
        self.trade_config = self.strategy_config.trade_config
        self.symbol_configs = self.strategy_config.symbol_configs
        self.optimization_params = self.strategy_config.optimization_params if hasattr(self.strategy_config, "optimization_params") else None

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

    def run_backtest(self, strategy_module_path):
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

        import importlib
        module_path, func_name = strategy_module_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        strategy_func_or_class = getattr(module, func_name)
        
        # 检查是否是策略类
        if hasattr(strategy_func_or_class, '__bases__') and strategy_func_or_class.__name__.endswith('Strategy'):
            # 这是一个策略类，需要实例化
            strategy_instance = strategy_func_or_class(strategy_params)
            strategy_func = strategy_instance.run
        else:
            strategy_func = strategy_func_or_class

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
        result_dict = self.result_calculator.generate_report(multi_data_source)

        self._last_multi_data_source = multi_data_source

        return result_dict


    def run_backtest_optimization(self, strategy_module_path, max_processes: int =4):
        optimization_params = self.strategy_config.optimization_params.to_dict()
        params_names = list(optimization_params.keys())
        params_values = list(optimization_params.values())
        param_combinations = [
            dict(zip(params_names, combo)) 
            for combo in itertools.product(*params_values)
        ]
        self.logger.log_message(f"🎯 生成 {len(param_combinations)} 个参数组合")
    
        # 2. 准备数据（只加载一次）
        multi_data_source = self.data_manager.build_datasources(
            symbol_configs=self.symbol_configs
        ).align_data(
            align_index=self.base_config.align_data,
            fill_method=self.base_config.fill_method,
        )
    

        # 3. 创建共享内存
        shared_data = {}
        shared_memories = {}
        
        for ds in multi_data_source.data_sources:
            if not ds.data.is_empty():
                ds_key = f"{ds.symbol}_{ds.kline_period}"
                shared_data[ds_key] = {}
                shared_memories[ds_key] = {}
                
                # 共享OHLCV数据和datetime
                for col in ['datetime', 'open', 'high', 'low', 'close', 'volume']:
                    if col in ds.data.columns:
                        data_array = ds.data[col].to_numpy()
                        
                        # 创建共享内存
                        shm = shared_memory.SharedMemory(create=True, size=data_array.nbytes)
                        shared_array = np.ndarray(data_array.shape, dtype=data_array.dtype, buffer=shm.buf)
                        shared_array[:] = data_array
                        
                        shared_memories[ds_key][col] = shm
                        shared_data[ds_key][col] = {
                            'name': shm.name,
                            'shape': data_array.shape,
                            'dtype': str(data_array.dtype)
                        }
        
        self.logger.log_message("📦 共享内存创建完成")
        
        # 4. 设置优化模式
        original_debug = self.base_config.debug
        self.base_config.debug = False  # 优化期间关闭详细日志
        self._in_optimization_mode = True
        try:
            # 5. 并行执行
            start_time = time.time()
            
            tasks = [
                (shared_data, params, self.strategy_config.strategy_params.model_dump(), strategy_module_path)
                for params in param_combinations
            ]
            
            # 限制每个worker最多处理20个任务后重启，更积极地控制内存
            with Pool(max_processes, maxtasksperchild=20) as pool:
                # 使用imap_unordered来显示实时进度
                results = []
                completed = 0
                total_tasks = len(tasks)
                
                self.logger.log_message(f"🚀 开始处理 {total_tasks} 个参数组合...")
                self.logger.log_message(f"⚙️  使用 {max_processes} 个进程，正在启动worker进程...")
                
                # 设置较小的chunksize并限制每个worker的任务数量（防止内存累积）
                # maxtasksperchild限制每个worker最多处理20个任务后重启
                worker_pids = set()  # 跟踪worker进程ID
                
                for result in pool.imap_unordered(self.optimization_worker, tasks, chunksize=1):
                    results.append(result)
                    completed += 1
                    
                    # 第一个任务完成时的特殊提示
                    if completed == 1:
                        self.logger.log_message("✅ 第一个任务完成，worker进程启动成功！")
                    
                    # 优化后的进度报告：每50个任务或每1%进度显示一次
                    if completed % 50 == 0 or completed % max(1, total_tasks // 100) == 0:
                        progress = (completed / total_tasks) * 100
                        valid_count = len([r for r in results if r is not None])
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total_tasks - completed) / rate if rate > 0 else 0
                        
                        self.logger.log_message(
                            f"📊 进度: {completed}/{total_tasks} ({progress:.1f}%) | "
                            f"有效结果: {valid_count} | 成功率: {(valid_count/completed)*100:.1f}% | "
                            f"速度: {rate:.1f}任务/秒 | 预计剩余: {eta/60:.1f}分钟"
                        )
            
            # 过滤有效结果
            valid_results = [r for r in results if r is not None]
            
            elapsed_time = time.time() - start_time
            self.logger.log_message(
                f"✅ 优化完成: {len(valid_results)}/{len(param_combinations)} 成功, "
                f"耗时 {elapsed_time:.2f}秒"
            )
            
            # 6. 结果分析
            if valid_results:
                self._analyze_results(valid_results)
            
            return valid_results
            
        finally:
            # 7. 清理
            self.base_config.debug = original_debug
            self._in_optimization_mode = False
            
            # 清理共享内存
            for ds_memories in shared_memories.values():
                for shm in ds_memories.values():
                    try:
                        shm.close()
                        shm.unlink()
                    except:
                        pass



    def optimization_worker(self, args):
        shared_data, test_params, base_params, strategy_module_path = args
        
        # 简单的worker开始日志
        import os
        worker_id = os.getpid()
        
        # 监控内存使用（验证共享内存效果）
        process = None
        start_memory = 0
        try:
            import psutil
            process = psutil.Process(worker_id)
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 只在内存异常时警告
            if start_memory > 1000:  # 超过1GB说明有问题
                print(f"🚨 WORKER_{worker_id}: 新worker内存异常! 开始内存={start_memory:.1f}MB, 参数={test_params}")
        except:
            pass  # 静默处理内存监控失败
        
        try:
            import polars as pl
            from extend.api.strategy_api import create_strategy_api
            from extend.data.data_source import DataSource, MultiDataSource
            
            # 创建MultiDataSource，使用真正的共享内存（无复制）
            multi_data_source = MultiDataSource()
            shared_memories_in_worker = {}  # 保存共享内存引用，防止被垃圾回收
            
            for ds_key, data_info in shared_data.items():
                symbol, period = ds_key.split('_', 1)
                
                # 重建数据数组 - 关键：不调用.copy()
                data_dict = {}
                shared_memories_in_worker[ds_key] = {}
                
                for col, info in data_info.items():
                    existing_shm = shared_memory.SharedMemory(name=info['name'])
                    # 直接使用共享内存，不复制！
                    data_array = np.ndarray(
                        info['shape'], 
                        dtype=np.dtype(info['dtype']), 
                        buffer=existing_shm.buf
                    )
                    # 保存共享内存引用，防止被垃圾回收
                    shared_memories_in_worker[ds_key][col] = existing_shm
                    data_dict[col] = data_array
                
                # 尝试使用polars，如果失败则使用SharedMemoryDataFrame
                try:
                    # 尝试使用polars DataFrame（可能会复制数据，但功能完整）
                    df = pl.DataFrame(data_dict)
                    
                    # 只在异常情况下打印
                    if len(df) == 0:
                        print(f"⚠️ WORKER_{worker_id}: polars DataFrame为空！")
                    elif "close" not in df.columns:
                        print(f"⚠️ WORKER_{worker_id}: polars DataFrame缺少close列！")
                        
                except Exception as e:
                    # 如果polars失败，使用SharedMemoryDataFrame
                    df = SharedMemoryDataFrame(data_dict, len(data_dict['datetime']))
                    print(f"⚠️ WORKER_{worker_id}: polars失败，使用SharedMemoryDataFrame: {e}")
                    
                    # 只在异常情况下验证
                    if len(df) == 0:
                        print(f"⚠️ WORKER_{worker_id}: SharedMemoryDataFrame为空！")
                
                # 使用add_data_source方法来正确设置double_dict
                multi_data_source.add_data_source(symbol, period, df)
            
            # 2. 合并参数
            strategy_params = {**base_params, **test_params}
            
            # 创建一个临时的trade_config用于资金分配
            from extend.core.backtest_config import TradeConfig
            fund_amount = strategy_params.get('Fund', 100000)
            temp_trade_config = TradeConfig(
                total_capital=fund_amount,  # 使用策略参数中的资金
                commission=0.00025,  # 与yaml配置一致
                slippage=0.0,
                total_margin_rate=0.3  # 与yaml配置一致
            )
            multi_data_source.allocate_fund(temp_trade_config)
            
            # 3. 创建API并运行策略
            context = {
                "data": multi_data_source,
                "log": lambda x: None,  # 优化模式下完全静默
                "params": strategy_params
            }
            api = create_strategy_api(context)
            
            # 4. 动态导入策略函数或类
            import importlib
            module_path, func_name = strategy_module_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            strategy_func_or_class = getattr(module, func_name)
            
            # 检查是否是策略类
            if hasattr(strategy_func_or_class, '__bases__') and strategy_func_or_class.__name__.endswith('Strategy'):
                # 这是一个策略类，需要实例化
                strategy_instance = strategy_func_or_class(strategy_params)
                
                # 验证关键参数，避免后续计算错误
                M = strategy_params.get('M', 60)
                S = strategy_params.get('S', 10)
                if S != 0:
                    period_1 = max(1, int(M/S))
                    if period_1 <= 0:
                        # 参数无效，直接返回None跳过此参数组合（静默跳过）
                        return None
                
                strategy_func = strategy_instance.run
            else:
                # 这是一个普通函数
                strategy_func = strategy_func_or_class
            
            # 5. 简化的回测循环
            min_length = min([len(ds.data) for ds in multi_data_source.data_sources])
            for i in range(min_length):
                for ds in multi_data_source.data_sources:
                    ds.current_idx = i
                    row = ds.data.row(i, named=True)
                    ds.current_price = row["close"]
                    ds.current_datetime = row["datetime"]
                    ds._process_pending_orders(log_callback=None)
                
                strategy_func(api)
            
            # 6. 使用专业的结果计算器计算详细结果
            from extend.core.backtest_result import BacktestResultCalculator
            
            result_calculator = BacktestResultCalculator(logger=None)  # 优化期间不打印日志
            results = result_calculator.generate_report(multi_data_source)
            performance = results.get('_overall_performance', {})
            
            # 提取关键指标用于优化
            if performance and results:
                net_value = performance.get('total_final_equity', 0) / performance.get('total_initial_capital', 1) if performance.get('total_initial_capital', 1) > 0 else 1
                total_trades = performance.get('total_trades', 0)
                total_net_profit = performance.get('total_net_profit', 0)
                sharpe_ratio = performance.get('weighted_sharpe_ratio', 0)
                max_drawdown_pct = performance.get('weighted_max_drawdown_pct', 0)
                win_rate = performance.get('overall_win_rate', 0)
                
                # 提取月收益率指标
                monthly_return_avg = performance.get('overall_monthly_return_avg', 0)
                monthly_return_std = performance.get('overall_monthly_return_std', 0)
                monthly_stability_score = performance.get('overall_monthly_stability', 0)
                
                # 计算年化收益率（复利方式，使用365天）
                total_initial_capital = performance.get('total_initial_capital', 1)
                total_final_equity = performance.get('total_final_equity', 0)
                if results:
                    # 获取交易日总数（从第一个数据源）
                    first_ds_result = next(iter([v for k, v in results.items() if k != '_overall_performance']), {})
                    trading_days = first_ds_result.get('trading_days', 365)
                    trading_years = trading_days / 365 if trading_days > 0 else 1
                    # 复利年化收益率：((期末/期初)^(1/年数) - 1) * 100
                    annual_return = ((total_final_equity / total_initial_capital) ** (1/trading_years) - 1) * 100 if trading_years > 0 and total_initial_capital > 0 else 0
                else:
                    annual_return = 0
                    trading_days = 0
            else:
                # 如果没有交易，返回基础值
                net_value = 1.0
                total_trades = 0
                total_net_profit = 0
                sharpe_ratio = 0
                max_drawdown_pct = 0
                win_rate = 0
                monthly_return_avg = 0
                monthly_return_std = 0
                monthly_stability_score = 0
                annual_return = 0
                trading_days = 0
            
            result = {
                'params': test_params,
                'net_value': net_value,
                'annual_return': annual_return,
                'trading_days': trading_days,
                'total_trades': total_trades,
                'total_net_profit': total_net_profit,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown_pct,
                'win_rate': win_rate,
                'monthly_return_avg': monthly_return_avg,
                'monthly_return_std': monthly_return_std,
                'monthly_stability_score': monthly_stability_score,
                'detailed_results': results,  # 包含详细的每个数据源的结果
                'performance': performance     # 包含整体性能指标
            }
            
            # 记录完成时的内存使用（只在内存增长异常时打印）
            if process:
                try:
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_increase = end_memory - start_memory
                    # 只在内存增长过大时打印警告
                    if memory_increase > 50:  # 超过50MB增长时才警告
                        print(f"⚠️ WORKER_{worker_id}: 内存增长异常! 净值={net_value:.4f}, 交易={total_trades}, 内存增长={memory_increase:.1f}MB")
                except:
                    pass  # 静默处理内存监控失败
            return result
            
        except Exception as e:
            # 只在严重错误时打印，减少噪音
            if "Memory" in str(e) or "ImportError" in str(e) or "NameError" in str(e):
                print(f"❌ WORKER_{worker_id}: 严重错误 - {str(e)}")
            return None
        
        finally:
            # 清理共享内存引用（但不unlink，由主进程负责）
            if 'shared_memories_in_worker' in locals():
                for ds_memories in shared_memories_in_worker.values():
                    for shm in ds_memories.values():
                        try:
                            shm.close()  # 只关闭当前进程的连接
                        except:
                            pass
            
            # 强制清理大对象，防止内存泄漏
            try:
                # 清理策略相关对象
                if 'strategy_instance' in locals():
                    del strategy_instance
                if 'strategy_func' in locals():
                    del strategy_func
                if 'api' in locals():
                    del api
                    
                # 清理数据源对象
                if 'multi_data_source' in locals():
                    # 尝试清理DataSource中的数据
                    for ds in multi_data_source.data_sources:
                        if hasattr(ds, 'data'):
                            ds.data = None
                        if hasattr(ds, 'trades'):
                            ds.trades = []
                    del multi_data_source
                
                # 清理结果对象和DataFrame
                if 'results' in locals():
                    del results
                if 'data_dict' in locals():
                    del data_dict
                
                # 清理任何pandas DataFrame缓存
                try:
                    import pandas as pd
                    # 清理pandas内部缓存
                    if hasattr(pd, 'core') and hasattr(pd.core, 'common') and hasattr(pd.core.common, '_values_from_object'):
                        # 这是pandas的内部清理，但可能不总是有效
                        pass
                except:
                    pass
                
                # 强制垃圾回收（静默执行）
                import gc
                gc.collect()
                
            except Exception as cleanup_error:
                # 只在严重的清理错误时打印
                if "Memory" in str(cleanup_error) or "Access" in str(cleanup_error):
                    print(f"⚠️ WORKER_{worker_id}: 严重清理错误: {cleanup_error}")
                pass  # 清理失败不影响主流程


    def _analyze_results(self, results):
        """分析优化结果（使用BacktestResultCalculator进行计算）"""
        if not results:
            return
        
        # 使用BacktestResultCalculator进行增强指标计算
        from extend.core.backtest_result import BacktestResultCalculator
        calculator = BacktestResultCalculator(logger=None)  # 临时计算器，不需要日志
        
        # 增强结果数据
        enhanced_results = calculator.enhance_results_with_metrics(results)
        
        # 计算统计信息
        stats = calculator.calculate_statistics(enhanced_results)
        
        # 开始输出分析结果
        self.logger.log_message("=" * 70)
        self.logger.log_message("🏆 参数优化结果分析")
        self.logger.log_message("=" * 70)
        
        # 1. 按年化收益率排序（主要指标）
        sorted_by_annual_return = calculator.get_sorted_results(enhanced_results, 'annual_return', 5)
        self._log_ranking_results("📈 按年化收益率排序", sorted_by_annual_return, 
                                  lambda r: f"净值={r['net_value']:.4f} | 夏普={r.get('sharpe_ratio', 0):.2f} | 胜率={r.get('win_rate', 0):.2%} | 回撤={r.get('max_drawdown_pct', 0):.2%} | 交易={r['total_trades']} | 交易日={r.get('trading_days', 0)} | 月均收益={r.get('monthly_return_avg', 0):.2f}% | 月收益标差={r.get('monthly_return_std', 0):.2f}% | 月稳定性={r.get('monthly_stability_score', 0):.2f}")
        
        # 2. 按夏普比率排序
        sorted_by_sharpe = calculator.get_sorted_results(enhanced_results, 'sharpe_ratio', 5)
        if sorted_by_sharpe:
            self._log_ranking_results("📊 按夏普比率排序", sorted_by_sharpe,
                                      lambda r: f"净值={r['net_value']:.4f} | 夏普={r.get('sharpe_ratio', 0):.2f} | 胜率={r.get('win_rate', 0):.2%} | 回撤={r.get('max_drawdown_pct', 0):.2%} | 交易={r['total_trades']} | 交易日={r.get('trading_days', 0)} | 月均收益={r.get('monthly_return_avg', 0):.2f}% | 月收益标差={r.get('monthly_return_std', 0):.2f}% | 月稳定性={r.get('monthly_stability_score', 0):.2f}")
        
        # 3. 按最大回撤排序（越小越好）
        sorted_by_drawdown = calculator.get_sorted_results(enhanced_results, 'max_drawdown_pct', 5)
        if sorted_by_drawdown:
            self._log_ranking_results("📉 按最大回撤排序（最小回撤）", sorted_by_drawdown,
                                      lambda r: f"净值={r['net_value']:.4f} | 夏普={r.get('sharpe_ratio', 0):.2f} | 胜率={r.get('win_rate', 0):.2%} | 回撤={r.get('max_drawdown_pct', 0):.2%} | 交易={r['total_trades']} | 交易日={r.get('trading_days', 0)} | 月均收益={r.get('monthly_return_avg', 0):.2f}% | 月收益标差={r.get('monthly_return_std', 0):.2f}% | 月稳定性={r.get('monthly_stability_score', 0):.2f}")
        
        # 4. 按胜率排序
        sorted_by_winrate = calculator.get_sorted_results(enhanced_results, 'win_rate', 5)
        if sorted_by_winrate:
            self._log_ranking_results("🎯 按胜率排序", sorted_by_winrate,
                                      lambda r: f"净值={r['net_value']:.4f} | 夏普={r.get('sharpe_ratio', 0):.2f} | 胜率={r.get('win_rate', 0):.2%} | 回撤={r.get('max_drawdown_pct', 0):.2%} | 交易={r['total_trades']} | 交易日={r.get('trading_days', 0)} | 月均收益={r.get('monthly_return_avg', 0):.2f}% | 月收益标差={r.get('monthly_return_std', 0):.2f}% | 月稳定性={r.get('monthly_stability_score', 0):.2f}")
        
        # 5. 综合评分排序
        sorted_by_composite = calculator.get_sorted_results(enhanced_results, 'composite_score', 5)
        self._log_ranking_results("🏆 综合评分排序", sorted_by_composite,
                                  lambda r: f"综合评分={r['composite_score']:.2f} | 净值={r['net_value']:.4f} | 夏普={r.get('sharpe_ratio', 0):.2f} | 胜率={r.get('win_rate', 0):.2%} | 回撤={r.get('max_drawdown_pct', 0):.2%} | 交易={r['total_trades']} | 交易日={r.get('trading_days', 0)} | 月均收益={r.get('monthly_return_avg', 0):.2f}% | 月收益标差={r.get('monthly_return_std', 0):.2f}% | 月稳定性={r.get('monthly_stability_score', 0):.2f}")
        
        # 最优参数详细信息（按年化收益率）
        best = sorted_by_annual_return[0] if sorted_by_annual_return else enhanced_results[0]
        self._log_best_result_details(best)
        
        # 统计信息
        self._log_optimization_statistics(stats)
        
        self.logger.log_message("=" * 70)
    
    def _log_ranking_results(self, title, results, format_func):
        """输出排名结果的通用方法"""
        self.logger.log_message(f"\n{title} Top 5:")
        self.logger.log_message("-" * 70)
        for i, result in enumerate(results, 1):
            params_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
            metrics_str = format_func(result)
            self.logger.log_message(f"{i}. {metrics_str} | {params_str}")
    
    def _log_best_result_details(self, best):
        """输出最优结果详细信息"""
        self.logger.log_message(f"\n🥇 最优参数组合（按年化收益率）:")
        self.logger.log_message("-" * 40)
        for k, v in best['params'].items():
            self.logger.log_message(f"  {k}: {v}")
        
        self.logger.log_message(f"\n📋 最优结果详细指标:")
        self.logger.log_message(f"  净值: {best['net_value']:.4f}")
        self.logger.log_message(f"  年化收益率: {best.get('annual_return', 0):.2f}%")
        self.logger.log_message(f"  夏普比率: {best.get('sharpe_ratio', 0):.2f}")
        self.logger.log_message(f"  胜率: {best.get('win_rate', 0):.2%}")
        self.logger.log_message(f"  最大回撤: {best.get('max_drawdown_pct', 0):.2%}")
        self.logger.log_message(f"  交易次数: {best['total_trades']}")
        self.logger.log_message(f"  回测交易日: {best.get('trading_days', 0)}")
        self.logger.log_message(f"  月平均收益: {best.get('monthly_return_avg', 0):.2f}%")
        self.logger.log_message(f"  月收益标准差: {best.get('monthly_return_std', 0):.2f}%")
        self.logger.log_message(f"  月收益稳定性: {best.get('monthly_stability_score', 0):.2f}")
        self.logger.log_message(f"  净利润: {best.get('total_net_profit', 0):.2f}")
    
    def _log_optimization_statistics(self, stats):
        """输出优化统计信息"""
        self.logger.log_message(f"\n📊 优化统计:")
        self.logger.log_message(f"  参数组合总数: {stats['total_combinations']}")
        self.logger.log_message(f"  有效结果数: {stats['valid_results_count']}")
        self.logger.log_message(f"  平均净值: {stats['avg_net_value']:.4f}")
        
        if 'avg_annual_return' in stats:
            self.logger.log_message(f"  平均年化收益率: {stats['avg_annual_return']:.2f}%")
        
        if 'avg_sharpe_ratio' in stats:
            self.logger.log_message(f"  平均夏普比率: {stats['avg_sharpe_ratio']:.2f}")
        
        if 'avg_win_rate' in stats:
            self.logger.log_message(f"  平均胜率: {stats['avg_win_rate']:.2%}")
        
        # 月收益率统计
        if 'avg_monthly_return' in stats:
            self.logger.log_message(f"  平均月收益: {stats['avg_monthly_return']:.2f}%")
            self.logger.log_message(f"  平均月收益标准差: {stats['avg_monthly_std']:.2f}%")
            self.logger.log_message(f"  平均月收益稳定性: {stats['avg_monthly_stability']:.2f}")

