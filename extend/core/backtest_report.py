import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from extend.utils import PathTools


class BacktestReportGenerator:

    def __init__(self, logger=None):
        self.logger = logger

    def log(self, message):
        if self.logger:
            self.logger.log_message(message)
        else:
            print(message)
    
    def _create_benchmark_returns(self, multi_data_source, start_date, end_date):
        """
        创建基线收益率（买入持有策略）
        使用第一个有效数据源的价格数据
        """
        try:
            # 找到第一个有效的数据源
            primary_ds = None
            for ds in multi_data_source.data_sources:
                if not ds.data.is_empty():
                    primary_ds = ds
                    break
            
            if primary_ds is None:
                self.log("警告：无法找到有效数据源创建基线")
                return None
            
            # 提取价格数据
            price_data = []
            for i in range(len(primary_ds.data)):
                row = primary_ds.data.row(i, named=True)
                price_data.append({
                    'datetime': pd.to_datetime(row['datetime']),
                    'close': float(row['close'])
                })
            
            # 转换为DataFrame
            price_df = pd.DataFrame(price_data)
            price_df.set_index('datetime', inplace=True)
            
            # 创建完整的日期范围
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # 重新索引并前向填充
            price_series = price_df['close'].reindex(date_range).ffill()
            
            # 计算买入持有收益率
            benchmark_returns = price_series.pct_change().fillna(0)
            
            self.log(f"基线收益率创建成功：{primary_ds.symbol}_{primary_ds.kline_period} 买入持有")
            return benchmark_returns
            
        except Exception as e:
            self.log(f"创建基线收益率时出错: {str(e)}")
            return None
    
    def _prepare_equity_curve_for_quantstats(self, multi_data_source):
        """
        准备quantstats需要的权益曲线数据
        将多数据源的交易记录合并成时间序列的权益曲线
        """
        try:
            all_trades = []
            total_initial_capital = 0
            
            # 收集所有数据源的交易记录和初始资金
            for ds in multi_data_source.data_sources:
                if ds.trades:
                    for trade in ds.trades:
                        trade_copy = trade.copy()
                        trade_copy['symbol'] = ds.symbol
                        trade_copy['period'] = ds.kline_period
                        all_trades.append(trade_copy)
                
                total_initial_capital += ds.capital
            
            if not all_trades:
                self.log("警告：没有找到交易记录，无法生成quantstats报告")
                return None, None
            
            # 转换为DataFrame并按时间排序
            trades_df = pd.DataFrame(all_trades)
            trades_df['datetime'] = pd.to_datetime(trades_df['datetime'])
            trades_df = trades_df.sort_values('datetime')
            
            # 计算累积盈亏和权益曲线
            trades_df['cumulative_pnl'] = (trades_df['pnl'] - trades_df['fee']).cumsum()
            trades_df['equity'] = total_initial_capital + trades_df['cumulative_pnl']
            
            # 创建完整的时间序列（包括无交易的日期）
            # 获取数据的完整时间范围
            start_date = None
            end_date = None
            
            for ds in multi_data_source.data_sources:
                if not ds.data.is_empty():
                    # 安全获取开始和结束时间
                    try:
                        # 统一使用row方法获取数据，兼容polars和SharedMemoryDataFrame
                        first_row = ds.data.row(0, named=True)
                        last_row = ds.data.row(len(ds.data)-1, named=True)
                        
                        ds_start = pd.to_datetime(first_row['datetime'])
                        ds_end = pd.to_datetime(last_row['datetime'])
                        
                        if start_date is None or ds_start < start_date:
                            start_date = ds_start
                        if end_date is None or ds_end > end_date:
                            end_date = ds_end
                            
                    except Exception as e:
                        self.log(f"获取数据源 {ds.symbol}_{ds.kline_period} 时间范围时出错: {e}")
                        continue
            
            if start_date is None or end_date is None:
                self.log("警告：无法确定数据时间范围")
                return None, None
            
            # 创建完整的日期范围
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # 创建权益曲线时间序列
            equity_series = pd.Series(index=date_range, dtype=float)
            equity_series.iloc[0] = total_initial_capital  # 设置初始权益
            
            # 填充交易日的权益值
            current_equity = total_initial_capital
            for _, trade in trades_df.iterrows():
                trade_date = trade['datetime'].normalize()  # 去除时间部分，只保留日期
                current_equity = trade['equity']
                if trade_date in equity_series.index:
                    equity_series[trade_date] = current_equity
            
            # 前向填充无交易日的权益值
            equity_series = equity_series.ffill()  # 使用新的方法替代fillna(method='ffill')
            equity_series = equity_series.fillna(total_initial_capital)  # 填充开始日期之前的值
            
            # 计算收益率
            returns = equity_series.pct_change().fillna(0)
            
            return returns, equity_series
            
        except Exception as e:
            self.log(f"准备quantstats数据时出错: {str(e)}")
            return None, None
    
    def generate_quantstats_report(self, multi_data_source, strategy_name="backtest", save_report=True, include_benchmark=True):
        """
        使用quantstats生成量化分析报告
        
        Args:
            multi_data_source: MultiDataSource对象，包含交易数据
            strategy_name: 策略名称，用于报告文件命名
            save_report: 是否保存报告到文件
            include_benchmark: 是否包含基线对比（买入持有策略），默认True
            
        Returns:
            dict: 包含报告路径和关键指标的字典
        """
        try:
            # 检查是否安装了quantstats
            try:
                # 首先配置matplotlib和警告过滤
                import warnings
                import os
                import logging
                
                # 抑制各种警告
                warnings.filterwarnings('ignore', category=UserWarning)
                warnings.filterwarnings('ignore', category=FutureWarning)
                warnings.filterwarnings('ignore', message='.*Font family.*not found.*')
                
                # 设置matplotlib日志级别
                logging.getLogger('matplotlib').setLevel(logging.ERROR)
                logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
                
                # 配置matplotlib
                import matplotlib
                matplotlib.use('Agg')  # 使用无GUI后端
                
                # 设置环境变量来抑制字体警告
                os.environ['MPLCONFIGDIR'] = '/tmp'
                
                import quantstats as qs
                
            except ImportError:
                self.log("错误：未安装quantstats库，请运行：pip install quantstats")
                return None
            
            self.log("开始生成quantstats分析报告...")
            
            # 准备数据
            returns, equity_curve = self._prepare_equity_curve_for_quantstats(multi_data_source)
            
            if returns is None or equity_curve is None:
                self.log("数据准备失败，无法生成quantstats报告")
                return None
            
            if returns.sum() == 0:
                self.log("警告：所有收益率为0，可能没有有效的交易数据")
                return None
            
            self.log(f"数据准备完成：时间范围 {returns.index[0]} 到 {returns.index[-1]}")
            self.log(f"总交易日数：{len(returns)}，非零收益日数：{(returns != 0).sum()}")
            
            # 准备基线数据（如果需要）
            benchmark_returns = None
            if include_benchmark:
                benchmark_returns = self._create_benchmark_returns(
                    multi_data_source, 
                    returns.index[0], 
                    returns.index[-1]
                )
                if benchmark_returns is not None:
                    # 确保基线和策略收益率有相同的索引
                    benchmark_returns = benchmark_returns.reindex(returns.index, fill_value=0)
                    self.log(f"基线数据准备完成：非零收益日数：{(benchmark_returns != 0).sum()}")
                else:
                    self.log("基线数据创建失败，将生成无基线对比的报告")
                    include_benchmark = False
            
            # 创建报告目录
            if save_report:
                results_path = PathTools.get_results_path()
                results_path.mkdir(parents=True, exist_ok=True)
                
                # 生成文件名（包含时间戳）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_filename = f"{strategy_name}_quantstats_report_{timestamp}.html"
                report_path = results_path / report_filename
                
                # 生成HTML报告
                self.log(f"正在生成HTML报告：{report_path}")
                if include_benchmark and benchmark_returns is not None:
                    # 生成包含基线对比的报告
                    qs.reports.html(
                        returns, 
                        benchmark=benchmark_returns,
                        output=str(report_path), 
                        title=f"{strategy_name} 量化分析报告（vs 买入持有）"
                    )
                    self.log("已生成包含基线对比的HTML报告")
                else:
                    # 生成无基线对比的报告
                    qs.reports.html(returns, output=str(report_path), title=f"{strategy_name} 量化分析报告")
                
                # 生成基础统计报告
                stats_filename = f"{strategy_name}_stats_{timestamp}.txt"
                stats_path = results_path / stats_filename
                
                with open(stats_path, 'w', encoding='utf-8') as f:
                    title = f"{strategy_name} 量化分析统计报告"
                    if include_benchmark and benchmark_returns is not None:
                        title += "（含基线对比）"
                    f.write(f"{title}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # 定义安全的格式化函数
                    def safe_format(value, format_str):
                        """安全格式化quantstats返回值，处理Series和标量"""
                        try:
                            if hasattr(value, 'iloc'):
                                # 如果是Series，取第一个值
                                return format_str.format(float(value.iloc[0]))
                            elif pd.isna(value):
                                return "N/A"
                            else:
                                return format_str.format(float(value))
                        except:
                            return "N/A"
                    
                    # 定义安全的统计函数调用
                    def safe_stat_call(func, default_value="N/A"):
                        """安全调用quantstats统计函数"""
                        try:
                            return func()
                        except (AttributeError, TypeError, ValueError):
                            return default_value
                    
                    # 定义安全提取数值的函数
                    def safe_extract_value(stat_value):
                        """安全提取quantstats统计值，返回标量"""
                        try:
                            if hasattr(stat_value, 'iloc'):
                                return float(stat_value.iloc[0])
                            elif pd.isna(stat_value):
                                return 0.0
                            else:
                                return float(stat_value)
                        except:
                            return 0.0
                    
                    # 基本统计指标
                    f.write("基本统计指标:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"总收益率: {safe_format(qs.stats.comp(returns), '{:.2%}')}\n")
                    f.write(f"年化收益率: {safe_format(qs.stats.cagr(returns), '{:.2%}')}\n")
                    f.write(f"夏普比率: {safe_format(qs.stats.sharpe(returns), '{:.2f}')}\n")
                    f.write(f"最大回撤: {safe_format(qs.stats.max_drawdown(returns), '{:.2%}')}\n")
                    f.write(f"波动率: {safe_format(qs.stats.volatility(returns), '{:.2%}')}\n")
                    f.write(f"胜率: {safe_format(qs.stats.win_rate(returns), '{:.2%}')}\n")
                    f.write(f"盈亏比: {safe_format(qs.stats.profit_ratio(returns), '{:.2f}')}\n")
                    
                    # 风险指标
                    f.write(f"\n风险指标:\n")
                    f.write("-" * 30 + "\n")
                    
                    # 使用安全调用来处理可能不存在的函数
                    var_value = safe_stat_call(lambda: qs.stats.var(returns), 0)
                    cvar_value = safe_stat_call(lambda: qs.stats.cvar(returns), 0)
                    calmar_value = safe_stat_call(lambda: qs.stats.calmar(returns), 0)
                    
                    f.write(f"VaR (95%): {safe_format(var_value, '{:.2%}')}\n")
                    f.write(f"CVaR (95%): {safe_format(cvar_value, '{:.2%}')}\n")
                    f.write(f"Calmar比率: {safe_format(calmar_value, '{:.2f}')}\n")
                    
                    # 基线对比（如果有）
                    if include_benchmark and benchmark_returns is not None:
                        f.write(f"\n基线对比（买入持有策略）:\n")
                        f.write("-" * 30 + "\n")
                        f.write(f"基线总收益率: {safe_format(qs.stats.comp(benchmark_returns), '{:.2%}')}\n")
                        f.write(f"基线年化收益率: {safe_format(qs.stats.cagr(benchmark_returns), '{:.2%}')}\n")
                        f.write(f"基线夏普比率: {safe_format(qs.stats.sharpe(benchmark_returns), '{:.2f}')}\n")
                        f.write(f"基线最大回撤: {safe_format(qs.stats.max_drawdown(benchmark_returns), '{:.2%}')}\n")
                        f.write(f"基线波动率: {safe_format(qs.stats.volatility(benchmark_returns), '{:.2%}')}\n")
                        
                        # 计算策略相对基线的优势
                        try:
                            strategy_total = safe_extract_value(qs.stats.comp(returns))
                            benchmark_total = safe_extract_value(qs.stats.comp(benchmark_returns))
                            excess_return = strategy_total - benchmark_total
                            f.write(f"\n策略 vs 基线:\n")
                            f.write("-" * 30 + "\n")
                            f.write(f"超额收益: {excess_return:.2%}\n")
                            
                            # 计算信息比率（如果有）
                            try:
                                excess_returns = returns - benchmark_returns
                                if excess_returns.std() > 0:
                                    information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                                    f.write(f"信息比率: {information_ratio:.2f}\n")
                            except:
                                pass
                                
                        except Exception as e:
                            f.write(f"基线对比计算出错: {str(e)}\n")
                    
                    # 月度表现
                    f.write(f"\n月度收益率表现:\n")
                    f.write("-" * 30 + "\n")
                    monthly_returns = qs.stats.monthly_returns(returns)
                    f.write(str(monthly_returns))
                
                self.log(f"统计报告已保存：{stats_path}")
                
                # 定义安全提取数值的函数
                def safe_extract_value(stat_value):
                    """安全提取quantstats统计值，返回标量"""
                    try:
                        if hasattr(stat_value, 'iloc'):
                            return float(stat_value.iloc[0])
                        elif pd.isna(stat_value):
                            return 0.0
                        else:
                            return float(stat_value)
                    except:
                        return 0.0
                
                return {
                    'html_report_path': str(report_path),
                    'stats_report_path': str(stats_path),
                    'total_return': safe_extract_value(qs.stats.comp(returns)),
                    'annual_return': safe_extract_value(qs.stats.cagr(returns)),
                    'sharpe_ratio': safe_extract_value(qs.stats.sharpe(returns)),
                    'max_drawdown': safe_extract_value(qs.stats.max_drawdown(returns)),
                    'volatility': safe_extract_value(qs.stats.volatility(returns)),
                    'win_rate': safe_extract_value(qs.stats.win_rate(returns))
                }
            else:
                # 定义安全提取数值的函数
                def safe_extract_value(stat_value):
                    """安全提取quantstats统计值，返回标量"""
                    try:
                        if hasattr(stat_value, 'iloc'):
                            return float(stat_value.iloc[0])
                        elif pd.isna(stat_value):
                            return 0.0
                        else:
                            return float(stat_value)
                    except:
                        return 0.0
                
                # 仅返回关键指标，不保存文件
                return {
                    'total_return': safe_extract_value(qs.stats.comp(returns)),
                    'annual_return': safe_extract_value(qs.stats.cagr(returns)),
                    'sharpe_ratio': safe_extract_value(qs.stats.sharpe(returns)),
                    'max_drawdown': safe_extract_value(qs.stats.max_drawdown(returns)),
                    'volatility': safe_extract_value(qs.stats.volatility(returns)),
                    'win_rate': safe_extract_value(qs.stats.win_rate(returns))
                }
                
        except Exception as e:
            self.log(f"生成quantstats报告时出错: {str(e)}")
            import traceback
            self.log(f"详细错误信息: {traceback.format_exc()}")
            return None
