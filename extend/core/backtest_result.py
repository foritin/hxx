import pandas as pd
import numpy as np
from extend.data.data_source import MultiDataSource
import os

# 注释掉全局pandas配置，避免内存问题
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', 1000)


class BacktestResultCalculator:

    def __init__(self, logger=None):
        self.logger = logger
        self.results = {}
    
    def calculate_monthly_returns_from_trades(self, trades_df, initial_capital):
        """基于实际交易数据计算每月收益率"""
        try:
            if trades_df.empty:
                return [], 0, 0, 0
            
            # 确保datetime列是datetime类型
            if 'datetime' in trades_df.columns:
                trades_df['datetime'] = pd.to_datetime(trades_df['datetime'])
                # 添加年月列
                trades_df['year_month'] = trades_df['datetime'].dt.to_period('M')
            else:
                return [], 0, 0, 0
            
            # 按月分组计算收益
            monthly_returns = []
            current_capital = initial_capital
            
            for period in trades_df['year_month'].unique():
                month_trades = trades_df[trades_df['year_month'] == period]
                
                # 计算该月的净盈亏
                month_net_pnl = month_trades['net_pnl'].sum() if 'net_pnl' in month_trades.columns else 0
                
                # 计算月收益率
                if current_capital > 0:
                    monthly_return = (month_net_pnl / current_capital) * 100  # 百分比
                    monthly_returns.append(monthly_return)
                    current_capital += month_net_pnl
            
            if not monthly_returns:
                return [], 0, 0, 0
            
            # 计算统计指标
            monthly_return_avg = np.mean(monthly_returns)
            monthly_return_std = np.std(monthly_returns)
            
            # 月收益稳定性得分：平均收益越高、波动越小得分越高
            if monthly_return_std > 0:
                stability_score = monthly_return_avg / monthly_return_std
            else:
                stability_score = monthly_return_avg if monthly_return_avg > 0 else 0
            
            return monthly_returns, monthly_return_avg, monthly_return_std, stability_score
            
        except Exception as e:
            if self.logger and self.logger.debug_mode:
                self.log(f"计算月收益率时出错: {e}")
            return [], 0, 0, 0
    
    def calculate_composite_score(self, result):
        """计算综合评分（更新权重配置）"""
        net_value_score = (result['net_value'] - 1) * 100  # 净值超过1的部分百分比
        sharpe_score = result.get('sharpe_ratio', 0) * 20   # 夏普比率权重
        winrate_score = result.get('win_rate', 0) * 20      # 胜率权重（与夏普比相同）
        monthly_stability_score = result.get('monthly_stability_score', 0) * 15  # 月收益稳定性权重
        drawdown_penalty = result.get('max_drawdown_pct', 0) * 100  # 回撤惩罚
        trade_bonus = min(result['total_trades'] / 10, 5)  # 交易次数奖励（最多5分）
        
        # 综合评分 = 净值得分 + 夏普得分 + 胜率得分 + 月收益稳定性得分 + 交易奖励 - 回撤惩罚
        return net_value_score + sharpe_score + winrate_score + monthly_stability_score + trade_bonus - drawdown_penalty
    
    def enhance_results_with_metrics(self, results):
        """为结果添加增强的指标计算"""
        enhanced_results = []
        
        for result in results:
            # 从优化结果中提取月收益率指标（这些在optimization_worker中已经计算过了）
            # 如果结果中没有这些字段，设置默认值
            result['monthly_return_avg'] = result.get('monthly_return_avg', 0)
            result['monthly_return_std'] = result.get('monthly_return_std', 0) 
            result['monthly_stability_score'] = result.get('monthly_stability_score', 0)
            
            # 计算综合评分
            result['composite_score'] = self.calculate_composite_score(result)
            
            enhanced_results.append(result)
        
        return enhanced_results
    
    def get_sorted_results(self, results, sort_by='annual_return', top_n=5):
        """获取按指定指标排序的结果"""
        if sort_by == 'annual_return':
            return sorted(results, key=lambda x: x.get('annual_return', 0), reverse=True)[:top_n]
        elif sort_by == 'net_value':
            return sorted(results, key=lambda x: x['net_value'], reverse=True)[:top_n]
        elif sort_by == 'sharpe_ratio':
            valid_results = [r for r in results if r.get('sharpe_ratio', 0) > 0]
            return sorted(valid_results, key=lambda x: x['sharpe_ratio'], reverse=True)[:top_n]
        elif sort_by == 'max_drawdown_pct':
            valid_results = [r for r in results if r.get('max_drawdown_pct') is not None]
            return sorted(valid_results, key=lambda x: x['max_drawdown_pct'])[:top_n]
        elif sort_by == 'win_rate':
            valid_results = [r for r in results if r.get('win_rate', 0) > 0]
            return sorted(valid_results, key=lambda x: x.get('win_rate', 0), reverse=True)[:top_n]
        elif sort_by == 'composite_score':
            return sorted(results, key=lambda x: x['composite_score'], reverse=True)[:top_n]
        else:
            return results[:top_n]
    
    def calculate_statistics(self, results):
        """计算优化结果的统计信息"""
        stats = {
            'total_combinations': len(results),
            'valid_results_count': len([r for r in results if r['total_trades'] > 0]),
            'avg_net_value': sum(r['net_value'] for r in results) / len(results) if results else 0,
        }
        
        # 年化收益率统计
        if results:
            stats['avg_annual_return'] = sum(r.get('annual_return', 0) for r in results) / len(results)
        
        # 夏普比率统计
        valid_sharpe_results = [r for r in results if r.get('sharpe_ratio', 0) > 0]
        if valid_sharpe_results:
            stats['avg_sharpe_ratio'] = sum(r['sharpe_ratio'] for r in valid_sharpe_results) / len(valid_sharpe_results)
        
        # 胜率统计
        valid_winrate_results = [r for r in results if r.get('win_rate', 0) > 0]
        if valid_winrate_results:
            stats['avg_win_rate'] = sum(r.get('win_rate', 0) for r in valid_winrate_results) / len(valid_winrate_results)
        
        # 月收益率统计
        valid_monthly_results = [r for r in results if r.get('monthly_stability_score', 0) != 0]
        if valid_monthly_results:
            stats['avg_monthly_return'] = sum(r.get('monthly_return_avg', 0) for r in valid_monthly_results) / len(valid_monthly_results)
            stats['avg_monthly_std'] = sum(r.get('monthly_return_std', 0) for r in valid_monthly_results) / len(valid_monthly_results)
            stats['avg_monthly_stability'] = sum(r.get('monthly_stability_score', 0) for r in valid_monthly_results) / len(valid_monthly_results)
        
        return stats

    def log(self, msg):
        if self.logger:
            self.logger.log_message(msg)
        else:
            pass
            # print(msg)

    def calculate_drawdown(self, equity_curve):
        """计算最大回撤"""
        if len(equity_curve) == 0:
            return 0, 0
        
        equity_curve = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = peak - equity_curve
        max_drawdown = np.max(drawdown)
        max_drawdown_pct = max_drawdown / np.max(peak) if np.max(peak) > 0 else 0
        
        return max_drawdown, max_drawdown_pct

    def calculate_sharpe_ratio(self, equity_curve, risk_free_rate=0.03):
        """计算夏普比率（使用365天年化）"""
        if len(equity_curve) <= 1:
            return 0
        
        equity_curve = np.array(equity_curve)
        # 计算日收益率
        daily_returns = np.diff(equity_curve) / equity_curve[:-1]
        
        if len(daily_returns) == 0 or np.std(daily_returns) == 0:
            return 0
        
        # 计算年化夏普比率（使用365天）
        excess_return = np.mean(daily_returns) - risk_free_rate / 365
        sharpe_ratio = excess_return / np.std(daily_returns) * np.sqrt(365)
        
        return sharpe_ratio


    def generate_report(self, multi_data_source: MultiDataSource):
        """生成完整的回测报告（合并了calculate_results和calculate_performance的功能）"""
        results = {}

        for i, ds in enumerate(multi_data_source.data_sources):
            trades = ds.trades

            if not trades:
                # 在优化模式下静默处理无交易情况
                if self.logger and self.logger.debug_mode:
                    self.log(f"datasource {i} has no trades")
                continue
            
            # 优化版本：避免深拷贝，直接计算指标
            initial_capital = ds.capital
            
            # 直接从原始trades计算需要的数据，避免深拷贝
            trade_data = []
            cumulative_net_pnl = 0
            
            for trade in trades:
                net_pnl = trade.get('pnl', 0) - trade.get('fee', 0)
                cumulative_net_pnl += net_pnl
                equity = initial_capital + cumulative_net_pnl
                
                # 只创建需要的数据，不修改原始trade
                trade_row = {
                    'datetime': trade.get('datetime'),
                    'action': trade.get('action'),
                    'pnl': trade.get('pnl', 0),
                    'fee': trade.get('fee', 0),
                    'net_pnl': net_pnl,
                    'cumulative_net_pnl': cumulative_net_pnl,
                    'equity': equity
                }
                trade_data.append(trade_row)
            
            # 创建DataFrame (使用pandas)
            trade_df = pd.DataFrame(trade_data)
            
            # 基础统计计算 (使用pandas)
            # 从原始数据源获取日期范围
            if hasattr(ds.data, 'to_pandas'):
                # 如果是polars DataFrame，转换为pandas
                data_df = ds.data.to_pandas()
            else:
                # 如果已经是pandas DataFrame
                data_df = ds.data
            
            days_count = (data_df['datetime'].max() - data_df['datetime'].min()).days
            win_trades = len(trade_df[trade_df['pnl'] > 0])
            loss_trades = len(trade_df[trade_df['pnl'] < 0])
            total_trades = len(trades) // 2
            win_rate = win_trades / (win_trades + loss_trades) if (win_trades + loss_trades) > 0 else 0
            
            # 盈亏统计 (使用pandas)
            total_pnl = trade_df['pnl'].sum()
            total_fee = ds.total_fee
            total_net_profit = total_pnl - total_fee
            
            # 平均盈亏 (使用pandas)
            win_pnl_df = trade_df[trade_df['pnl'] > 0]
            loss_pnl_df = trade_df[trade_df['pnl'] < 0]
            avg_win = win_pnl_df['pnl'].mean() if win_trades > 0 else 0
            avg_loss = abs(loss_pnl_df['pnl'].mean()) if loss_trades > 0 else 0
            
            # 盈亏比 (使用pandas)
            win_sum = win_pnl_df['pnl'].sum() if len(win_pnl_df) > 0 else 0
            loss_sum = loss_pnl_df['pnl'].sum() if len(loss_pnl_df) > 0 else 0
            profit_factor = abs(win_sum / loss_sum) if loss_trades > 0 and loss_sum != 0 else float('inf') if win_trades > 0 else 0
            
            # 权益曲线和回撤计算 (使用pandas)
            equity_curve = trade_df['equity'].values if 'equity' in trade_df.columns else np.array([initial_capital])
            final_equity = equity_curve[-1] if len(equity_curve) > 0 else initial_capital
            net_value = final_equity / initial_capital if initial_capital > 0 else 1
            
            # 最大回撤计算
            max_drawdown, max_drawdown_pct = self.calculate_drawdown(equity_curve)
            
            # 年化收益率
            trading_years = days_count / 365 if days_count > 0 else 1
            annual_return = ((final_equity / initial_capital) ** (1/trading_years) - 1) * 100 if trading_years > 0 and initial_capital > 0 else 0
            
            # 夏普比率（假设无风险利率为3%，使用365天年化）
            sharpe_ratio = self.calculate_sharpe_ratio(equity_curve, risk_free_rate=0.03)
            
            # 计算月收益率指标（使用实际交易数据）
            monthly_returns, monthly_return_avg, monthly_return_std, monthly_stability_score = self.calculate_monthly_returns_from_trades(trade_df, initial_capital)
            
            # 保存结果（优化版本：不保存大对象）
            ds_result = {
                'symbol': ds.symbol,
                'kline_period': ds.kline_period,
                'total_trades': total_trades,
                'win_trades': win_trades,
                'loss_trades': loss_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_fee': total_fee,
                'total_net_profit': total_net_profit,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'initial_capital': initial_capital,
                'final_equity': final_equity,
                'net_value': net_value,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'trading_days': days_count,
                'monthly_returns': monthly_returns,
                'monthly_return_avg': monthly_return_avg,
                'monthly_return_std': monthly_return_std,
                'monthly_stability_score': monthly_stability_score,
                # 在优化模式下不保存大对象，减少内存使用
                'trades': trades if (self.logger and self.logger.debug_mode) else [],
                'trade_df': None,  # 不保存DataFrame，避免内存泄漏
                'equity_curve': equity_curve
            }
            
            results[f'datasource_{i}'] = ds_result
            
            # 在优化模式下跳过详细日志打印，减少内存使用
            if self.logger and not self.logger.debug_mode:
                # 只打印汇总信息，不打印详细交易记录
                pass
            else:
                # 打印包含权益变动的交易记录 (使用pandas)
                self.log(f"数据源 {i} ({ds.symbol}) 交易记录:")
                selected_columns = ["datetime", "action", "open_price", "close_price", "volume", "current_pos", "pnl", "fee", "net_pnl", "equity"]
                # 只选择存在的列
                available_columns = [col for col in selected_columns if col in trade_df.columns]
                self.log(trade_df[available_columns])
                
            # 立即清理DataFrame引用，释放内存
            del trade_data
            del trade_df
            
            # 打印汇总统计
            self.log(f"数据源 {i} ({ds.symbol}) 回测结果:")
            self.log(f"  总交易次数: {total_trades}")
            self.log(f"  胜率: {win_rate:.2%}")
            self.log(f"  总盈亏: {total_net_profit:.2f}")
            self.log(f"  最大回撤: {max_drawdown_pct:.2%}")
            self.log(f"  年化收益率: {annual_return:.2f}%")
            self.log(f"  夏普比率: {sharpe_ratio:.2f}")
            self.log(f"  盈亏比: {profit_factor:.2f}")
            self.log(f"  月收益率平均: {monthly_return_avg:.2f}%")
            self.log(f"  月收益率标准差: {monthly_return_std:.2f}%")
            self.log(f"  月收益稳定性得分: {monthly_stability_score:.2f}")
        
        # 计算整体表现指标并合并到results中
        if results:
            # 汇总所有数据源的结果
            total_trades = sum(r['total_trades'] for r in results.values())
            total_win_trades = sum(r['win_trades'] for r in results.values())
            total_loss_trades = sum(r['loss_trades'] for r in results.values())
            overall_win_rate = total_win_trades / (total_win_trades + total_loss_trades) if (total_win_trades + total_loss_trades) > 0 else 0
            
            total_net_profit = sum(r['total_net_profit'] for r in results.values())
            total_initial_capital = sum(r['initial_capital'] for r in results.values())
            total_final_equity = sum(r['final_equity'] for r in results.values())
            
            overall_return = (total_final_equity - total_initial_capital) / total_initial_capital if total_initial_capital > 0 else 0
            
            # 计算加权平均的其他指标
            weights = [r['total_trades'] for r in results.values()]
            total_weight = sum(weights) if sum(weights) > 0 else 1
            
            weighted_sharpe = sum(r['sharpe_ratio'] * w for r, w in zip(results.values(), weights)) / total_weight
            weighted_max_drawdown_pct = sum(r['max_drawdown_pct'] * w for r, w in zip(results.values(), weights)) / total_weight
            
            # 计算整体月收益率指标
            all_monthly_returns = []
            for r in results.values():
                if r.get('monthly_returns'):
                    all_monthly_returns.extend(r['monthly_returns'])
            
            overall_monthly_return_avg = np.mean(all_monthly_returns) if all_monthly_returns else 0
            overall_monthly_return_std = np.std(all_monthly_returns) if all_monthly_returns else 0
            overall_monthly_stability = overall_monthly_return_avg / overall_monthly_return_std if overall_monthly_return_std > 0 else 0
            
            # 将整体性能指标添加到results中
            results['_overall_performance'] = {
                'total_datasources': len(results),  # 数据源总数
                'total_trades': total_trades,
                'overall_win_rate': overall_win_rate,
                'total_net_profit': total_net_profit,
                'total_initial_capital': total_initial_capital,
                'total_final_equity': total_final_equity,
                'overall_return': overall_return,
                'overall_return_pct': overall_return * 100,
                'weighted_sharpe_ratio': weighted_sharpe,
                'weighted_max_drawdown_pct': weighted_max_drawdown_pct,
                'overall_monthly_return_avg': overall_monthly_return_avg,
                'overall_monthly_return_std': overall_monthly_return_std,
                'overall_monthly_stability': overall_monthly_stability
            }
            
            # 打印整体报告
            self.log("\n" + "="*50)
            self.log("回测整体报告")
            self.log("="*50)
            
            overall = results['_overall_performance']
            self.log(f"数据源数量: {overall['total_datasources']}")
            self.log(f"总交易次数: {overall['total_trades']}")
            self.log(f"整体胜率: {overall['overall_win_rate']:.2%}")
            self.log(f"总净盈亏: {overall['total_net_profit']:.2f}")
            self.log(f"总初始资金: {overall['total_initial_capital']:.2f}")
            self.log(f"总最终权益: {overall['total_final_equity']:.2f}")
            self.log(f"整体收益率: {overall['overall_return_pct']:.2f}%")
            self.log(f"加权夏普比率: {overall['weighted_sharpe_ratio']:.2f}")
            self.log(f"加权最大回撤: {overall['weighted_max_drawdown_pct']:.2%}")
            self.log(f"整体月收益率平均: {overall['overall_monthly_return_avg']:.2f}%")
            self.log(f"整体月收益率标准差: {overall['overall_monthly_return_std']:.2f}%")
            self.log(f"整体月收益稳定性: {overall['overall_monthly_stability']:.2f}")
            
            self.log("="*50)
        
        return results
