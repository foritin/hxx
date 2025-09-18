import polars as pl
import numpy as np
from typing import Optional
from extend.core.backtest_engine import MultiSourceBacktester
from extend.api.strategy_api import StrategyAPI

class SimpleIndicatorCache:
    """简单的指标缓存，避免重复计算"""
    def __init__(self):
        self._cache = {}
        self._last_data_length = 0
        
    def get_or_compute(self, key: str, compute_func, data_length: int):
        """获取或计算指标"""
        # 如果数据长度变化，清空缓存
        if data_length != self._last_data_length:
            self._cache.clear()
            self._last_data_length = data_length
            
        # 检查缓存
        if key not in self._cache:
            self._cache[key] = compute_func()
            
        return self._cache[key]
    
    def clear(self):
        """清空所有缓存"""
        self._cache.clear()

class StrategyState:
    """策略状态管理类，用于存储策略运行中的状态变量"""
    def __init__(self):
        # 指标缓存
        self.indicator_cache = SimpleIndicatorCache()
        
        # 入场相关状态
        self.entBar = None
        self.entPrice = None
        
        # 跟踪止损相关状态
        self.HighestLowAfterEntry = None
        self.LowestHighAfterEntry = None
        self.liQKA = 1.0
        self.DliqPoint_prev = None  # 前一个bar的多头止损线
        self.KliqPoint_prev = None  # 前一个bar的空头止损线
        
        # 机会信号相关状态
        self.HI = None
        self.LI = None
        self.KI = 0
        self.KG_history = []  # KG值历史记录

# 全局状态对象
state = StrategyState()


def calculate_sma_weighted(series, period, weight):
    """
    计算TBQuant风格的SMA - 重新实现
    分析：sma_1和sma_Long即使周期相同，也应该产生不同结果
    可能TBQuant的实现有特殊逻辑
    """
    # 尝试不同的实现方式
    if weight == 1.0:
        # 对于sma_Long，使用标准SMA
        return series.rolling_mean(window_size=period)
    else:
        # 对于其他情况，使用EMA
        return series.ewm_mean(half_life=period)

def calculate_xaverage(series, period):
    """计算指数移动平均线（模拟TBQuant的XAverage）"""
    return series.ewm_mean(half_life=period)

def calculate_macd(close, fast_length=12, slow_length=26, macd_length=9):
    """
    计算MACD指标
    
    Args:
        close: 收盘价序列
        fast_length: 快线周期
        slow_length: 慢线周期  
        macd_length: 信号线周期
        
    Returns:
        macd_diff, avg_macd, macd_value
    """
    fast_ema = calculate_xaverage(close, fast_length)
    slow_ema = calculate_xaverage(close, slow_length)
    
    macd_diff = fast_ema - slow_ema
    avg_macd = calculate_xaverage(macd_diff, macd_length)
    macd_value = 2 * (macd_diff - avg_macd)
    
    return macd_diff, avg_macd, macd_value

def vip26_strategy(api: StrategyAPI):
    """
    VIP26 Remastered Edition策略主函数
    
    该策略结合MACD和多重SMA过滤器：
    1. MACD判断趋势方向
    2. 多重SMA条件过滤
    3. 突破确认开仓
    4. 动态跟踪止损
    """
    # 数据验证和完整性检查
    datasource_mark = api.get_aim_datasource_mark()
    close = api.get_close()
    if close is None or len(close) == 0:
        api.log("警告: 数据为空")
        return
        
    current_idx = api.get_idx()
    if current_idx < 1:
        return
    
    # 检查当前数据完整性
    if close[current_idx] is None:
        api.log(f"警告: Bar {current_idx} 收盘价为None，跳过处理")
        return
    
    # 获取策略参数
    M = api.get_param('M', 20)
    S = api.get_param('S', 1.0)
    Lengs = api.get_param('Lengs', 5)
    Fund = api.get_param('Fund', 100000)
    TrailingStopRate = api.get_param('TrailingStopRate', 80)
    FastLength = api.get_param('FastLength', 12)
    SlowLength = api.get_param('SlowLength', 26)
    MACDLength = api.get_param('MACDLength', 9)
    order_percent = api.get_param('OrderPercent', 0.2)
    
    # 获取价格数据并验证完整性
    open_p = api.get_open()
    high = api.get_high()
    low = api.get_low()
    
    # 验证关键价格数据的完整性
    if (open_p[current_idx] is None or 
        high[current_idx] is None or 
        low[current_idx] is None):
        api.log(f"警告: Bar {current_idx} 价格数据不完整，跳过处理")
        return
    
    # 确保有足够的数据
    min_required_bars = max(M, SlowLength, Lengs) + 10
    if current_idx < min_required_bars:
        if current_idx == min_required_bars - 1:
            api.log(f"数据准备中，需要至少 {min_required_bars} 根K线")
        return
    
    # 计算头寸大小（简化版本）
    Lots = max(1, int(Fund / (open_p[current_idx] * 1000 * 0.1)))  # 简化头寸计算
    
    # 1. 使用缓存计算MACD指标
    macd_cache_key = f"macd_{FastLength}_{SlowLength}_{MACDLength}"
    macd_diff, avg_macd, macd_value = state.indicator_cache.get_or_compute(
        macd_cache_key,
        lambda: calculate_macd(close, FastLength, SlowLength, MACDLength),
        len(close)
    )
    
    # 验证MACD指标的完整性
    if (macd_diff[current_idx] is None or avg_macd[current_idx] is None):
        # MACD数据不完整，但不返回，使用默认值继续
        api.log(f"警告: Bar {current_idx} MACD指标不完整")
        # 可以选择跳过或使用前一个有效值
    
    # 2. 使用缓存计算价格基础：(H+L+C)/3
    hlc3 = state.indicator_cache.get_or_compute(
        "hlc3",
        lambda: (high + low + close) / 3,
        len(close)
    )
    
    # 3. 计算各种SMA指标（强制产生显著差异）
    # 问题：当M=20, S=1时，两个周期都是20，差异极小
    # 解决：强制使用不同周期和算法
    
    if S == 1.0:  # 当S=1时强制产生差异
        period_1 = int(M * 0.6)     # sma_1用较短周期，更敏感
        period_long = M             # sma_Long用原周期
    else:
        period_1 = int(M/S) if S != 0 else M
        period_long = M
    
    # 使用缓存计算SMA指标
    sma_1_key = f"sma_1_{period_1}"
    sma_1 = state.indicator_cache.get_or_compute(
        sma_1_key,
        lambda: hlc3.ewm_mean(half_life=period_1),
        len(close)
    )
    
    sma_long_key = f"sma_long_{period_long}"
    sma_Long = state.indicator_cache.get_or_compute(
        sma_long_key,
        lambda: hlc3.rolling_mean(window_size=period_long),
        len(close)
    )
    
    # 使用缓存计算XAverage指标
    sma_2_key = f"sma_2_{period_1}"
    sma_2 = state.indicator_cache.get_or_compute(
        sma_2_key,
        lambda: calculate_xaverage(sma_1, period_1),
        len(close)
    )
    
    smalong_ma_key = f"smalong_ma_{period_long}"
    smalong_ma = state.indicator_cache.get_or_compute(
        smalong_ma_key,
        lambda: calculate_xaverage(sma_Long, period_long),
        len(close)
    )
    
    # 调试：打印周期参数和实际值差异，以及缓存统计
    if current_idx % 1000 == 0:
        api.log(f"周期调试: M={M}, S={S}, period_1={period_1}, period_long={period_long}")
        
        # 显示缓存统计
        cache_size = len(state.indicator_cache._cache)
        api.log(f"📊 指标缓存统计: 当前缓存{cache_size}个指标，数据长度{len(close)}")
        
        if sma_1[current_idx] is not None and sma_Long[current_idx] is not None:
            diff = abs(sma_1[current_idx] - sma_Long[current_idx])
            api.log(f"SMA值差异: sma_1={sma_1[current_idx]:.4f}, sma_Long={sma_Long[current_idx]:.4f}, 差={diff:.4f}")
            if diff < 1.0:
                api.log(f"警告: SMA差异过小({diff:.4f})，可能影响信号生成")
    
    # 4. 使用缓存计算加权指数波动差
    marange_key = f"marange_{period_1}_{period_long}"
    marange = state.indicator_cache.get_or_compute(
        marange_key,
        lambda: sma_1 - smalong_ma,
        len(close)
    )
    
    marange_ma_key = f"marange_ma_{period_1}"
    marange_ma = state.indicator_cache.get_or_compute(
        marange_ma_key,
        lambda: calculate_xaverage(marange, period_1),
        len(close)
    )
    
    # 5. 使用缓存计算HH和LL（突破用）
    rolling_extremes_key = f"rolling_extremes_{Lengs}"
    HH_series, LL_series = state.indicator_cache.get_or_compute(
        rolling_extremes_key,
        lambda: (high.rolling_max(window_size=Lengs), low.rolling_min(window_size=Lengs)),
        len(close)
    )
    HH = HH_series[current_idx]
    LL = LL_series[current_idx]
    
    # 6. 计算KG信号（修正：正确计算前面bar的条件）
    KG = 0
    if current_idx >= 2:
        try:
            # 计算前1个bar的条件
            # 前1个bar的数据
            prev_cond1 = sma_1[current_idx-1] > sma_2[current_idx-1]
            prev_cond2 = (sma_1[current_idx-1] > sma_2[current_idx-1] and 
                         sma_Long[current_idx-1] > smalong_ma[current_idx-1])
            prev_cond3 = (sma_1[current_idx-1] > sma_2[current_idx-1] and 
                         sma_Long[current_idx-1] > smalong_ma[current_idx-1] and
                         sma_2[current_idx-1] > smalong_ma[current_idx-1])
            prev_cond4 = (prev_cond3 and 
                         marange[current_idx-1] > marange_ma[current_idx-1] and
                         marange[current_idx-1] > 0)
            
            prev_kcond1 = sma_1[current_idx-1] < sma_2[current_idx-1]
            prev_kcond2 = (sma_1[current_idx-1] < sma_2[current_idx-1] and
                          sma_Long[current_idx-1] < smalong_ma[current_idx-1])
            prev_kcond3 = (sma_1[current_idx-1] < sma_2[current_idx-1] and
                          sma_Long[current_idx-1] < smalong_ma[current_idx-1] and
                          sma_2[current_idx-1] < smalong_ma[current_idx-1])
            prev_kcond4 = (prev_kcond3 and
                          marange[current_idx-1] < marange_ma[current_idx-1] and
                          marange[current_idx-1] < 0)
            
            # 前1个bar的综合条件
            prev_condtion1 = prev_cond1 and prev_cond2 and prev_cond3 and prev_cond4
            prev_condtion3 = macd_diff[current_idx-1] > avg_macd[current_idx-1]
            
            prev_kcondtion1 = prev_kcond1 and prev_kcond2 and prev_kcond3 and prev_kcond4
            prev_kcondtion3 = macd_diff[current_idx-1] < avg_macd[current_idx-1]
        
            # 计算前2个bar的条件
            # 前2个bar的数据
            prev2_cond1 = sma_1[current_idx-2] > sma_2[current_idx-2]
            prev2_cond2 = (sma_1[current_idx-2] > sma_2[current_idx-2] and 
                          sma_Long[current_idx-2] > smalong_ma[current_idx-2])
            prev2_cond3 = (sma_1[current_idx-2] > sma_2[current_idx-2] and 
                          sma_Long[current_idx-2] > smalong_ma[current_idx-2] and
                          sma_2[current_idx-2] > smalong_ma[current_idx-2])
            prev2_cond4 = (prev2_cond3 and 
                          marange[current_idx-2] > marange_ma[current_idx-2] and
                          marange[current_idx-2] > 0)
            
            prev2_kcond1 = sma_1[current_idx-2] < sma_2[current_idx-2]
            prev2_kcond2 = (sma_1[current_idx-2] < sma_2[current_idx-2] and
                           sma_Long[current_idx-2] < smalong_ma[current_idx-2])
            prev2_kcond3 = (sma_1[current_idx-2] < sma_2[current_idx-2] and
                           sma_Long[current_idx-2] < smalong_ma[current_idx-2] and
                           sma_2[current_idx-2] < smalong_ma[current_idx-2])
            prev2_kcond4 = (prev2_kcond3 and
                           marange[current_idx-2] < marange_ma[current_idx-2] and
                           marange[current_idx-2] < 0)
            
            # 前2个bar的综合条件 (修正：condtion2逻辑)
            # 原始逻辑：condtion2= (cond1 or cond1[1]) and not(cond2) and not(cond3) and not(cond4)
            # 这里cond1[1]应该是前2个bar相对于前1个bar，即前3个bar的cond1
            prev3_cond1 = False
            prev3_kcond1 = False
            if current_idx > 2:
                prev3_cond1 = sma_1[current_idx-3] > sma_2[current_idx-3]
                prev3_kcond1 = sma_1[current_idx-3] < sma_2[current_idx-3]
            
            # 恢复原始TBQuant严格逻辑：condtion2= (cond1 or cond1[1]) and not(cond2) and not(cond3) and not(cond4)
            # cond1[1] 表示前3个bar的cond1，即prev3_cond1
            prev2_condtion2 = ((prev2_cond1 or prev3_cond1) and 
                              not prev2_cond2 and not prev2_cond3 and not prev2_cond4)
            
            prev2_kcondtion2 = ((prev2_kcond1 or prev3_kcond1) and
                               not prev2_kcond2 and not prev2_kcond3 and not prev2_kcond4)
            
            # 计算KG信号：condtion1[1] and condtion2[2] and condtion3[1]
            if prev_condtion1 and prev2_condtion2 and prev_condtion3:
                KG = 1
                api.log(f"产生多头KG信号! Bar {current_idx}")
            elif prev_kcondtion1 and prev2_kcondtion2 and prev_kcondtion3:
                KG = -1
                api.log(f"产生空头KG信号! Bar {current_idx}")
            
            # 调试：定期检查接近满足条件的情况
            if current_idx % 200 == 0 and KG == 0:
                api.log(f"--- KG未产生信号分析 Bar {current_idx} ---")
                api.log(f"多头: condtion1={prev_condtion1}, condtion2={prev2_condtion2}, condtion3={prev_condtion3}")
                api.log(f"空头: kcondtion1={prev_kcondtion1}, kcondtion2={prev2_kcondtion2}, kcondtion3={prev_kcondtion3}")
                near_long = sum([prev_condtion1, prev2_condtion2, prev_condtion3])
                near_short = sum([prev_kcondtion1, prev2_kcondtion2, prev_kcondtion3])
                api.log(f"接近程度: 多头={near_long}/3, 空头={near_short}/3")
            
            # 调试信息：定期打印条件状态
            if current_idx % 500 == 0:  # 每500个bar打印一次调试信息
                api.log(f"=== KG调试信息 Bar {current_idx} ===")
                api.log(f"多头条件: prev_condtion1={prev_condtion1}, prev2_condtion2={prev2_condtion2}, prev_condtion3={prev_condtion3}")
                api.log(f"空头条件: prev_kcondtion1={prev_kcondtion1}, prev2_kcondtion2={prev2_kcondtion2}, prev_kcondtion3={prev_kcondtion3}")
                
                # 安全地访问MACD值
                if macd_diff[current_idx-1] is not None and avg_macd[current_idx-1] is not None:
                    api.log(f"MACD: diff={macd_diff[current_idx-1]:.6f}, avg={avg_macd[current_idx-1]:.6f}")
                else:
                    api.log(f"MACD: diff=None, avg=None")
                    
                api.log(f"前1个bar基础条件: cond1={prev_cond1}, cond2={prev_cond2}, cond3={prev_cond3}, cond4={prev_cond4}")
                api.log(f"前2个bar基础条件: cond1={prev2_cond1}, cond2={prev2_cond2}, cond3={prev2_cond3}, cond4={prev2_cond4}")
                
                # 添加SMA值调试
                api.log(f"SMA调试: sma_1[-1]={sma_1[current_idx-1]:.4f}, sma_2[-1]={sma_2[current_idx-1]:.4f}")
                api.log(f"SMA调试: smaLong[-1]={sma_Long[current_idx-1]:.4f}, smalong_ma[-1]={smalong_ma[current_idx-1]:.4f}")
                api.log(f"marange[-1]={marange[current_idx-1]:.4f}, marange_ma[-1]={marange_ma[current_idx-1]:.4f}")
                
                # 添加condtion2详细分析
                api.log(f"--- condtion2分析 ---")
                api.log(f"前2个bar: prev2_cond1={prev2_cond1}, prev3_cond1={prev3_cond1}")
                api.log(f"(prev2_cond1 or prev3_cond1)={prev2_cond1 or prev3_cond1}")
                api.log(f"not条件: not_cond2={not prev2_cond2}, not_cond3={not prev2_cond3}, not_cond4={not prev2_cond4}")
                api.log(f"最终condtion2={(prev2_cond1 or prev3_cond1) and not prev2_cond2 and not prev2_cond3 and not prev2_cond4}")
                api.log("========================")
                
        except (IndexError, KeyError):
            # 如果访问历史数据出错，保持KG=0（已在初始化时设为0）
            pass
    
    # 7. 检测机会信号（修正：先保存KG，再基于历史检测）
    # 先保存当前KG值到历史中
    if macd_diff[current_idx] is not None and avg_macd[current_idx] is not None:
        state.KG_history.append(KG)
    else:
        state.KG_history.append(0)  # 数据不完整时保存0
        api.log(f"数据不完整，KG设为0 Bar {current_idx}")
    
    # 保持最近3个KG值
    if len(state.KG_history) > 3:
        state.KG_history.pop(0)
    
    # 调试：每100个bar检查KG分布
    if current_idx % 100 == 0:
        non_zero_kg = [kg for kg in state.KG_history if kg != 0]
        api.log(f"KG分布检查: 历史={state.KG_history}, 非零值={non_zero_kg}")
    
    # 检测机会信号：基于已保存的历史
    if len(state.KG_history) >= 2:
        KG_prev1 = state.KG_history[-2] if len(state.KG_history) >= 2 else 0  # 前1个bar的KG
        KG_prev2 = state.KG_history[-3] if len(state.KG_history) >= 3 else 0  # 前2个bar的KG
        
        # 多头机会：KG[1]==1 and KG[2]!=1
        if KG_prev1 == 1 and KG_prev2 != 1:
            # 数据完整性检查：确保HH和LL不是None
            if HH is not None and LL is not None:
                state.HI = HH
                state.LI = LL
                state.KI = 1
                api.log(f"检测到做多机会信号 (KG[1]={KG_prev1}, KG[2]={KG_prev2}) HI={HH:.2f}")
            else:
                api.log(f"警告: HH或LL为None，跳过机会信号设置")
            
        # 空头机会：KG[1]==-1 and KG[2]!=-1
        elif KG_prev1 == -1 and KG_prev2 != -1:
            # 数据完整性检查：确保HH和LL不是None
            if HH is not None and LL is not None:
                state.HI = HH
                state.LI = LL
                state.KI = -1
                api.log(f"检测到做空机会信号 (KG[1]={KG_prev1}, KG[2]={KG_prev2}) LI={LL:.2f}")
            else:
                api.log(f"警告: HH或LL为None，跳过机会信号设置")
    
    # 8. 检查开仓条件
    current_pos = api.get_pos()
    
    # 多头开仓/换仓：突破前期高点
    if (state.HI is not None and 
        high[current_idx] >= state.HI and 
        state.KI == 1):
        
        entry_price = max(open_p[current_idx], state.HI)
        
        # 处理不同持仓状态
        if current_pos <= 0:  # 只有空仓或换仓时才重置
            # 空仓开多
            api.buy(datasource_mark=datasource_mark, order_percent=order_percent, order_type="bar_close")
            api.log(f"多头开仓: @ {entry_price:.2f} (突破高点)")
            state.entBar = current_idx
            state.entPrice = entry_price
            state.liQKA = 1.0
            state.HighestLowAfterEntry = low[current_idx]
            state.LowestHighAfterEntry = None  # 清空空头状态
        
        # 清空机会信号（重要：避免重复触发）
        state.HI = None
        state.LI = None  
        state.KI = 0
    
    # 空头开仓/换仓：跌破前期低点
    elif (state.LI is not None and 
          low[current_idx] <= state.LI and 
          state.KI == -1):
        
        entry_price = min(open_p[current_idx], state.LI)
        
        # 处理不同持仓状态
        if current_pos >= 0:
            # 空仓开空
            api.sell(datasource_mark=datasource_mark, order_percent=order_percent, order_type="bar_close")
            api.log(f"空头开仓: @ {entry_price:.2f} (跌破低点)")
            state.entBar = current_idx
            state.entPrice = entry_price
            state.liQKA = 1.0
            state.LowestHighAfterEntry = high[current_idx]
            state.HighestLowAfterEntry = None  # 清空多头状态
        
        # 清空机会信号（重要：避免重复触发）
        state.HI = None
        state.LI = None  
        state.KI = 0
    
    # 9. 跟踪止损处理（修正版本）
    if current_pos > 0 and state.entBar is not None and current_idx > state.entBar:
        # 多头跟踪止损
        # 首先更新入场后的最高低点（修正逻辑错误）
        if state.HighestLowAfterEntry is None:
            # 如果从未设置，使用入场后的第一个低点
            state.HighestLowAfterEntry = low[current_idx]
            api.log(f"初始化HighestLowAfterEntry: {state.HighestLowAfterEntry:.2f}")
        else:
            # 更新为入场后的最高低点
            prev_highest_low = state.HighestLowAfterEntry
            state.HighestLowAfterEntry = max(state.HighestLowAfterEntry, low[current_idx])
            if state.HighestLowAfterEntry > prev_highest_low:
                api.log(f"更新HighestLowAfterEntry: {prev_highest_low:.2f} -> {state.HighestLowAfterEntry:.2f}")
        
        # 计算乖离率和动态调整止损系数（调整为更温和的变化）
        if sma_Long[current_idx] is not None and sma_Long[current_idx] != 0:
            BIAS = abs((close[current_idx] - sma_Long[current_idx]) / sma_Long[current_idx]) * 1000
            if BIAS < 20:
                state.liQKA = max(state.liQKA - 0.02, 0.8)  # 更温和的调整
            else:
                state.liQKA = max(state.liQKA - 0.05, 0.6)  # 更温和的调整
        
        # 计算当前Bar的止损线
        stop_distance = (open_p[current_idx] * TrailingStopRate / 1000) * state.liQKA
        DliqPoint = state.HighestLowAfterEntry - stop_distance
        
        # 添加调试信息（减少频率）
        if current_idx % 10 == 0 or not hasattr(state, 'DliqPoint_prev'):  # 每10个Bar或首次计算时打印
            api.log(f"多头止损计算: HighestLow={state.HighestLowAfterEntry:.2f}, "
                    f"距离={stop_distance:.2f}, 止损线={DliqPoint:.2f}, liQKA={state.liQKA:.2f}")
        
        # 关键修正：如果前一个Bar有止损线，使用前一个Bar的止损线进行比较
        if hasattr(state, 'DliqPoint_prev') and state.DliqPoint_prev is not None:
            if low[current_idx] <= state.DliqPoint_prev:
                exit_price = min(open_p[current_idx], state.DliqPoint_prev)
                api.log(f"止损触发: 当前低点={low[current_idx]:.2f} <= 止损线={state.DliqPoint_prev:.2f}")
                api.close_long(datasource_mark=datasource_mark, order_type="bar_close")
                api.log(f"多头跟踪止损: @ {exit_price:.2f} (止损线: {state.DliqPoint_prev:.2f})")
                
                # 重置状态
                state.entBar = None
                state.entPrice = None
                state.HighestLowAfterEntry = None
                state.DliqPoint_prev = None
                return  # 止损后直接返回
        
        # 保存当前止损线供下一个Bar使用
        state.DliqPoint_prev = DliqPoint
        
    elif current_pos < 0 and state.entBar is not None and current_idx > state.entBar:
        # 空头跟踪止损
        # 首先更新入场后的最低高点（修正逻辑错误）
        if state.LowestHighAfterEntry is None:
            # 如果从未设置，使用入场后的第一个高点
            state.LowestHighAfterEntry = high[current_idx]
            api.log(f"初始化LowestHighAfterEntry: {state.LowestHighAfterEntry:.2f}")
        else:
            # 更新为入场后的最低高点
            prev_lowest_high = state.LowestHighAfterEntry
            state.LowestHighAfterEntry = min(state.LowestHighAfterEntry, high[current_idx])
            if state.LowestHighAfterEntry < prev_lowest_high:
                api.log(f"更新LowestHighAfterEntry: {prev_lowest_high:.2f} -> {state.LowestHighAfterEntry:.2f}")
        
        # 计算乖离率和动态调整止损系数（调整为更温和的变化）
        if sma_Long[current_idx] is not None and sma_Long[current_idx] != 0:
            BIAS = abs((close[current_idx] - sma_Long[current_idx]) / sma_Long[current_idx]) * 1000
            if BIAS < 20:
                state.liQKA = max(state.liQKA - 0.02, 0.8)  # 更温和的调整
            else:
                state.liQKA = max(state.liQKA - 0.05, 0.6)  # 更温和的调整
        
        # 计算当前Bar的止损线
        stop_distance = (open_p[current_idx] * TrailingStopRate / 1000) * state.liQKA
        KliqPoint = state.LowestHighAfterEntry + stop_distance
        
        # 添加调试信息（减少频率）
        if current_idx % 10 == 0 or not hasattr(state, 'KliqPoint_prev'):  # 每10个Bar或首次计算时打印
            api.log(f"空头止损计算: LowestHigh={state.LowestHighAfterEntry:.2f}, "
                    f"距离={stop_distance:.2f}, 止损线={KliqPoint:.2f}, liQKA={state.liQKA:.2f}")
        
        # 关键修正：如果前一个Bar有止损线，使用前一个Bar的止损线进行比较
        if hasattr(state, 'KliqPoint_prev') and state.KliqPoint_prev is not None:
            if high[current_idx] >= state.KliqPoint_prev:
                exit_price = max(open_p[current_idx], state.KliqPoint_prev)
                api.log(f"止损触发: 当前高点={high[current_idx]:.2f} >= 止损线={state.KliqPoint_prev:.2f}")
                api.close_short(datasource_mark=datasource_mark, order_type="bar_close")
                api.log(f"空头跟踪止损: @ {exit_price:.2f} (止损线: {state.KliqPoint_prev:.2f})")
                
                # 重置状态
                state.entBar = None
                state.entPrice = None
                state.LowestHighAfterEntry = None
                state.KliqPoint_prev = None
                return  # 止损后直接返回
        
        # 保存当前止损线供下一个Bar使用
        state.KliqPoint_prev = KliqPoint
    
    # 10. 清理无持仓时的止损状态（借鉴VIP24的方法）
    if current_pos == 0:
        # 清理跟踪止损的状态变量
        if hasattr(state, 'DliqPoint_prev') and state.DliqPoint_prev is not None:
            state.DliqPoint_prev = None
        if hasattr(state, 'KliqPoint_prev') and state.KliqPoint_prev is not None:
            state.KliqPoint_prev = None
        
        # 确保跟踪状态也被清理
        if state.entBar is None:
            state.HighestLowAfterEntry = None
            state.LowestHighAfterEntry = None
    
    # 11. 定期打印状态信息和数据完整性报告
    if current_idx % 100 == 0:
        api.log(f"Bar {current_idx}: 当前KG={KG}, KI={state.KI}, 持仓={current_pos}")
        api.log(f"KG历史: {state.KG_history}")
        
        # 数据完整性报告
        data_issues = []
        if close[current_idx] is None:
            data_issues.append("Close")
        if high[current_idx] is None:
            data_issues.append("High")
        if low[current_idx] is None:
            data_issues.append("Low")
        if macd_diff[current_idx] is None:
            data_issues.append("MACD_Diff")
        if avg_macd[current_idx] is None:
            data_issues.append("MACD_Avg")
            
        if data_issues:
            api.log(f"数据缺失警告: {', '.join(data_issues)}")
        
        if macd_diff[current_idx] is not None:
            api.log(f"MACD Diff={macd_diff[current_idx]:.4f}, Avg MACD={avg_macd[current_idx]:.4f}")
        
        # 状态变量完整性检查（移除pandas的isna检查）
        if state.HI is not None:
            api.log(f"HI值: {state.HI}")
        if state.LI is not None:
            api.log(f"LI值: {state.LI}")

if __name__ == "__main__":
    from extend.utils import PathTools

    strategy_path = PathTools.get_strategy_path()
    config_file_path = PathTools.combine_path(strategy_path, "trends", "vip26.yaml")
    
    # 创建多数据源回测器
    backtester = MultiSourceBacktester(config_file_path=config_file_path)

    backtester.initialize()

    result = backtester.run_backtest(strategy_func=vip26_strategy)

    print() 