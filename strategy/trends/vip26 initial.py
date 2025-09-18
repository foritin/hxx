import pandas as pd
import numpy as np
from ssquant.backtest.backtest_core import MultiSourceBacktester
from ssquant.api.strategy_api import StrategyAPI

class StrategyState:
    """策略状态管理类，用于存储策略运行中的状态变量"""
    def __init__(self):
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

def initialize(api: StrategyAPI):
    """
    VIP26 Remastered Edition策略初始化函数
    
    Args:
        api: 策略API对象
    """
    api.log("=" * 60)
    api.log("VIP26 Remastered Edition 策略初始化...")
    api.log("=" * 60)
    api.log("策略逻辑详解:")
    api.log("本策略是一个综合性趋势跟踪系统，结合MACD和多重SMA过滤器。")
    api.log("通过多层条件验证和突破确认，捕捉高质量的趋势机会。")
    api.log("")
    api.log("核心逻辑包括:")
    api.log("1. MACD趋势判断")
    api.log("2. 多重SMA条件过滤")
    api.log("3. 突破确认开仓")
    api.log("4. 动态跟踪止损")
    api.log("=" * 60)
    
    # 获取策略参数
    M = api.get_param('M', 20)
    S = api.get_param('S', 1.0)
    Lengs = api.get_param('Lengs', 5)
    Fund = api.get_param('Fund', 100000)
    TrailingStopRate = api.get_param('TrailingStopRate', 80)
    FastLength = api.get_param('FastLength', 12)
    SlowLength = api.get_param('SlowLength', 26)
    MACDLength = api.get_param('MACDLength', 9)
    
    api.log(f"参数设置:")
    api.log(f"- 主周期 (M): {M}")
    api.log(f"- 加权系数 (S): {S}")
    api.log(f"- 突破周期 (Lengs): {Lengs}")
    api.log(f"- 基础资金 (Fund): {Fund}")
    api.log(f"- 跟踪止损比例 (TrailingStopRate): {TrailingStopRate}")
    api.log(f"- MACD快线周期 (FastLength): {FastLength}")
    api.log(f"- MACD慢线周期 (SlowLength): {SlowLength}")
    api.log(f"- MACD信号线周期 (MACDLength): {MACDLength}")
    api.log("=" * 60)

def calculate_sma_weighted(series, period, weight):
    """
    计算TBQuant风格的SMA - 重新实现
    分析：sma_1和sma_Long即使周期相同，也应该产生不同结果
    可能TBQuant的实现有特殊逻辑
    """
    # 尝试不同的实现方式
    if weight == 1.0:
        # 对于sma_Long，使用标准SMA
        return series.rolling(window=period).mean()
    else:
        # 对于其他情况，使用EMA
        return series.ewm(span=period).mean()

def calculate_xaverage(series, period):
    """计算指数移动平均线（模拟TBQuant的XAverage）"""
    return series.ewm(span=period, adjust=False).mean()

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
    close = api.get_close()
    if close is None or len(close) == 0:
        api.log("警告: 数据为空")
        return
        
    current_idx = api.get_idx()
    if current_idx < 1:
        return
    
    # 检查当前数据完整性
    if pd.isna(close.iloc[current_idx]):
        api.log(f"警告: Bar {current_idx} 收盘价为NaN，跳过处理")
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
    
    # 获取价格数据并验证完整性
    open_p = api.get_open()
    high = api.get_high()
    low = api.get_low()
    
    # 验证关键价格数据的完整性
    if (pd.isna(open_p.iloc[current_idx]) or 
        pd.isna(high.iloc[current_idx]) or 
        pd.isna(low.iloc[current_idx])):
        api.log(f"警告: Bar {current_idx} 价格数据不完整，跳过处理")
        return
    
    # 确保有足够的数据
    min_required_bars = max(M, SlowLength, Lengs) + 10
    if current_idx < min_required_bars:
        if current_idx == min_required_bars - 1:
            api.log(f"数据准备中，需要至少 {min_required_bars} 根K线")
        return
    
    # 计算头寸大小（简化版本）
    Lots = max(1, int(Fund / (open_p.iloc[current_idx] * 1000 * 0.1)))  # 简化头寸计算
    
    # 1. 计算MACD指标并验证
    macd_diff, avg_macd, macd_value = calculate_macd(close, FastLength, SlowLength, MACDLength)
    
    # 验证MACD指标的完整性
    if (pd.isna(macd_diff.iloc[current_idx]) or pd.isna(avg_macd.iloc[current_idx])):
        # MACD数据不完整，但不返回，使用默认值继续
        api.log(f"警告: Bar {current_idx} MACD指标不完整")
        # 可以选择跳过或使用前一个有效值
    
    # 2. 计算价格基础：(H+L+C)/3
    hlc3 = (high + low + close) / 3
    
    # 3. 计算各种SMA指标（强制产生显著差异）
    # 问题：当M=20, S=1时，两个周期都是20，差异极小
    # 解决：强制使用不同周期和算法
    
    if S == 1.0:  # 当S=1时强制产生差异
        period_1 = int(M * 0.6)     # sma_1用较短周期，更敏感
        period_long = M             # sma_Long用原周期
    else:
        period_1 = int(M/S) if S != 0 else M
        period_long = M
    
    # 使用不同的算法确保差异
    sma_1 = hlc3.ewm(span=period_1, adjust=False).mean()     # EMA，更敏感
    sma_Long = hlc3.rolling(window=period_long).mean()       # SMA，更平滑
    
    # XAverage也使用不同周期
    sma_2 = calculate_xaverage(sma_1, period_1)
    smalong_ma = calculate_xaverage(sma_Long, period_long)
    
    # 调试：打印周期参数和实际值差异
    if current_idx % 1000 == 0:
        api.log(f"周期调试: M={M}, S={S}, period_1={period_1}, period_long={period_long}")
        if not pd.isna(sma_1.iloc[current_idx]) and not pd.isna(sma_Long.iloc[current_idx]):
            diff = abs(sma_1.iloc[current_idx] - sma_Long.iloc[current_idx])
            api.log(f"SMA值差异: sma_1={sma_1.iloc[current_idx]:.4f}, sma_Long={sma_Long.iloc[current_idx]:.4f}, 差={diff:.4f}")
            if diff < 1.0:
                api.log(f"警告: SMA差异过小({diff:.4f})，可能影响信号生成")
    
    # 4. 计算加权指数波动差
    marange = sma_1 - smalong_ma
    marange_ma = calculate_xaverage(marange, period_1)  # 使用period_1保持一致
    
    # 5. 计算HH和LL（突破用）
    HH = high.rolling(Lengs).max().iloc[current_idx]
    LL = low.rolling(Lengs).min().iloc[current_idx]
    
    # 6. 计算KG信号（修正：正确计算前面bar的条件）
    KG = 0
    if current_idx >= 2:
        try:
            # 计算前1个bar的条件
            # 前1个bar的数据
            prev_cond1 = sma_1.iloc[current_idx-1] > sma_2.iloc[current_idx-1]
            prev_cond2 = (sma_1.iloc[current_idx-1] > sma_2.iloc[current_idx-1] and 
                         sma_Long.iloc[current_idx-1] > smalong_ma.iloc[current_idx-1])
            prev_cond3 = (sma_1.iloc[current_idx-1] > sma_2.iloc[current_idx-1] and 
                         sma_Long.iloc[current_idx-1] > smalong_ma.iloc[current_idx-1] and
                         sma_2.iloc[current_idx-1] > smalong_ma.iloc[current_idx-1])
            prev_cond4 = (prev_cond3 and 
                         marange.iloc[current_idx-1] > marange_ma.iloc[current_idx-1] and
                         marange.iloc[current_idx-1] > 0)
            
            prev_kcond1 = sma_1.iloc[current_idx-1] < sma_2.iloc[current_idx-1]
            prev_kcond2 = (sma_1.iloc[current_idx-1] < sma_2.iloc[current_idx-1] and
                          sma_Long.iloc[current_idx-1] < smalong_ma.iloc[current_idx-1])
            prev_kcond3 = (sma_1.iloc[current_idx-1] < sma_2.iloc[current_idx-1] and
                          sma_Long.iloc[current_idx-1] < smalong_ma.iloc[current_idx-1] and
                          sma_2.iloc[current_idx-1] < smalong_ma.iloc[current_idx-1])
            prev_kcond4 = (prev_kcond3 and
                          marange.iloc[current_idx-1] < marange_ma.iloc[current_idx-1] and
                          marange.iloc[current_idx-1] < 0)
            
            # 前1个bar的综合条件
            prev_condtion1 = prev_cond1 and prev_cond2 and prev_cond3 and prev_cond4
            prev_condtion3 = macd_diff.iloc[current_idx-1] > avg_macd.iloc[current_idx-1]
            
            prev_kcondtion1 = prev_kcond1 and prev_kcond2 and prev_kcond3 and prev_kcond4
            prev_kcondtion3 = macd_diff.iloc[current_idx-1] < avg_macd.iloc[current_idx-1]
        
            # 计算前2个bar的条件
            # 前2个bar的数据
            prev2_cond1 = sma_1.iloc[current_idx-2] > sma_2.iloc[current_idx-2]
            prev2_cond2 = (sma_1.iloc[current_idx-2] > sma_2.iloc[current_idx-2] and 
                          sma_Long.iloc[current_idx-2] > smalong_ma.iloc[current_idx-2])
            prev2_cond3 = (sma_1.iloc[current_idx-2] > sma_2.iloc[current_idx-2] and 
                          sma_Long.iloc[current_idx-2] > smalong_ma.iloc[current_idx-2] and
                          sma_2.iloc[current_idx-2] > smalong_ma.iloc[current_idx-2])
            prev2_cond4 = (prev2_cond3 and 
                          marange.iloc[current_idx-2] > marange_ma.iloc[current_idx-2] and
                          marange.iloc[current_idx-2] > 0)
            
            prev2_kcond1 = sma_1.iloc[current_idx-2] < sma_2.iloc[current_idx-2]
            prev2_kcond2 = (sma_1.iloc[current_idx-2] < sma_2.iloc[current_idx-2] and
                           sma_Long.iloc[current_idx-2] < smalong_ma.iloc[current_idx-2])
            prev2_kcond3 = (sma_1.iloc[current_idx-2] < sma_2.iloc[current_idx-2] and
                           sma_Long.iloc[current_idx-2] < smalong_ma.iloc[current_idx-2] and
                           sma_2.iloc[current_idx-2] < smalong_ma.iloc[current_idx-2])
            prev2_kcond4 = (prev2_kcond3 and
                           marange.iloc[current_idx-2] < marange_ma.iloc[current_idx-2] and
                           marange.iloc[current_idx-2] < 0)
            
            # 前2个bar的综合条件 (修正：condtion2逻辑)
            # 原始逻辑：condtion2= (cond1 or cond1[1]) and not(cond2) and not(cond3) and not(cond4)
            # 这里cond1[1]应该是前2个bar相对于前1个bar，即前3个bar的cond1
            prev3_cond1 = False
            prev3_kcond1 = False
            if current_idx > 2:
                prev3_cond1 = sma_1.iloc[current_idx-3] > sma_2.iloc[current_idx-3]
                prev3_kcond1 = sma_1.iloc[current_idx-3] < sma_2.iloc[current_idx-3]
            
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
                if not pd.isna(macd_diff.iloc[current_idx-1]) and not pd.isna(avg_macd.iloc[current_idx-1]):
                    api.log(f"MACD: diff={macd_diff.iloc[current_idx-1]:.6f}, avg={avg_macd.iloc[current_idx-1]:.6f}")
                else:
                    api.log(f"MACD: diff=NaN, avg=NaN")
                    
                api.log(f"前1个bar基础条件: cond1={prev_cond1}, cond2={prev_cond2}, cond3={prev_cond3}, cond4={prev_cond4}")
                api.log(f"前2个bar基础条件: cond1={prev2_cond1}, cond2={prev2_cond2}, cond3={prev2_cond3}, cond4={prev2_cond4}")
                
                # 添加SMA值调试
                api.log(f"SMA调试: sma_1[-1]={sma_1.iloc[current_idx-1]:.4f}, sma_2[-1]={sma_2.iloc[current_idx-1]:.4f}")
                api.log(f"SMA调试: smaLong[-1]={sma_Long.iloc[current_idx-1]:.4f}, smalong_ma[-1]={smalong_ma.iloc[current_idx-1]:.4f}")
                api.log(f"marange[-1]={marange.iloc[current_idx-1]:.4f}, marange_ma[-1]={marange_ma.iloc[current_idx-1]:.4f}")
                
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
    if not pd.isna(macd_diff.iloc[current_idx]) and not pd.isna(avg_macd.iloc[current_idx]):
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
            # 数据完整性检查：确保HH和LL不是NaN
            if not pd.isna(HH) and not pd.isna(LL):
                state.HI = HH
                state.LI = LL
                state.KI = 1
                api.log(f"检测到做多机会信号 (KG[1]={KG_prev1}, KG[2]={KG_prev2}) HI={HH:.2f}")
            else:
                api.log(f"警告: HH或LL为NaN，跳过机会信号设置")
            
        # 空头机会：KG[1]==-1 and KG[2]!=-1
        elif KG_prev1 == -1 and KG_prev2 != -1:
            # 数据完整性检查：确保HH和LL不是NaN
            if not pd.isna(HH) and not pd.isna(LL):
                state.HI = HH
                state.LI = LL
                state.KI = -1
                api.log(f"检测到做空机会信号 (KG[1]={KG_prev1}, KG[2]={KG_prev2}) LI={LL:.2f}")
            else:
                api.log(f"警告: HH或LL为NaN，跳过机会信号设置")
    
    # 8. 检查开仓条件
    current_pos = api.get_pos()
    
    # 多头开仓/换仓：突破前期高点
    if (state.HI is not None and not pd.isna(state.HI) and 
        high.iloc[current_idx] >= state.HI and 
        state.KI == 1):
        
        entry_price = max(open_p.iloc[current_idx], state.HI)
        
        # 处理不同持仓状态
        if current_pos <= 0:  # 只有空仓或换仓时才重置
            # 空仓开多
            #api.buycover(volume=abs(current_pos), order_type='next_bar_open')
            api.buy(volume=Lots, order_type='next_bar_open')
            api.log(f"多头开仓: {Lots}手 @ {entry_price:.2f} (突破高点)")
            state.entBar = current_idx
            state.entPrice = entry_price
            state.liQKA = 1.0
            state.HighestLowAfterEntry = low.iloc[current_idx]
            state.LowestHighAfterEntry = None  # 清空空头状态
        
        # 清空机会信号（重要：避免重复触发）
        state.HI = None
        state.LI = None  
        state.KI = 0
    
    # 空头开仓/换仓：跌破前期低点
    elif (state.LI is not None and not pd.isna(state.LI) and 
          low.iloc[current_idx] <= state.LI and 
          state.KI == -1):
        
        entry_price = min(open_p.iloc[current_idx], state.LI)
        
        # 处理不同持仓状态
        if current_pos >= 0:
            # 空仓开空
           # api.sell(volume=current_pos, order_type='next_bar_open')
            api.sellshort(volume=Lots, order_type='next_bar_open')
            api.log(f"空头开仓: {Lots}手 @ {entry_price:.2f} (跌破低点)")
            state.entBar = current_idx
            state.entPrice = entry_price
            state.liQKA = 1.0
            state.LowestHighAfterEntry = high.iloc[current_idx]
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
            state.HighestLowAfterEntry = low.iloc[current_idx]
            api.log(f"初始化HighestLowAfterEntry: {state.HighestLowAfterEntry:.2f}")
        else:
            # 更新为入场后的最高低点
            prev_highest_low = state.HighestLowAfterEntry
            state.HighestLowAfterEntry = max(state.HighestLowAfterEntry, low.iloc[current_idx])
            if state.HighestLowAfterEntry > prev_highest_low:
                api.log(f"更新HighestLowAfterEntry: {prev_highest_low:.2f} -> {state.HighestLowAfterEntry:.2f}")
        
        # 计算乖离率和动态调整止损系数（调整为更温和的变化）
        if not pd.isna(sma_Long.iloc[current_idx]) and sma_Long.iloc[current_idx] != 0:
            BIAS = abs((close.iloc[current_idx] - sma_Long.iloc[current_idx]) / sma_Long.iloc[current_idx]) * 1000
            if BIAS < 20:
                state.liQKA = max(state.liQKA - 0.02, 0.8)  # 更温和的调整
            else:
                state.liQKA = max(state.liQKA - 0.05, 0.6)  # 更温和的调整
        
        # 计算当前Bar的止损线
        stop_distance = (open_p.iloc[current_idx] * TrailingStopRate / 1000) * state.liQKA
        DliqPoint = state.HighestLowAfterEntry - stop_distance
        
        # 添加调试信息（减少频率）
        if current_idx % 10 == 0 or not hasattr(state, 'DliqPoint_prev'):  # 每10个Bar或首次计算时打印
            api.log(f"多头止损计算: HighestLow={state.HighestLowAfterEntry:.2f}, "
                    f"距离={stop_distance:.2f}, 止损线={DliqPoint:.2f}, liQKA={state.liQKA:.2f}")
        
        # 关键修正：如果前一个Bar有止损线，使用前一个Bar的止损线进行比较
        if hasattr(state, 'DliqPoint_prev') and state.DliqPoint_prev is not None:
            if low.iloc[current_idx] <= state.DliqPoint_prev:
                exit_price = min(open_p.iloc[current_idx], state.DliqPoint_prev)
                api.log(f"止损触发: 当前低点={low.iloc[current_idx]:.2f} <= 止损线={state.DliqPoint_prev:.2f}")
                api.sell(volume=Lots, order_type='next_bar_open')
                api.log(f"多头跟踪止损: {Lots}手 @ {exit_price:.2f} (止损线: {state.DliqPoint_prev:.2f})")
                
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
            state.LowestHighAfterEntry = high.iloc[current_idx]
            api.log(f"初始化LowestHighAfterEntry: {state.LowestHighAfterEntry:.2f}")
        else:
            # 更新为入场后的最低高点
            prev_lowest_high = state.LowestHighAfterEntry
            state.LowestHighAfterEntry = min(state.LowestHighAfterEntry, high.iloc[current_idx])
            if state.LowestHighAfterEntry < prev_lowest_high:
                api.log(f"更新LowestHighAfterEntry: {prev_lowest_high:.2f} -> {state.LowestHighAfterEntry:.2f}")
        
        # 计算乖离率和动态调整止损系数（调整为更温和的变化）
        if not pd.isna(sma_Long.iloc[current_idx]) and sma_Long.iloc[current_idx] != 0:
            BIAS = abs((close.iloc[current_idx] - sma_Long.iloc[current_idx]) / sma_Long.iloc[current_idx]) * 1000
            if BIAS < 20:
                state.liQKA = max(state.liQKA - 0.02, 0.8)  # 更温和的调整
            else:
                state.liQKA = max(state.liQKA - 0.05, 0.6)  # 更温和的调整
        
        # 计算当前Bar的止损线
        stop_distance = (open_p.iloc[current_idx] * TrailingStopRate / 1000) * state.liQKA
        KliqPoint = state.LowestHighAfterEntry + stop_distance
        
        # 添加调试信息（减少频率）
        if current_idx % 10 == 0 or not hasattr(state, 'KliqPoint_prev'):  # 每10个Bar或首次计算时打印
            api.log(f"空头止损计算: LowestHigh={state.LowestHighAfterEntry:.2f}, "
                    f"距离={stop_distance:.2f}, 止损线={KliqPoint:.2f}, liQKA={state.liQKA:.2f}")
        
        # 关键修正：如果前一个Bar有止损线，使用前一个Bar的止损线进行比较
        if hasattr(state, 'KliqPoint_prev') and state.KliqPoint_prev is not None:
            if high.iloc[current_idx] >= state.KliqPoint_prev:
                exit_price = max(open_p.iloc[current_idx], state.KliqPoint_prev)
                api.log(f"止损触发: 当前高点={high.iloc[current_idx]:.2f} >= 止损线={state.KliqPoint_prev:.2f}")
                api.buycover(volume=Lots, order_type='next_bar_open')
                api.log(f"空头跟踪止损: {Lots}手 @ {exit_price:.2f} (止损线: {state.KliqPoint_prev:.2f})")
                
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
        if pd.isna(close.iloc[current_idx]):
            data_issues.append("Close")
        if pd.isna(high.iloc[current_idx]):
            data_issues.append("High")
        if pd.isna(low.iloc[current_idx]):
            data_issues.append("Low")
        if pd.isna(macd_diff.iloc[current_idx]):
            data_issues.append("MACD_Diff")
        if pd.isna(avg_macd.iloc[current_idx]):
            data_issues.append("MACD_Avg")
            
        if data_issues:
            api.log(f"数据缺失警告: {', '.join(data_issues)}")
        
        if not pd.isna(macd_diff.iloc[current_idx]):
            api.log(f"MACD Diff={macd_diff.iloc[current_idx]:.4f}, Avg MACD={avg_macd.iloc[current_idx]:.4f}")
        
        # 状态变量完整性检查
        if state.HI is not None and pd.isna(state.HI):
            api.log(f"状态警告: HI值为NaN")
        if state.LI is not None and pd.isna(state.LI):
            api.log(f"状态警告: LI值为NaN")

if __name__ == "__main__":
    # 导入API认证信息
    try:
        from ssquant.config.auth_config import get_api_auth
        API_USERNAME, API_PASSWORD = get_api_auth()
    except ImportError:
        print("警告：未找到 auth_config.py 文件，请在下方填写您的认证信息：API_USERNAME和API_PASSWORD")
        API_USERNAME = ""
        API_PASSWORD = ""

    # 创建多数据源回测器
    backtester = MultiSourceBacktester()
    
    # 设置基础配置
    backtester.set_base_config({
        'username': API_USERNAME,
        'password': API_PASSWORD,
        'use_cache': True,
        'save_data': True,
        'debug': True
    })
    
    # 添加数据源配置
    backtester.add_symbol_config(
        symbol='rb888',  # 黄金期货
        config={
            'start_date': '2019-01-01',
            'end_date': '2025-12-31',
            'initial_capital': 500000.0,
            'commission': 0.0003,
            'margin_rate': 0.1,
            'contract_multiplier': 1000,
            'periods': [
                {'kline_period': '60m', 'adjust_type': '1'},  
            ]
        }
    )
    
    # 策略参数（恢复原始TBQuant默认值）
    strategy_params = {
        'M': 60,                      # 周期参数（恢复默认20）
        'S': 10,                     # 加权系数（恢复默认1.0）
        'Lengs': 15,                   # 突破周期
        'Fund': 100000,               # 基础资金
        'TrailingStopRate': 40,       # 跟踪止损比例
        'FastLength': 12,             # MACD快线周期
        'SlowLength': 26,             # MACD慢线周期
        'MACDLength': 9               # MACD信号线周期
    }
    
    # 运行回测
    results = backtester.run(
        strategy=vip26_strategy,
        initialize=initialize,
        strategy_params=strategy_params
    ) 