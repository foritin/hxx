import polars as pl
import numpy as np
from typing import Optional
from extend.core.backtest_engine import MultiSourceBacktester
from extend.api.strategy_api import StrategyAPI


class StrategyState:
    """策略状态管理类，用于存储策略运行中的状态变量"""

    def __init__(self):
        # Heikin-Ashi相关状态
        self.ha_op_prev: Optional[float] = None
        self.ha_cl_prev: Optional[float] = None

        # 趋势变化状态
        self.trendChange_prev = False

        # SAR相关状态
        self.TbPosition_prev = 1
        self.Transition_prev = 1  # 初始趋势为上升
        self.Af_prev = 0.02
        self.HHValue_prev = None
        self.LLValue_prev = None
        self.SAR_prev = None
        self.ParOpen_prev = None

        # 跟踪止损相关状态
        self.Trailing_Stop_L_prev: Optional[float] = None
        self.Trailing_Stop_H_prev: Optional[float] = None
        self.Trailing_Stop_S_prev: Optional[float] = None
        self.HighestLowAfterEntry: Optional[float] = None
        self.LowestHighAfterEntry: Optional[float] = None
        self.entPrice: Optional[float] = None
        self.entBar: Optional[int] = None


# 全局状态对象
state = StrategyState()


def initialize(api: StrategyAPI):
    """
    VIP24多因子趋势跟踪策略初始化函数

    Args:
        api: 策略API对象
    """
    api.log("=" * 60)
    api.log("VIP24 多因子趋势跟踪策略初始化...")
    api.log("=" * 60)
    api.log("策略逻辑详解:")
    api.log("本策略是一个综合性的趋势跟踪系统，结合了波动率分析、趋势判断和风险管理三大要素。")
    api.log("通过多指标协同验证，旨在捕捉中期趋势行情并有效控制风险。")
    api.log("")
    api.log("核心逻辑分为四个部分：")
    api.log("1. 波动率分析（TVI指标）")
    api.log("2. 趋势判断（SAR指标）")
    api.log("3. 交易信号生成")
    api.log("4. 风险管理")
    api.log("=" * 60)

    # 获取策略参数
    lengs = api.get_param("lengs", 14)
    AfStep = api.get_param("AfStep", 0.02)
    X = api.get_param("X", 20)
    Lots = api.get_param("Lots", 1)
    TrailingStopRate = api.get_param("TrailingStopRate", 50)

    api.log(f"参数设置:")
    api.log(f"- ATR周期 (lengs): {lengs}")
    api.log(f"- SAR加速因子 (AfStep): {AfStep}")
    api.log(f"- 区间周期 (X): {X}")
    api.log(f"- 交易手数 (Lots): {Lots}")
    api.log(f"- 跟踪止损比例 (TrailingStopRate): {TrailingStopRate}")
    api.log("=" * 60)


def calculate_sma(series, period):
    """计算简单移动平均线"""
    return series.rolling_mean(window_size=period)


def calculate_gini_mean_diff(close, lengs):
    """
    计算Gini平均差（波动性指标）

    Args:
        close: 收盘价序列
        lengs: 基础周期

    Returns:
        gini_mean_diff: Gini平均差序列
    """
    # 计算不同周期的简单移动平均线
    sma1 = calculate_sma(close, lengs)  # 短期均线
    sma4 = calculate_sma(close, lengs * 2)  # 中期均线
    sma6 = calculate_sma(close, lengs * 3)  # 长期均线
    sma9 = calculate_sma(close, lengs * 4)  # 超长期均线

    # 计算均线间的绝对差值
    abs_diff_1_4 = (sma1 - sma4).abs()
    abs_diff_1_6 = (sma1 - sma6).abs()
    abs_diff_1_9 = (sma1 - sma9).abs()
    abs_diff_4_6 = (sma4 - sma6).abs()
    abs_diff_4_9 = (sma4 - sma9).abs()
    abs_diff_6_9 = (sma6 - sma9).abs()

    # 计算Gini平均差
    gini_mean_diff = (abs_diff_1_4 + abs_diff_1_6 + abs_diff_1_9 + abs_diff_4_6 + abs_diff_4_9 + abs_diff_6_9) / 6

    return gini_mean_diff


def calculate_tvi_heikin_ashi(TVI, current_idx):
    """
    计算TVI的Heikin-Ashi蜡烛图

    Args:
        TVI: 趋势波动指标序列
        current_idx: 当前索引

    Returns:
        ha_op, ha_cl: Heikin-Ashi开盘价和收盘价
    """
    # TVI蜡烛图计算
    cl = TVI[current_idx] if current_idx < len(TVI) else TVI[-1]
    op = TVI[current_idx - 1] if current_idx > 0 else cl

    # 简化处理：hi和lo直接使用cl和op
    hi = max(op, cl)
    lo = min(op, cl)

    # 计算Heikin-Ashi蜡烛图
    if current_idx == 0:
        ha_cl = (op + hi + lo + cl) / 4
        ha_op = (op + cl) / 2
    else:
        ha_cl = (op + hi + lo + cl) / 4
        # ha_op需要使用前一个值，这里通过全局状态管理

    return ha_op, ha_cl


def calculate_sar(high, low, current_idx, AfStep, AfLimit=0.2):
    """
    计算抛物线转向指标(SAR)

    Args:
        high: 最高价序列
        low: 最低价序列
        current_idx: 当前索引
        AfStep: 加速因子步长
        AfLimit: 加速因子上限

    Returns:
        SAR, TbPosition, Transition, Af, HHValue, LLValue, ParOpen
    """
    global state

    # 初始化阶段 - 检查是否需要初始化
    if current_idx == 0 or state.HHValue_prev is None or state.LLValue_prev is None or state.SAR_prev is None:

        state.TbPosition_prev = 1
        state.Transition_prev = 1  # 初始第一个Bar可以有信号
        state.Af_prev = AfStep
        state.HHValue_prev = high[current_idx]
        state.LLValue_prev = low[current_idx]
        state.SAR_prev = state.LLValue_prev
        state.ParOpen_prev = state.SAR_prev + state.Af_prev * (state.HHValue_prev - state.SAR_prev)

        if state.ParOpen_prev > low[current_idx]:
            state.ParOpen_prev = low[current_idx]

        return (state.SAR_prev, state.TbPosition_prev, state.Transition_prev, state.Af_prev, state.HHValue_prev, state.LLValue_prev, state.ParOpen_prev)

    # 更新最高值和最低值
    HHValue = max(high[current_idx], state.HHValue_prev)
    LLValue = min(low[current_idx], state.LLValue_prev)

    # 初始化返回值
    TbPosition = state.TbPosition_prev
    Transition = 0  # 默认无反转
    Af = state.Af_prev
    SAR = state.SAR_prev
    ParOpen = state.ParOpen_prev

    # 多头持仓处理
    if state.TbPosition_prev == 1:
        if low[current_idx] <= state.ParOpen_prev:
            # 趋势反转：转为空头
            TbPosition = -1
            Transition = -1
            SAR = state.HHValue_prev
            HHValue = high[current_idx]
            LLValue = low[current_idx]
            Af = AfStep
            ParOpen = SAR + Af * (LLValue - SAR)

            if ParOpen < high[current_idx]:
                ParOpen = high[current_idx]
            if current_idx > 0 and ParOpen < high[current_idx - 1]:
                ParOpen = high[current_idx - 1]
        else:
            # 保持多头持仓
            TbPosition = 1
            SAR = state.ParOpen_prev

            if high[current_idx] > state.HHValue_prev and state.Af_prev < AfLimit:
                Af = min(state.Af_prev + AfStep, AfLimit)
            else:
                Af = state.Af_prev

            ParOpen = SAR + Af * (HHValue - SAR)

            if ParOpen > low[current_idx]:
                ParOpen = low[current_idx]
            if current_idx > 0 and ParOpen > low[current_idx - 1]:
                ParOpen = low[current_idx - 1]

    # 空头持仓处理
    else:
        if high[current_idx] >= state.ParOpen_prev:
            # 趋势反转：转为多头
            TbPosition = 1
            Transition = 1
            SAR = state.LLValue_prev
            HHValue = high[current_idx]
            LLValue = low[current_idx]
            Af = AfStep
            ParOpen = SAR + Af * (HHValue - SAR)

            if ParOpen > low[current_idx]:
                ParOpen = low[current_idx]
            if current_idx > 0 and ParOpen > low[current_idx - 1]:
                ParOpen = low[current_idx - 1]
        else:
            # 保持空头持仓
            TbPosition = -1
            SAR = state.ParOpen_prev

            if low[current_idx] < state.LLValue_prev and state.Af_prev < AfLimit:
                Af = min(state.Af_prev + AfStep, AfLimit)
            else:
                Af = state.Af_prev

            ParOpen = SAR + Af * (LLValue - SAR)

            if ParOpen < high[current_idx]:
                ParOpen = high[current_idx]
            if current_idx > 0 and ParOpen < high[current_idx - 1]:
                ParOpen = high[current_idx - 1]

    # 更新状态
    state.TbPosition_prev = TbPosition
    state.Transition_prev = Transition
    state.Af_prev = Af
    state.HHValue_prev = HHValue
    state.LLValue_prev = LLValue
    state.SAR_prev = SAR
    state.ParOpen_prev = ParOpen

    return SAR, TbPosition, Transition, Af, HHValue, LLValue, ParOpen


def vip24_strategy(api: StrategyAPI):
    """
    VIP24多因子趋势跟踪策略主函数

    该策略结合了波动率分析、趋势判断和风险管理三大要素：
    1. 波动率分析：通过TVI指标判断市场波动性
    2. 趋势判断：使用SAR指标确定趋势方向
    3. 交易信号：三重确认机制生成交易信号
    4. 风险管理：动态跟踪止损机制
    """
    # 数据验证
    close = api.get_close()
    if close is None or len(close) == 0:
        api.log("警告: 数据为空")
        return

    current_idx = api.get_idx()
    if current_idx < 1:
        return

    # 获取策略参数
    lengs = api.get_param("lengs", 14)
    AfStep = api.get_param("AfStep", 0.02)
    X = api.get_param("X", 20)
    Lots = api.get_param("Lots", 1)  # 固定手数
    TrailingStopRate = api.get_param("TrailingStopRate", 50)
    AfLimit = api.get_param("AfLimit", 0.2)
    order_percent = api.get_param("OrderPercent", 0.2)

    # 获取价格数据
    open_p = api.get_open()
    high = api.get_high()
    low = api.get_low()

    # 确保有足够的数据
    min_required_bars = max(lengs * 4, X) + 5

    # 1. 计算TVI指标（趋势波动指标）
    gini_mean_diff = calculate_gini_mean_diff(close, lengs)
    TVI = gini_mean_diff

    # 2. 初始化检查 - 如果状态变量未初始化，先进行初始化
    if state.ha_op_prev is None or state.ha_cl_prev is None:
        # 初始化HA值
        cl = TVI[current_idx] if TVI[current_idx] else 0
        op = TVI[current_idx - 1] if current_idx > 0 and TVI[current_idx - 1] else cl

        state.ha_cl_prev = (op + cl + cl + cl) / 4  # 简化：hi=lo=cl
        state.ha_op_prev = (op + cl) / 2
        state.trendChange_prev = False

        api.log(f"Bar {current_idx}: 策略状态初始化完成")

    # 如果数据不足，继续等待
    if current_idx < min_required_bars:
        if current_idx == lengs * 4:  # 只在第一次打印
            api.log(f"数据准备中，需要至少 {min_required_bars} 根K线")
        return

    # 3. 获取前一个Bar的状态值（用于信号判断）
    prev_transition = state.Transition_prev
    prev_trend_change = state.trendChange_prev

    # 4. 计算当前Bar的TVI HA蜡烛图（修正版本）
    cl = TVI[current_idx] if TVI[current_idx] else 0
    op = TVI[current_idx - 1] if current_idx > 0 and TVI[current_idx - 1] else cl

    # 修正：计算TVI的上下限（原始代码中的关键逻辑）
    lowerTVI = np.round(cl, 0)  # Round(TVI,0)
    upperTVI = np.round(cl, 0)  # Round(TVI, 0)

    # 修正：正确计算HA蜡烛图的hi和lo
    hi = max(op, max(upperTVI, cl))
    lo = min(op, min(lowerTVI, cl))

    ha_op = (state.ha_op_prev + state.ha_cl_prev) / 2
    ha_cl = (op + hi + lo + cl) / 4  # 使用正确的OHLC

    # 5. 判断趋势变化
    if ha_cl > ha_op:
        trendChange = True
    elif ha_cl < ha_op:
        trendChange = False
    else:
        trendChange = state.trendChange_prev

        # 暂时不更新状态，等信号判断完成后再更新

    # 6. 计算当前Bar的SAR指标
    SAR, TbPosition, Transition, Af, HHValue, LLValue, ParOpen = calculate_sar(high, low, current_idx, AfStep, AfLimit)

    # 7. 计算最高价/最低价区间
    HH = high.rolling_max(window_size=X)[current_idx]
    LL = low.rolling_min(window_size=X)[current_idx]
    HH_prev = high.rolling_max(window_size=X)[current_idx - 1] if current_idx > 0 else HH
    LL_prev = low.rolling_min(window_size=X)[current_idx - 1] if current_idx > 0 else LL

    # 获取当前持仓
    current_pos = api.get_pos()

    # 开多条件：前一Bar的SAR转多 + 前一Bar的TVI趋势变化 + 当前Bar突破前高
    if prev_transition == 1 and prev_trend_change and high[current_idx] >= HH_prev and current_pos == 0:

        entry_price = max(open_p[current_idx], HH_prev)
        api.buy(datasource_mark="solusdt_5m", order_percent=order_percent, order_type="next_bar_open")
        api.log(f"开多: {Lots}手 @ {entry_price:.2f} (前Bar SAR转多+TVI变化+突破前高)")

        # 初始化跟踪止损
        state.HighestLowAfterEntry = open_p[current_idx]
        state.entPrice = open_p[current_idx]
        state.entBar = current_idx

    # 开空条件：前一Bar的SAR转空 + 前一Bar的TVI趋势变化 + 当前Bar突破前低
    elif prev_transition == -1 and prev_trend_change and low[current_idx] <= LL_prev and current_pos == 0:

        entry_price = min(open_p[current_idx], LL_prev)
        api.sellshort(volume=Lots, order_type="next_bar_open")
        api.log(f"开空: {Lots}手 @ {entry_price:.2f} (前Bar SAR转空+TVI变化+突破前低)")

        # 初始化跟踪止损
        state.LowestHighAfterEntry = open_p[current_idx]
        state.entPrice = open_p[current_idx]
        state.entBar = current_idx

    # 7. 更新状态变量（在信号判断完成后）
    state.ha_op_prev = ha_op
    state.ha_cl_prev = ha_cl
    state.trendChange_prev = trendChange
    # SAR状态在calculate_sar函数中已经更新

    # 8. 跟踪止损逻辑（修正：使用前一Bar的止损线进行比较）
    if current_pos > 0 and state.entBar is not None and current_idx > state.entBar:
        # 多头跟踪止损
        # 首先更新最高最低价
        if current_idx == state.entBar:
            state.HighestLowAfterEntry = low[current_idx]
        else:
            state.HighestLowAfterEntry = max(state.HighestLowAfterEntry, low[current_idx])

        # 计算当前Bar的止损线
        Trailing_Stop_L = state.HighestLowAfterEntry - open_p[current_idx] * TrailingStopRate / 1000

        # 关键修正：如果前一个Bar有止损线，使用前一个Bar的止损线进行比较
        if hasattr(state, "Trailing_Stop_L_prev") and current_idx > state.entBar:
            if low[current_idx] <= state.Trailing_Stop_L_prev:
                exit_price = min(open_p[current_idx], state.Trailing_Stop_L_prev)
                api.sell(volume=Lots, order_type="next_bar_open")
                api.log(f"多头止损: {Lots}手 @ {exit_price:.2f} (止损线: {state.Trailing_Stop_L_prev:.2f})")

                # 重置状态
                state.entPrice = None
                state.entBar = None
                delattr(state, "Trailing_Stop_L_prev")

        # 保存当前止损线供下一个Bar使用
        state.Trailing_Stop_L_prev = Trailing_Stop_L

    elif current_pos < 0 and state.entBar is not None and current_idx > state.entBar:
        # 空头跟踪止损
        # 首先更新最低最高价
        if current_idx == state.entBar:
            state.LowestHighAfterEntry = high[current_idx]
        else:
            state.LowestHighAfterEntry = min(state.LowestHighAfterEntry, high[current_idx])

        # 计算当前Bar的止损线
        Trailing_Stop_S = state.LowestHighAfterEntry + open_p[current_idx] * TrailingStopRate / 1000

        # 关键修正：如果前一个Bar有止损线，使用前一个Bar的止损线进行比较
        if hasattr(state, "Trailing_Stop_S_prev") and current_idx > state.entBar:
            if high[current_idx] >= state.Trailing_Stop_S_prev:
                exit_price = max(open_p[current_idx], state.Trailing_Stop_S_prev)
                api.buycover(volume=Lots, order_type="next_bar_open")
                api.log(f"空头止损: {Lots}手 @ {exit_price:.2f} (止损线: {state.Trailing_Stop_S_prev:.2f})")

                # 重置状态
                state.entPrice = None
                state.entBar = None
                delattr(state, "Trailing_Stop_S_prev")

        # 保存当前止损线供下一个Bar使用
        state.Trailing_Stop_S_prev = Trailing_Stop_S

    # 8.1. 清理无持仓时的止损状态
    if current_pos == 0:
        # 清理跟踪止损的状态变量
        if hasattr(state, "Trailing_Stop_L_prev"):
            delattr(state, "Trailing_Stop_L_prev")
        if hasattr(state, "Trailing_Stop_S_prev"):
            delattr(state, "Trailing_Stop_S_prev")

    # 9. 定期打印状态信息（每100个bar打印一次）
    if current_idx % 100 == 0:
        api.log(f"Bar {current_idx}: SAR={SAR:.4f}, PrevTransition={prev_transition}, " f"PrevTrendChange={prev_trend_change}, TVI={cl:.6f}, 持仓={current_pos}")
        api.log(f"HH_prev={HH_prev:.2f}, LL_prev={LL_prev:.2f}, High={high[current_idx]:.2f}, Low={low[current_idx]:.2f}")

    # 10. 记录所有交易信号（无论是否开仓）
    if prev_transition == 1 and prev_trend_change and high[current_idx] >= HH_prev:
        if current_pos == 0:
            # 已经在上面记录了开仓日志
            pass
        else:
            api.log(f"多头信号触发但已有持仓: Transition={prev_transition}, TrendChange={prev_trend_change}, H>HH_prev")

    elif prev_transition == -1 and prev_trend_change and low[current_idx] <= LL_prev:
        if current_pos == 0:
            # 已经在上面记录了开仓日志
            pass
        else:
            api.log(f"空头信号触发但已有持仓: Transition={prev_transition}, TrendChange={prev_trend_change}, L<LL_prev")


if __name__ == "__main__":
    from extend.utils import PathTools

    strategy_path = PathTools.get_strategy_path()
    config_file_path = PathTools.combine_path(strategy_path, "trends", "vip24.yaml")
    # 创建多数据源回测器
    backtester = MultiSourceBacktester(config_file_path=config_file_path)

    backtester.initialize()

    backtester.run_backtest(strategy_func=vip24_strategy)

    # 运行回测
    # results = backtester.run_backtest(strategy_func=vip24_strategy, strategy_params=strategy_params)
