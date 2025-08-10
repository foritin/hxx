import pandas as pd
import numpy as np
from extend.core.backtest_engine import MultiSourceBacktester
from extend.api.strategy_api import StrategyAPI

class StrategyState:
    def __init__(self):
        self.ha_op_prev = None
        self.ha_cl_prev = None
        self.trendChange_prev = False
        self.TbPosition_prev = 1
        self.Transition_prev = 1
        self.Af_prev = 0.02
        self.HHValue_prev = None
        self.LLValue_prev = None
        self.SAR_prev = None
        self.ParOpen_prev = None
        self.HighestLowAfterEntry = None
        self.LowestHighAfterEntry = None
        self.entPrice = None
        self.entBar = None

state = StrategyState()

def initialize(api: StrategyAPI):
    """VIP24多因子趋势跟踪策略初始化函数"""

def strategy_function(api: StrategyAPI):
    """VIP24多因子趋势跟踪策略主函数"""
    # 获取参数
    lengs = api.get_param('lengs', 14)
    AfStep = api.get_param('AfStep', 0.02)
    X = api.get_param('X', 20)
    Lots = api.get_param('Lots', 1)
    TrailingStopRate = api.get_param('TrailingStopRate', 50)
    AfLimit = api.get_param('AfLimit', 0.2) 
    
    
    # 数据验证
    close = api.get_close()
    if close is None or len(close) == 0:
        api.log("警告: 数据为空")
        return
        
    current_idx = api.get_idx()
    if current_idx < X:  # 确保至少有X个数据点
        return
    
    
    
    # 获取价格数据
    open_p = api.get_open()
    high = api.get_high()
    low = api.get_low()
    
    # 初始化全局变量
    if current_idx == X:
        state.ha_op_prev = (open_p[0] + close[0]) / 2
        state.ha_cl_prev = (open_p[0] + high[0] + low[0] + close[0]) / 4
        state.trendChange_prev = False
        state.TbPosition_prev = 1
        state.Transition_prev = 1
        state.Af_prev = AfStep
        state.HHValue_prev = high[0]
        state.LLValue_prev = low[0]
        state.SAR_prev = state.LLValue_prev
        state.ParOpen_prev = state.SAR_prev + state.Af_prev * (state.HHValue_prev - state.SAR_prev)
        state.HighestLowAfterEntry = low[0]
        state.LowestHighAfterEntry = high[0]
        state.entPrice = open_p[0]
        state.entBar = 0
        return
    
    # 计算Heikin-Ashi蜡烛图
    if state.ha_op_prev is None or state.ha_cl_prev is None:
        state.ha_op_prev = (open_p[current_idx-1] + close[current_idx-1]) / 2
        state.ha_cl_prev = (open_p[current_idx-1] + high[current_idx-1] + 
                           low[current_idx-1] + close[current_idx-1]) / 4
    
    ha_op = (state.ha_op_prev + state.ha_cl_prev) / 2
    ha_cl = (open_p[current_idx] + high[current_idx] + 
            low[current_idx] + close[current_idx]) / 4
    
    # 存储当前HA值
    state.ha_op_prev = ha_op
    state.ha_cl_prev = ha_cl
    
    # 判断趋势变化
    trendChange = (ha_cl > ha_op) if (ha_cl != ha_op) else state.trendChange_prev
    state.trendChange_prev = trendChange
    
    # 计算SAR指标 - 修复变量作用域问题
    HHValue = max(high[current_idx], state.HHValue_prev)
    LLValue = min(low[current_idx], state.LLValue_prev)
    
    # 初始化SAR计算变量
    TbPosition = state.TbPosition_prev
    Transition = state.Transition_prev
    Af = state.Af_prev
    SAR = state.SAR_prev
    ParOpen = state.ParOpen_prev
    
    if state.TbPosition_prev == 1:  # 多头处理
        if low[current_idx] <= state.ParOpen_prev:
            TbPosition = -1
            Transition = -1
            SAR = state.HHValue_prev
            Af = AfStep
            ParOpen = SAR + Af * (LLValue - SAR)
        else:
            TbPosition = 1
            SAR = state.ParOpen_prev
            if high[current_idx] > state.HHValue_prev and state.Af_prev < AfLimit:
                Af = min(state.Af_prev + AfStep, AfLimit)
            else:
                Af = state.Af_prev
            ParOpen = SAR + Af * (HHValue - SAR)
    else:  # 空头处理
        if high[current_idx] >= state.ParOpen_prev:
            TbPosition = 1
            Transition = 1
            SAR = state.LLValue_prev
            Af = AfStep
            ParOpen = SAR + Af * (HHValue - SAR)
        else:
            TbPosition = -1
            SAR = state.ParOpen_prev
            if low[current_idx] < state.LLValue_prev and state.Af_prev < AfLimit:
                Af = min(state.Af_prev + AfStep, AfLimit)
            else:
                Af = state.Af_prev
            ParOpen = SAR + Af * (LLValue - SAR)
    
    # 存储当前SAR值
    state.TbPosition_prev = TbPosition
    state.Transition_prev = Transition
    state.Af_prev = Af
    state.HHValue_prev = HHValue
    state.LLValue_prev = LLValue
    state.SAR_prev = SAR
    state.ParOpen_prev = ParOpen
    
    # 计算最高价/最低价区间
    window_start = max(0, current_idx - X + 1)
    HH = high[window_start: current_idx + 1].max()
    LL = low[window_start: current_idx +1].max()
    
    # 交易信号生成
    current_pos = api.get_pos()
    
    # 开多条件
    if (state.Transition_prev == 1 and trendChange and 
        high[current_idx] >= HH and current_pos == 0):
        entry_price = max(open_p[current_idx], HH)
        api.buy(volume=Lots, order_type='next_bar_open')
        api.log(f"开多: {Lots}手 @ {entry_price:.2f}")
        state.HighestLowAfterEntry = open_p[current_idx]
        state.entPrice = open_p[current_idx]
        state.entBar = current_idx
    
    # 开空条件
    elif (state.Transition_prev == -1 and trendChange and 
          low[current_idx] <= LL and current_pos == 0):
        entry_price = min(open_p[current_idx], LL)
        api.sellshort(volume=Lots, order_type='next_bar_open')
        api.log(f"开空: {Lots}手 @ {entry_price:.2f}")
        state.LowestHighAfterEntry = open_p[current_idx]
        state.entPrice = open_p[current_idx]
        state.entBar = current_idx
    
    # 跟踪止损逻辑
    if current_pos > 0 and current_idx > state.entBar:
        state.HighestLowAfterEntry = max(state.HighestLowAfterEntry, low[current_idx])
        Trailing_Stop_L = state.HighestLowAfterEntry - open_p[current_idx] * TrailingStopRate / 1000
        if low[current_idx] <= Trailing_Stop_L:
            exit_price = min(open_p[current_idx], Trailing_Stop_L)
            api.sell(volume=Lots, order_type='next_bar_open')
            api.log(f"多头止损: {Lots}手 @ {exit_price:.2f}")
    
    elif current_pos < 0 and current_idx > state.entBar:
        state.LowestHighAfterEntry = min(state.LowestHighAfterEntry, high[current_idx])
        Trailing_Stop_S = state.LowestHighAfterEntry + open_p[current_idx] * TrailingStopRate / 1000
        if high[current_idx] >= Trailing_Stop_S:
            exit_price = max(open_p[current_idx], Trailing_Stop_S)
            api.buycover(volume=Lots, order_type='next_bar_open')
            api.log(f"空头止损: {Lots}手 @ {exit_price:.2f}")

if __name__ == "__main__":
    
    base_config = {
        'use_cache': True,
        'save_data': True,
        'debug': True,
        'inital_capital': 100000.0,
        'commission': 0.001,
        'margin_rate': 0.1,
        'contract_multiplier': 10
    }

    backtester = MultiSourceBacktester(base_config).set_optimization_mode(enable=False)
    
    backtester.add_symbol_config(
        symbol='btcusdt',
        config={
            'start_date': '2024-01-01',
            'end_date': '2025-12-31',
            'source_type': 'csv',
            'data_type': 'swap',            
            'interval': "30m"
        }
    )
    
    backtester.run_backtest(
        strategy_func=strategy_function,
        strategy_params={
            'lengs': 14,
            'AfStep': 0.02,
            'X': 20,
            'Lots': 2,
            'TrailingStopRate': 50,
            'AfLimit': 0.2
        },
    )
    # results = backtester.run(
    #     strategy=strategy_function,
    #     initialize=initialize,
    #     strategy_params={
    #         'lengs': 14,
    #         'AfStep': 0.02,
    #         'X': 20,
    #         'Lots': 2,
    #         'TrailingStopRate': 50,
    #         'AfLimit': 0.2
    #     },
    # )