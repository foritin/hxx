import polars as pl
import numpy as np
from typing import Optional, Dict, Any
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


class Strategy:
    """
    VIP26 Remastered Edition策略类
    
    该策略结合MACD和多重SMA过滤器：
    1. MACD判断趋势方向
    2. 多重SMA条件过滤
    3. 突破确认开仓
    4. 动态跟踪止损
    """
    
    def __init__(self, strategy_params: Optional[Dict[str, Any]] = None):
        """
        初始化策略
        
        Args:
            strategy_params: 策略参数字典，如果为None则使用默认参数
        """
        # 策略参数（带默认值）
        self.default_params = {
            'M': 20,
            'S': 1.0,
            'Lengs': 5,
            'Fund': 100000,
            'TrailingStopRate': 80,
            'FastLength': 12,
            'SlowLength': 26,
            'MACDLength': 9,
            'OrderPercent': 0.2
        }
        
        # 合并用户参数和默认参数
        self.params = {**self.default_params, **(strategy_params or {})}
        
        # 初始化状态
        self.reset()
    
    def reset(self):
        """重置策略状态"""
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
    
    def get_param(self, key: str, default_value=None):
        """获取策略参数"""
        return self.params.get(key, default_value)
    
    def run(self, api: StrategyAPI):
        """
        策略主运行方法
        
        Args:
            api: 策略API接口
        """
        # 数据验证和完整性检查
        if not self._validate_data(api):
            return
        
        current_idx = api.get_idx()
        
        # 确保有足够的数据
        min_required_bars = max(
            self.get_param('M'), 
            self.get_param('SlowLength'), 
            self.get_param('Lengs')
        ) + 10
        
        if current_idx < min_required_bars:
            if current_idx == min_required_bars - 1:
                api.log(f"数据准备中，需要至少 {min_required_bars} 根K线")
            return
        
        # 1. 计算所有指标
        indicators = self._calculate_indicators(api)
        if not indicators:
            return
        
        # 2. 计算KG信号
        KG = self._calculate_kg_signal(api, indicators, current_idx)
        
        # 3. 检测机会信号
        self._check_opportunity_signals(api, KG, indicators, current_idx)
        
        # 4. 检查开仓条件
        self._check_entry_conditions(api, current_idx)
        
        # 5. 跟踪止损处理
        self._handle_trailing_stop(api, current_idx)
        
        # 6. 清理无持仓时的状态
        self._cleanup_no_position_states(api)
        
        # 7. 定期打印状态信息
        self._periodic_status_report(api, KG, current_idx)
        
        # 8. 添加优化调试信息（每50个bar）
        if current_idx % 50 == 0:
            api.log(f"调试 Bar {current_idx}: KG={KG}, KI={self.KI}, HI={self.HI}, LI={self.LI}")
            if len(self.KG_history) >= 2:
                api.log(f"KG历史: {self.KG_history[-3:] if len(self.KG_history) >= 3 else self.KG_history}")
            
            # 检查数据质量
            close = api.get_close()
            if close[current_idx] is not None:
                api.log(f"当前价格: {close[current_idx]:.2f}, 持仓: {api.get_pos()}")
    
    def _validate_data(self, api: StrategyAPI) -> bool:
        """验证数据完整性"""
        close = api.get_close()
        if close is None or len(close) == 0:
            api.log("警告: 数据为空")
            return False
            
        current_idx = api.get_idx()
        if current_idx < 1:
            return False
        
        # 检查当前数据完整性
        if close[current_idx] is None:
            api.log(f"警告: Bar {current_idx} 收盘价为None，跳过处理")
            return False
        
        # 验证关键价格数据的完整性
        open_p = api.get_open()
        high = api.get_high()
        low = api.get_low()
        
        if (open_p[current_idx] is None or 
            high[current_idx] is None or 
            low[current_idx] is None):
            api.log(f"警告: Bar {current_idx} 价格数据不完整，跳过处理")
            return False
        
        return True
    
    def _calculate_indicators(self, api: StrategyAPI) -> Optional[Dict[str, Any]]:
        """计算所有技术指标"""
        try:
            close = api.get_close()
            high = api.get_high()
            low = api.get_low()
            current_idx = api.get_idx()
            
            # 计算MACD指标
            FastLength = self.get_param('FastLength')
            SlowLength = self.get_param('SlowLength')
            MACDLength = self.get_param('MACDLength')
            
            macd_cache_key = f"macd_{FastLength}_{SlowLength}_{MACDLength}"
            macd_diff, avg_macd, macd_value = self.indicator_cache.get_or_compute(
                macd_cache_key,
                lambda: self._calculate_macd(close, FastLength, SlowLength, MACDLength),
                len(close)
            )
            
            # 验证MACD指标的完整性
            if (macd_diff[current_idx] is None or avg_macd[current_idx] is None):
                api.log(f"警告: Bar {current_idx} MACD指标不完整")
            
            # 计算价格基础：(H+L+C)/3
            hlc3 = self.indicator_cache.get_or_compute(
                "hlc3",
                lambda: (high + low + close) / 3,
                len(close)
            )
            
            # 计算SMA指标
            M = self.get_param('M')
            S = self.get_param('S')
            
            if S == 1.0:  # 当S=1时强制产生差异
                period_1 = int(M * 0.6)     # sma_1用较短周期，更敏感
                period_long = M             # sma_Long用原周期
            else:
                # 确保period_1至少为1，避免ewm_mean(half_life=0)错误
                period_1 = max(1, int(M/S)) if S != 0 else M
                period_long = M
            
            # 参数验证：确保所有周期都大于0
            if period_1 <= 0 or period_long <= 0:
                api.log(f"参数错误: M={M}, S={S}, period_1={period_1}, period_long={period_long}")
                return None
            
            # 使用缓存计算SMA指标
            sma_1_key = f"sma_1_{period_1}"
            sma_1 = self.indicator_cache.get_or_compute(
                sma_1_key,
                lambda: hlc3.ewm_mean(half_life=period_1),
                len(close)
            )
            
            sma_long_key = f"sma_long_{period_long}"
            sma_Long = self.indicator_cache.get_or_compute(
                sma_long_key,
                lambda: hlc3.rolling_mean(window_size=period_long),
                len(close)
            )
            
            # 使用缓存计算XAverage指标
            sma_2_key = f"sma_2_{period_1}"
            sma_2 = self.indicator_cache.get_or_compute(
                sma_2_key,
                lambda: self._calculate_xaverage(sma_1, period_1),
                len(close)
            )
            
            smalong_ma_key = f"smalong_ma_{period_long}"
            smalong_ma = self.indicator_cache.get_or_compute(
                smalong_ma_key,
                lambda: self._calculate_xaverage(sma_Long, period_long),
                len(close)
            )
            
            # 计算加权指数波动差
            marange_key = f"marange_{period_1}_{period_long}"
            marange = self.indicator_cache.get_or_compute(
                marange_key,
                lambda: sma_1 - smalong_ma,
                len(close)
            )
            
            marange_ma_key = f"marange_ma_{period_1}"
            marange_ma = self.indicator_cache.get_or_compute(
                marange_ma_key,
                lambda: self._calculate_xaverage(marange, period_1),
                len(close)
            )
            
            # 计算HH和LL（突破用）
            Lengs = self.get_param('Lengs')
            rolling_extremes_key = f"rolling_extremes_{Lengs}"
            HH_series, LL_series = self.indicator_cache.get_or_compute(
                rolling_extremes_key,
                lambda: (high.rolling_max(window_size=Lengs), low.rolling_min(window_size=Lengs)),
                len(close)
            )
            HH = HH_series[current_idx]
            LL = LL_series[current_idx]
            
            # 调试：打印周期参数和实际值差异
            if current_idx % 1000 == 0:
                api.log(f"周期调试: M={M}, S={S}, period_1={period_1}, period_long={period_long}")
                
                # 显示缓存统计
                cache_size = len(self.indicator_cache._cache)
                api.log(f"📊 指标缓存统计: 当前缓存{cache_size}个指标，数据长度{len(close)}")
                
                if sma_1[current_idx] is not None and sma_Long[current_idx] is not None:
                    diff = abs(sma_1[current_idx] - sma_Long[current_idx])
                    api.log(f"SMA值差异: sma_1={sma_1[current_idx]:.4f}, sma_Long={sma_Long[current_idx]:.4f}, 差={diff:.4f}")
                    if diff < 1.0:
                        api.log(f"警告: SMA差异过小({diff:.4f})，可能影响信号生成")
            
            return {
                'macd_diff': macd_diff,
                'avg_macd': avg_macd,
                'macd_value': macd_value,
                'sma_1': sma_1,
                'sma_2': sma_2,
                'sma_Long': sma_Long,
                'smalong_ma': smalong_ma,
                'marange': marange,
                'marange_ma': marange_ma,
                'HH': HH,
                'LL': LL,
                'period_1': period_1,
                'period_long': period_long
            }
            
        except Exception as e:
            api.log(f"指标计算错误: {e}")
            return None
    
    def _calculate_macd(self, close, fast_length=12, slow_length=26, macd_length=9):
        """计算MACD指标"""
        fast_ema = self._calculate_xaverage(close, fast_length)
        slow_ema = self._calculate_xaverage(close, slow_length)
        
        macd_diff = fast_ema - slow_ema
        avg_macd = self._calculate_xaverage(macd_diff, macd_length)
        macd_value = 2 * (macd_diff - avg_macd)
        
        return macd_diff, avg_macd, macd_value
    
    def _calculate_xaverage(self, series, period):
        """计算指数移动平均线（模拟TBQuant的XAverage）"""
        return series.ewm_mean(half_life=period)
    
    def _calculate_kg_signal(self, api: StrategyAPI, indicators: Dict[str, Any], current_idx: int) -> int:
        """计算KG信号"""
        KG = 0
        if current_idx >= 2:
            try:
                # 获取指标数据
                sma_1 = indicators['sma_1']
                sma_2 = indicators['sma_2']
                sma_Long = indicators['sma_Long']
                smalong_ma = indicators['smalong_ma']
                marange = indicators['marange']
                marange_ma = indicators['marange_ma']
                macd_diff = indicators['macd_diff']
                avg_macd = indicators['avg_macd']
                
                # 计算前1个bar的条件
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
                
                # 前3个bar的条件
                prev3_cond1 = False
                prev3_kcond1 = False
                if current_idx > 2:
                    prev3_cond1 = sma_1[current_idx-3] > sma_2[current_idx-3]
                    prev3_kcond1 = sma_1[current_idx-3] < sma_2[current_idx-3]
                
                # 前2个bar的综合条件
                prev2_condtion2 = ((prev2_cond1 or prev3_cond1) and 
                                  not prev2_cond2 and not prev2_cond3 and not prev2_cond4)
                
                prev2_kcondtion2 = ((prev2_kcond1 or prev3_kcond1) and
                                   not prev2_kcond2 and not prev2_kcond3 and not prev2_kcond4)
                
                # 计算KG信号
                if prev_condtion1 and prev2_condtion2 and prev_condtion3:
                    KG = 1
                    api.log(f"产生多头KG信号! Bar {current_idx}")
                elif prev_kcondtion1 and prev2_kcondtion2 and prev_kcondtion3:
                    KG = -1
                    api.log(f"产生空头KG信号! Bar {current_idx}")
                
                # 调试信息
                if current_idx % 200 == 0 and KG == 0:
                    api.log(f"--- KG未产生信号分析 Bar {current_idx} ---")
                    api.log(f"多头: condtion1={prev_condtion1}, condtion2={prev2_condtion2}, condtion3={prev_condtion3}")
                    api.log(f"空头: kcondtion1={prev_kcondtion1}, kcondtion2={prev2_kcondtion2}, kcondtion3={prev_kcondtion3}")
                    near_long = sum([prev_condtion1, prev2_condtion2, prev_condtion3])
                    near_short = sum([prev_kcondtion1, prev2_kcondtion2, prev_kcondtion3])
                    api.log(f"接近程度: 多头={near_long}/3, 空头={near_short}/3")
                
            except (IndexError, KeyError):
                # 如果访问历史数据出错，保持KG=0
                pass
        
        return KG
    
    def _check_opportunity_signals(self, api: StrategyAPI, KG: int, indicators: Dict[str, Any], current_idx: int):
        """检测机会信号"""
        macd_diff = indicators['macd_diff']
        avg_macd = indicators['avg_macd']
        HH = indicators['HH']
        LL = indicators['LL']
        
        # 保存当前KG值到历史中
        if macd_diff[current_idx] is not None and avg_macd[current_idx] is not None:
            self.KG_history.append(KG)
        else:
            self.KG_history.append(0)  # 数据不完整时保存0
            api.log(f"数据不完整，KG设为0 Bar {current_idx}")
        
        # 保持最近3个KG值
        if len(self.KG_history) > 3:
            self.KG_history.pop(0)
        
        # 调试：每100个bar检查KG分布
        if current_idx % 100 == 0:
            non_zero_kg = [kg for kg in self.KG_history if kg != 0]
            api.log(f"KG分布检查: 历史={self.KG_history}, 非零值={non_zero_kg}")
        
        # 检测机会信号：基于已保存的历史
        if len(self.KG_history) >= 2:
            KG_prev1 = self.KG_history[-2] if len(self.KG_history) >= 2 else 0  # 前1个bar的KG
            KG_prev2 = self.KG_history[-3] if len(self.KG_history) >= 3 else 0  # 前2个bar的KG
            
            # 多头机会：KG[1]==1 and KG[2]!=1
            if KG_prev1 == 1 and KG_prev2 != 1:
                api.log(f"检测到多头机会信号! Bar {current_idx}, KG历史: {self.KG_history}")
                # 数据完整性检查：确保HH和LL不是None
                if HH is not None and LL is not None:
                    self.HI = HH
                    self.LI = LL
                    self.KI = 1
                    api.log(f"检测到做多机会信号 (KG[1]={KG_prev1}, KG[2]={KG_prev2}) HI={HH:.2f}")
                else:
                    api.log(f"警告: HH或LL为None，跳过机会信号设置")
                
            # 空头机会：KG[1]==-1 and KG[2]!=-1  
            elif KG_prev1 == -1 and KG_prev2 != -1:
                api.log(f"检测到空头机会信号! Bar {current_idx}, KG历史: {self.KG_history}")
                # 数据完整性检查：确保HH和LL不是None
                if HH is not None and LL is not None:
                    self.HI = HH
                    self.LI = LL
                    self.KI = -1
                    api.log(f"检测到做空机会信号 (KG[1]={KG_prev1}, KG[2]={KG_prev2}) LI={LL:.2f}")
                else:
                    api.log(f"警告: HH或LL为None，跳过机会信号设置")
    
    def _check_entry_conditions(self, api: StrategyAPI, current_idx: int):
        """检查开仓条件"""
        datasource_mark = api.get_aim_datasource_mark()
        current_pos = api.get_pos()
        high = api.get_high()
        low = api.get_low()
        open_p = api.get_open()
        order_percent = self.get_param('OrderPercent')
        
        # 多头开仓/换仓：突破前期高点
        if (self.HI is not None and 
            high[current_idx] >= self.HI and 
            self.KI == 1):
            
            entry_price = max(open_p[current_idx], self.HI)
            
            # 处理不同持仓状态
            if current_pos <= 0:  # 只有空仓或换仓时才重置
                if current_pos < 0:
                    # 平空
                    api.close_short(datasource_mark=datasource_mark, entry_price=self.entPrice, order_type="bar_close")
                # 空仓开多
                api.buy(datasource_mark=datasource_mark, order_percent=order_percent, order_type="bar_close")
                api.log(f"多头开仓: @ {entry_price:.2f} (突破高点)")
                self.entBar = current_idx
                self.entPrice = entry_price
                self.liQKA = 1.0
                self.HighestLowAfterEntry = low[current_idx]
                self.LowestHighAfterEntry = None  # 清空空头状态
            
            # 清空机会信号（重要：避免重复触发）
            self.HI = None
            self.LI = None  
            self.KI = 0
        
        # 空头开仓/换仓：跌破前期低点
        elif (self.LI is not None and 
              low[current_idx] <= self.LI and 
              self.KI == -1):
            
            entry_price = min(open_p[current_idx], self.LI)
            
            # 处理不同持仓状态
            if current_pos >= 0:
                if current_pos > 0:
                    # 平多
                    api.close_long(datasource_mark=datasource_mark, entry_price=self.entPrice, order_type="bar_close")
                # 空仓开空
                api.sell(datasource_mark=datasource_mark, order_percent=order_percent, order_type="bar_close")
                api.log(f"空头开仓: @ {entry_price:.2f} (跌破低点)")
                self.entBar = current_idx
                self.entPrice = entry_price
                self.liQKA = 1.0
                self.LowestHighAfterEntry = high[current_idx]
                self.HighestLowAfterEntry = None  # 清空多头状态
            
            # 清空机会信号（重要：避免重复触发）
            self.HI = None
            self.LI = None  
            self.KI = 0
    
    def _handle_trailing_stop(self, api: StrategyAPI, current_idx: int):
        """处理跟踪止损"""
        current_pos = api.get_pos()
        datasource_mark = api.get_aim_datasource_mark()
        close = api.get_close()
        high = api.get_high()
        low = api.get_low()
        open_p = api.get_open()
        TrailingStopRate = self.get_param('TrailingStopRate')
        
        if current_pos > 0 and self.entBar is not None and current_idx > self.entBar:
            # 多头跟踪止损
            self._handle_long_trailing_stop(api, current_idx, close, high, low, open_p, 
                                           TrailingStopRate, datasource_mark)
        
        elif current_pos < 0 and self.entBar is not None and current_idx > self.entBar:
            # 空头跟踪止损
            self._handle_short_trailing_stop(api, current_idx, close, high, low, open_p, 
                                            TrailingStopRate, datasource_mark)
    
    def _handle_long_trailing_stop(self, api: StrategyAPI, current_idx, close, high, low, open_p, 
                                  trailing_stop_rate, datasource_mark):
        """处理多头跟踪止损"""
        # 更新入场后的最高低点
        if self.HighestLowAfterEntry is None:
            self.HighestLowAfterEntry = low[current_idx]
            api.log(f"初始化HighestLowAfterEntry: {self.HighestLowAfterEntry:.2f}")
        else:
            prev_highest_low = self.HighestLowAfterEntry
            self.HighestLowAfterEntry = max(self.HighestLowAfterEntry, low[current_idx])
            if self.HighestLowAfterEntry > prev_highest_low:
                api.log(f"更新HighestLowAfterEntry: {prev_highest_low:.2f} -> {self.HighestLowAfterEntry:.2f}")
        
        # 计算乖离率和动态调整止损系数
        # 这里简化处理，可以后续优化
        if close[current_idx] is not None and close[current_idx] != 0:
            # 简化的乖离率计算
            self.liQKA = max(self.liQKA - 0.02, 0.8)  # 温和的调整
        
        # 计算当前Bar的止损线
        stop_distance = (open_p[current_idx] * trailing_stop_rate / 1000) * self.liQKA
        DliqPoint = self.HighestLowAfterEntry - stop_distance
        
        # 添加调试信息
        if current_idx % 10 == 0 or not hasattr(self, 'DliqPoint_prev'):
            api.log(f"多头止损计算: HighestLow={self.HighestLowAfterEntry:.2f}, "
                    f"距离={stop_distance:.2f}, 止损线={DliqPoint:.2f}, liQKA={self.liQKA:.2f}")
        
        # 关键修正：如果前一个Bar有止损线，使用前一个Bar的止损线进行比较
        if hasattr(self, 'DliqPoint_prev') and self.DliqPoint_prev is not None:
            if low[current_idx] <= self.DliqPoint_prev:
                exit_price = min(open_p[current_idx], self.DliqPoint_prev)
                api.log(f"止损触发: 当前低点={low[current_idx]:.2f} <= 止损线={self.DliqPoint_prev:.2f}")
                api.close_long(datasource_mark=datasource_mark, order_type="bar_close", entry_price=self.entPrice)
                api.log(f"多头跟踪止损: @ {exit_price:.2f} (止损线: {self.DliqPoint_prev:.2f})")
                
                # 重置状态
                self.entBar = None
                self.entPrice = None
                self.HighestLowAfterEntry = None
                self.DliqPoint_prev = None
                return  # 止损后直接返回
        
        # 保存当前止损线供下一个Bar使用
        self.DliqPoint_prev = DliqPoint
    
    def _handle_short_trailing_stop(self, api: StrategyAPI, current_idx, close, high, low, open_p, 
                                   trailing_stop_rate, datasource_mark):
        """处理空头跟踪止损"""
        # 更新入场后的最低高点
        if self.LowestHighAfterEntry is None:
            self.LowestHighAfterEntry = high[current_idx]
            api.log(f"初始化LowestHighAfterEntry: {self.LowestHighAfterEntry:.2f}")
        else:
            prev_lowest_high = self.LowestHighAfterEntry
            self.LowestHighAfterEntry = min(self.LowestHighAfterEntry, high[current_idx])
            if self.LowestHighAfterEntry < prev_lowest_high:
                api.log(f"更新LowestHighAfterEntry: {prev_lowest_high:.2f} -> {self.LowestHighAfterEntry:.2f}")
        
        # 计算乖离率和动态调整止损系数
        if close[current_idx] is not None and close[current_idx] != 0:
            # 简化的乖离率计算
            self.liQKA = max(self.liQKA - 0.02, 0.8)  # 温和的调整
        
        # 计算当前Bar的止损线
        stop_distance = (open_p[current_idx] * trailing_stop_rate / 1000) * self.liQKA
        KliqPoint = self.LowestHighAfterEntry + stop_distance
        
        # 添加调试信息
        if current_idx % 10 == 0 or not hasattr(self, 'KliqPoint_prev'):
            api.log(f"空头止损计算: LowestHigh={self.LowestHighAfterEntry:.2f}, "
                    f"距离={stop_distance:.2f}, 止损线={KliqPoint:.2f}, liQKA={self.liQKA:.2f}")
        
        # 关键修正：如果前一个Bar有止损线，使用前一个Bar的止损线进行比较
        if hasattr(self, 'KliqPoint_prev') and self.KliqPoint_prev is not None:
            if high[current_idx] >= self.KliqPoint_prev:
                exit_price = max(open_p[current_idx], self.KliqPoint_prev)
                api.log(f"止损触发: 当前高点={high[current_idx]:.2f} >= 止损线={self.KliqPoint_prev:.2f}")
                api.close_short(datasource_mark=datasource_mark, order_type="bar_close", entry_price=self.entPrice)
                api.log(f"空头跟踪止损: @ {exit_price:.2f} (止损线: {self.KliqPoint_prev:.2f})")
                
                # 重置状态
                self.entBar = None
                self.entPrice = None
                self.LowestHighAfterEntry = None
                self.KliqPoint_prev = None
                return  # 止损后直接返回
        
        # 保存当前止损线供下一个Bar使用
        self.KliqPoint_prev = KliqPoint
    
    def _cleanup_no_position_states(self, api: StrategyAPI):
        """清理无持仓时的止损状态"""
        current_pos = api.get_pos()
        
        if current_pos == 0:
            # 清理跟踪止损的状态变量
            if hasattr(self, 'DliqPoint_prev') and self.DliqPoint_prev is not None:
                self.DliqPoint_prev = None
            if hasattr(self, 'KliqPoint_prev') and self.KliqPoint_prev is not None:
                self.KliqPoint_prev = None
            
            # 确保跟踪状态也被清理
            if self.entBar is None:
                self.HighestLowAfterEntry = None
                self.LowestHighAfterEntry = None
    
    def _periodic_status_report(self, api: StrategyAPI, KG: int, current_idx: int):
        """定期打印状态信息和数据完整性报告"""
        if current_idx % 100 == 0:
            current_pos = api.get_pos()
            api.log(f"Bar {current_idx}: 当前KG={KG}, KI={self.KI}, 持仓={current_pos}")
            api.log(f"KG历史: {self.KG_history}")
            
            # 数据完整性报告
            close = api.get_close()
            high = api.get_high()
            low = api.get_low()
            
            data_issues = []
            if close[current_idx] is None:
                data_issues.append("Close")
            if high[current_idx] is None:
                data_issues.append("High")
            if low[current_idx] is None:
                data_issues.append("Low")
                
            if data_issues:
                api.log(f"数据缺失警告: {', '.join(data_issues)}")
            
            # 状态变量完整性检查
            if self.HI is not None:
                api.log(f"HI值: {self.HI}")
            if self.LI is not None:
                api.log(f"LI值: {self.LI}")



# 全局策略实例
_strategy_instance = None
