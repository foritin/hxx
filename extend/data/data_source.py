import polars as pl
from typing import Dict, Optional, Any


class DataSource:
    """
    数据源类，用于管理单个数据源的数据和交易操作
    """
    
    def __init__(self, symbol: str, kline_period: str, adjust_type: str = '1', multi_datasource=None):
        """
        初始化数据源
        
        Args:
            symbol: 品种代码，如'rb888'
            kline_period: K线周期，如'1h', 'D'
            adjust_type: 复权类型，'0'表示不复权，'1'表示后复权
            multi_datasource: MultiDataSource实例，用于资金管理
        """
        self.symbol = symbol
        self.kline_period = kline_period
        self.adjust_type = adjust_type
        self.multi_datasource: Optional[MultiDataSource]  = multi_datasource  # 引用MultiDataSource实例
        self.data = pl.DataFrame()
        self.current_pos = 0
        self.target_pos = 0
        self.signal_reason = ""
        self.trades = []
        self.current_idx = 0
        self.current_price = None
        self.current_datetime = None
        self.pending_orders = []  # 存储待执行的订单
        
    def set_data(self, data: pl.DataFrame):
        """设置数据"""
        self.data = data
        
    def get_data(self) -> pl.DataFrame:
        """获取数据"""
        return self.data
        
    def get_current_price(self) -> Optional[float]:
        """获取当前价格"""
        if self.current_price is not None:
            return self.current_price
        if not self.data.is_empty() and self.current_idx < len(self.data):
            return self.data[self.current_idx, 'close']
        return None
        
    def get_current_datetime(self) -> Optional[Any]:
        """获取当前日期时间"""
        if self.current_datetime is not None:
            return self.current_datetime
        if not self.data.is_empty() and self.current_idx < len(self.data):
            return self.data[self.current_idx, 'datetime'] if 'datetime' in self.data.columns else self.data[self.current_idx, 'date']
        return None
        
    def get_current_pos(self) -> int:
        """获取当前持仓"""
        return self.current_pos
        
    def _update_pos(self, log_callback=None):
        """更新实际持仓"""
        if self.current_pos != self.target_pos:
            old_pos = self.current_pos
            self.current_pos = self.target_pos
            if log_callback:
                # 添加debug参数检查
                debug_mode = getattr(log_callback, 'debug_mode', True)
                if debug_mode:
                    log_callback(f"{self.symbol} {self.kline_period} 持仓变化: {old_pos} -> {self.current_pos}")
                
    def set_target_pos(self, target_pos: int, log_callback=None):
        """设置目标持仓"""
        self.target_pos = target_pos
        self._update_pos(log_callback)
        
    def set_signal_reason(self, reason: str):
        """设置交易信号原因"""
        self.signal_reason = reason
        
    def add_trade(self, action: str, price: float, volume: int, reason: str, datetime=None):
        """添加交易记录"""
        if datetime is None:
            datetime = self.get_current_datetime()
        
        self.trades.append({
            'datetime': datetime,
            'action': action,
            'price': price,
            'volume': volume,
            'reason': ''  # 不再记录原因
        })
        
    def get_price_by_type(self, order_type='bar_close'):
        """
        根据订单类型获取价格
        
        Args:
            order_type (str): 订单类型，可选值：
                - 'bar_close': 当前K线收盘价（默认）
                - 'next_bar_open': 下一K线开盘价
                - 'next_bar_close': 下一K线收盘价
                - 'next_bar_high': 下一K线最高价
                - 'next_bar_low': 下一K线最低价
                - 'market': 市价单，按对手价成交，买入按ask1，卖出按bid1
        
        Returns:
            float: 价格，如果无法获取则返回None
        """
        if not self.data.is_empty():
            if order_type == 'bar_close':
                if self.current_idx < len(self.data):
                    return self.data[self.current_idx, 'close']
            elif order_type == 'next_bar_open':
                if self.current_idx + 1 < len(self.data) and 'open' in self.data.columns:
                    return self.data[self.current_idx + 1, 'open']
            elif order_type == 'next_bar_close':
                if self.current_idx + 1 < len(self.data):
                    return self.data[self.current_idx + 1, 'close']
            elif order_type == 'next_bar_high':
                if self.current_idx + 1 < len(self.data) and 'high' in self.data.columns:
                    return self.data[self.current_idx + 1, 'high']
            elif order_type == 'next_bar_low':
                if self.current_idx + 1 < len(self.data) and 'low' in self.data.columns:
                    return self.data[self.current_idx + 1, 'low']
            elif order_type == 'market':
                # 市价单，对于tick数据，可以使用买一卖一价格
                if self.current_idx < len(self.data):
                    if 'ask1' in self.data.columns and 'bid1' in self.data.columns:
                        # 对于tick数据，市价单买入使用卖一价格(ask1)，卖出使用买一价格(bid1)
                        # 这里返回None，在具体的buy/sell方法中根据买卖方向确定价格
                        return None
                    else:
                        # 对于K线数据，使用收盘价
                        return self.data[self.current_idx, 'close']
        return None
        
    def _process_pending_orders(self, log_callback=None):
        """处理待执行的订单"""
        if not self.pending_orders:
            return
        
        # 获取debug模式设置
        debug_mode = getattr(log_callback, 'debug_mode', True) if log_callback else True
        
        orders_to_remove = []
        for i, order in enumerate(self.pending_orders):
            # 获取执行时间
            execution_time = order.get('execution_time', self.current_idx + 1)
            
            # 获取订单类型（默认为next_bar_open）
            order_type = order.get('order_type', 'next_bar_open')
            
            # 判断是否到达执行时间
            if execution_time <= self.current_idx:
                # 执行订单
                action = order['action']
                volume = order['volume']
                reason = order['reason']
                
                # 根据订单类型获取执行价格
                # 如果已经预先计算了价格，就使用那个价格
                if 'price' in order and order['price'] is not None:
                    price = order['price']
                else:
                    # 否则根据订单类型获取当前价格
                    price = self.get_price_by_type(order_type)
                    if price is None:
                        # 如果仍然无法获取价格，则使用当前价格
                        price = self.get_current_price()
                        if price is None:
                            # 如果完全无法获取价格，跳过此订单
                            continue
                
                # 更新持仓
                if action == "开多":
                    self.target_pos = self.current_pos + volume
                elif action == "平多":
                    if volume is None:
                        volume = max(0, self.current_pos)
                    # 检查是否有多头持仓可平
                    actual_volume = min(volume, max(0, self.current_pos))
                    if actual_volume <= 0:
                        # 没有多头持仓可平，跳过此订单
                        orders_to_remove.append(i)
                        continue
                    self.target_pos = self.current_pos - actual_volume
                    volume = actual_volume  # 更新volume为实际交易量
                elif action == "开空":
                    self.target_pos = self.current_pos - volume
                elif action == "平空":
                    if volume is None:
                        volume = max(0, -self.current_pos)
                    # 检查是否有空头持仓可平
                    actual_volume = min(volume, max(0, -self.current_pos))
                    if actual_volume <= 0:
                        # 没有空头持仓可平，跳过此订单
                        orders_to_remove.append(i)
                        continue
                    self.target_pos = self.current_pos + actual_volume
                    volume = actual_volume  # 更新volume为实际交易量
                elif action == "平多开空":  # 支持反手交易
                    self.target_pos = -self.current_pos  # 从多头变为空头
                elif action == "平空开多":  # 支持反手交易
                    self.target_pos = -self.current_pos  # 从空头变为多头
                
                # 更新持仓
                self._update_pos(log_callback)
                
                # 记录交易
                self.add_trade(action, price, volume, reason)
                
                if log_callback and debug_mode:
                    log_callback(f"{self.symbol} {self.kline_period} 执行订单: {action} {volume}手 成交价:{price:.2f} 类型:{order_type} 原因:{reason}")
                
                # 标记为待移除
                orders_to_remove.append(i)
        
        # 移除已执行的订单（从后往前移除，避免索引问题）
        for i in sorted(orders_to_remove, reverse=True):
            self.pending_orders.pop(i)
        
    def buy(self, rate: float = 0.0, volume: int = 0, reason: str = "", log_callback=None, order_type='bar_close'):
        """
        开多仓
        
        Args:
            volume (int): 交易数量
            reason (str): 交易原因
            log_callback: 日志回调函数
            order_type (str): 订单类型，可选值：
                - 'bar_close': 当前K线收盘价（默认）
                - 'next_bar_open': 下一K线开盘价
                - 'next_bar_close': 下一K线收盘价
                - 'next_bar_high': 下一K线最高价
                - 'next_bar_low': 下一K线最低价
                - 'market': 市价单，按ask1价格成交（买入用卖一价）
        
        Returns:
            bool: 是否成功下单
        """

        # 使用MultiDataSource的资金管理
        if self.multi_datasource is None:
            raise ValueError("System error, DataSource must belong to MultiDataSource to trade")
        
        # 获取debug模式设置
        debug_mode = getattr(log_callback, 'debug_mode', True) if log_callback else True

        # 计算资金模式
        if rate:

        if order_type == "bar_close":
            # 立即执行
            price = self.get_current_price()
            if price is None:
                return False
            self.target_pos = self.current_pos + volume
            if reason:
                self.set_signal_reason(reason)
            self._update_pos(log_callback)
        
    def sell(self, volume: Optional[int] = None, reason: str = "", log_callback=None, order_type='bar_close'):
        """
        平多仓
        
        Args:
            volume (int, optional): 交易数量，None表示平掉所有多仓
            reason (str): 交易原因
            log_callback: 日志回调函数
            order_type (str): 订单类型，可选值同buy函数
        
        Returns:
            bool: 是否成功下单
        """
        # 使用MultiDataSource的资金管理
        if self.multi_datasource is None:
            raise ValueError("DataSource必须属于MultiDataSource才能进行交易")
        
        return self.multi_datasource.sell_with_datasource(self, volume, reason, log_callback, order_type)
        
    def sellshort(self, volume: int = 1, reason: str = "", log_callback=None, order_type='bar_close'):
        """
        开空仓
        
        Args:
            volume (int): 交易数量
            reason (str): 交易原因
            log_callback: 日志回调函数
            order_type (str): 订单类型，可选值同buy函数
        
        Returns:
            bool: 是否成功下单
        """
        # 使用MultiDataSource的资金管理
        if self.multi_datasource is None:
            raise ValueError("DataSource必须属于MultiDataSource才能进行交易")
        
        return self.multi_datasource.sellshort_with_datasource(self, volume, reason, log_callback, order_type)
        
    def buycover(self, volume: Optional[int] = None, reason: str = "", log_callback=None, order_type='bar_close'):
        """
        平空仓
        
        Args:
            volume (int, optional): 交易数量，None表示平掉所有空仓
            reason (str): 交易原因
            log_callback: 日志回调函数
            order_type (str): 订单类型，可选值同buy函数
        
        Returns:
            bool: 是否成功下单
        """
        # 使用MultiDataSource的资金管理
        if self.multi_datasource is None:
            raise ValueError("DataSource必须属于MultiDataSource才能进行交易")
        
        return self.multi_datasource.buycover_with_datasource(self, volume, reason, log_callback, order_type)
        
    def reverse_pos(self, reason: str = "", log_callback=None, order_type='bar_close'):
        """
        反手（多转空，空转多）
        
        Args:
            reason (str): 交易原因
            log_callback: 日志回调函数
            order_type (str): 订单类型，可选值同buy函数
        
        Returns:
            bool: 是否成功下单
        """
        # 获取debug模式设置
        debug_mode = getattr(log_callback, 'debug_mode', True) if log_callback else True
        
        old_pos = self.current_pos
        
        if order_type == 'bar_close':
            # 当前K线收盘价下单，立即执行
            price = self.get_current_price()
            if price is None:
                return False
                
            self.target_pos = -old_pos
            if reason:
                self.set_signal_reason(reason)
            self._update_pos(log_callback)
            
            # 记录交易
            if old_pos > 0:
                self.add_trade("平多开空", price, old_pos, reason)
            elif old_pos < 0:
                self.add_trade("平空开多", price, -old_pos, reason)
                
            return True
        elif order_type == 'market':
            # 市价单，对于tick数据，使用对应价格
            price = None
            
            # 不同持仓方向使用不同的价格
            if old_pos > 0:  # 平多开空，卖出用买一价格(bid1)，买入用卖一价格(ask1)
                if 'bid1' in self.data.columns and self.current_idx < len(self.data):
                    price = self.data[self.current_idx, 'bid1']
                else:
                    price = self.get_current_price()
            elif old_pos < 0:  # 平空开多，买入用卖一价格(ask1)
                if 'ask1' in self.data.columns and self.current_idx < len(self.data):
                    price = self.data[self.current_idx, 'ask1']
                else:
                    price = self.get_current_price()
            else:
                return True  # 无持仓，无需反手
            
            if price is None:
                return False
                
            self.target_pos = -old_pos
            if reason:
                self.set_signal_reason(reason)
            self._update_pos(log_callback)
            
            # 记录交易
            if old_pos > 0:
                self.add_trade("平多开空", price, old_pos, reason)
                if log_callback and debug_mode:
                    log_callback(f"{self.symbol} {self.kline_period} 市价反手: 平多开空 {old_pos}手 成交价:{price:.2f} 原因:{reason}")
            elif old_pos < 0:
                self.add_trade("平空开多", price, -old_pos, reason)
                if log_callback and debug_mode:
                    log_callback(f"{self.symbol} {self.kline_period} 市价反手: 平空开多 {-old_pos}手 成交价:{price:.2f} 原因:{reason}")
                
            return True
        else:
            # 下一K线价格下单，添加到待执行队列
            price = self.get_price_by_type(order_type)
            # 注意：如果是next_bar_open/high/low/close，价格可能为None，将在执行时获取
            
            # 添加到待执行队列
            if old_pos > 0:
                action = "平多开空"
                volume = old_pos
            elif old_pos < 0:
                action = "平空开多"
                volume = -old_pos
            else:
                return True  # 无持仓，无需反手
            
            self.pending_orders.append({
                'action': action,
                'volume': volume,
                'price': price,  # 可能为None，将在执行时重新获取
                'reason': reason,
                'order_type': order_type,  # 保存订单类型
                'execution_time': self.current_idx + 1  # 在下一K线执行
            })
            
            if log_callback and debug_mode:
                price_str = f"{price:.2f}" if price is not None else "待确定"
                log_callback(f"{self.symbol} {self.kline_period} 添加待执行订单: {action} {volume}手 订单类型:{order_type} 预计价格:{price_str} 原因:{reason}")
            
            return True
        
    def close_all(self, reason: str = "", log_callback=None, order_type='bar_close'):
        """
        平掉所有持仓
        
        Args:
            reason (str): 交易原因
            log_callback: 日志回调函数
            order_type (str): 订单类型，可选值同buy函数
        
        Returns:
            bool: 是否成功下单
        """
        # 获取debug模式设置
        debug_mode = getattr(log_callback, 'debug_mode', True) if log_callback else True
        
        if self.current_pos > 0:
            # 平多仓
            if order_type == 'market':
                # 市价单，对于tick数据，卖出使用买一价格(bid1)
                price = None
                if 'bid1' in self.data.columns and self.current_idx < len(self.data):
                    # 使用当前tick的买一价格作为成交价
                    price = self.data[self.current_idx, 'bid1']
                else:
                    # 如果不是tick数据或者无法获取bid1，则使用当前价格
                    price = self.get_current_price()
                
                if price is None:
                    return False
                    
                volume = self.current_pos
                self.target_pos = 0
                if reason:
                    self.set_signal_reason(reason)
                self._update_pos(log_callback)
                
                # 记录交易
                self.add_trade("平多", price, volume, reason)
                
                if log_callback and debug_mode:
                    log_callback(f"{self.symbol} {self.kline_period} 市价平仓: 平多 {volume}手 成交价:{price:.2f} 原因:{reason}")
                
                return True
            else:
                return self.sell(volume=None, reason=reason, log_callback=log_callback, order_type=order_type)
        elif self.current_pos < 0:
            # 平空仓
            if order_type == 'market':
                # 市价单，对于tick数据，买入使用卖一价格(ask1)
                price = None
                if 'ask1' in self.data.columns and self.current_idx < len(self.data):
                    # 使用当前tick的卖一价格作为成交价
                    price = self.data[self.current_idx, 'ask1']
                else:
                    # 如果不是tick数据或者无法获取ask1，则使用当前价格
                    price = self.get_current_price()
                
                if price is None:
                    return False
                    
                volume = -self.current_pos
                self.target_pos = 0
                if reason:
                    self.set_signal_reason(reason)
                self._update_pos(log_callback)
                
                # 记录交易
                self.add_trade("平空", price, volume, reason)
                
                if log_callback and debug_mode:
                    log_callback(f"{self.symbol} {self.kline_period} 市价平仓: 平空 {volume}手 成交价:{price:.2f} 原因:{reason}")
                
                return True
            else:
                return self.buycover(volume=None, reason=reason, log_callback=log_callback, order_type=order_type)
        return True  # 已经没有持仓
    
    # 数据访问方法
    def get_close(self) -> pl.Series:
        """获取收盘价序列"""
        return self.data['close'] if 'close' in self.data.columns else pl.Series()
        
    def get_open(self) -> pl.Series:
        """获取开盘价序列"""
        return self.data['open'] if 'open' in self.data.columns else pl.Series()
        
    def get_high(self) -> pl.Series:
        """获取最高价序列"""
        return self.data['high'] if 'high' in self.data.columns else pl.Series()
        
    def get_low(self) -> pl.Series:
        """获取最低价序列"""
        return self.data['low'] if 'low' in self.data.columns else pl.Series()
        
    def get_volume(self) -> pl.Series:
        """获取成交量序列"""
        return self.data['volume'] if 'volume' in self.data.columns else pl.Series()
        
    def get_klines(self) -> pl.DataFrame:
        """获取K线数据"""
        return self.data

    def get_tick(self) -> Optional[pl.Series]:
        """返回当前tick的所有字段（Series）"""
        if not self.data.is_empty() and self.current_idx < len(self.data):
            return self.data[self.current_idx].to_series()
        return None

    def get_ticks(self, window: int = 100) -> pl.DataFrame:
        """返回最近window条tick数据（DataFrame）"""
        if not self.data.is_empty() and self.current_idx < len(self.data):
            start = max(0, self.current_idx - window + 1)
            return self.data.slice(start, self.current_idx - start + 1)
        return pl.DataFrame()

class MultiDataSource:
    """
    多数据源管理类，用于管理多个数据源和统一资金管理
    """
    
    def __init__(self):
        """初始化多数据源管理器"""
        self.data_sources = []
        self.log_callback = None
        
        # 资金管理相关属性
        self.total_capital = 0.0          # 总资金
        self.available_capital = 0.0      # 可用资金
        self.commission = 0.0             # 手续费率
        self.margin_rate = 0.0            # 保证金率
        self.contract_multiplier = 0.0     # 合约乘数
        self.frozen_capital = 0.0         # 冻结资金总额
        self.unrealized_pnl = 0.0         # 未实现盈亏总额
        self.realized_pnl = 0.0           # 已实现盈亏总额
        
        # 每个数据源的详细信息
        self.datasource_info = {}          # key: datasource_index, value: dict containing capital info
        
    def set_log_callback(self, callback):
        """设置日志回调函数"""
        self.log_callback = callback
        
    def add_data_source(self, symbol: str, kline_interval: str, adjust_type: str = '1', data: Optional[pl.DataFrame] = None) -> int:
        """
        添加数据源
        
        Args:
            symbol: 品种代码，如'rb888'
            kline_interval: K线周期，如'1h', 'D'
            adjust_type: 复权类型，'0'表示不复权，'1'表示后复权
            data: 数据，如果为None则创建空数据源
            
        Returns:
            数据源索引
        """
        # 将自身MultiDataSource实例传递给DataSource
        data_source = DataSource(symbol, kline_interval, adjust_type, self)
        if data is not None:
            data_source.set_data(data)
        self.data_sources.append(data_source)
        
        # 为新数据源初始化资金信息
        datasource_index = len(self.data_sources) - 1
        self.datasource_info[datasource_index] = {
            'frozen_capital': 0.0,        # 该datasource冻结的资金
            'unrealized_pnl': 0.0,        # 该datasource未平仓盈亏
            'realized_pnl': 0.0,          # 该datasource已实现盈亏
            'position_value': 0.0,        # 持仓价值
            'margin_used': 0.0,           # 保证金使用
            'symbol': symbol,              # 品种代码
            'kline_period': kline_interval  # K线周期
        }
        
        return datasource_index
    
    def initialize_trade_config(self, trade_config: Dict):
        """
        初始化共用资金池
        
        Args:
            capital: 初始资金，默认为100000
        """
        self.total_capital = trade_config.get("initial_captial")
        self.available_capital = trade_config.get("initial_captial")
        self.commission = trade_config.get("commission")
        self.margin_rate = trade_config.get("margin_rate")
        self.contract_multiplier = trade_config.get("contract_multiplier")
        self.frozen_capital = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        if self.log_callback:
            self.log_callback(f"Capital initialized to {capital:.2f}")
    
    def get_data_source(self, index: int) -> Optional[DataSource]:
        """获取指定索引的数据源"""
        if 0 <= index < len(self.data_sources):
            return self.data_sources[index]
        return None
        
    def get_data_sources_count(self) -> int:
        """获取数据源数量"""
        return len(self.data_sources)
        
    def __getitem__(self, index: int) -> Optional[DataSource]:
        """通过索引访问数据源"""
        return self.get_data_source(index)
        
    def __len__(self) -> int:
        """获取数据源数量"""
        return self.get_data_sources_count()
        
    def align_data(self, align_index: bool = True, fill_method: Optional[str] = 'ffill'):
        """
        对齐所有数据源的数据
        
        Args:
            align_index: 是否对齐索引
            fill_method: 填充方法，可选值：'ffill', 'bfill', None
        """
        if len(self.data_sources) <= 1:
            return  # 只有一个或没有数据源，不需要对齐
            
        if not align_index:
            return  # 如果不对齐索引，直接返回
            
        # 收集所有数据源的索引（假设数据框有'datetime'列）
        all_datetime_values = []
        valid_datasources = []
        
        for i, ds in enumerate(self.data_sources):
            if not ds.data.is_empty() and 'datetime' in ds.data.columns:
                datetime_series = ds.data['datetime']
                if not datetime_series.is_empty():
                    all_datetime_values.append(set(datetime_series.to_list()))
                    valid_datasources.append(i)
                    
        if not all_datetime_values:
            if self.log_callback:
                self.log_callback("No valid data sources found for alignment")
            return  # 没有有效的数据源
            
        # 计算所有数据源共有的datetime值
        common_datetime = set.intersection(*all_datetime_values)
        
        # 统一转换为毫秒精度，避免精度不匹配问题
        common_datetime = {dt.replace(microsecond=dt.microsecond // 1000 * 1000) for dt in common_datetime}
        
        if not common_datetime:
            if self.log_callback:
                self.log_callback("Warning: No common time points found across all data sources, unable to align data")
            return
            
        if self.log_callback:
            self.log_callback(f"Found {len(common_datetime)} common time points, starting alignment")
            
        # 创建一个参考数据框，包含共同的时间点
        if common_datetime:
            # 统一转换为毫秒精度
            common_datetime_ms = {dt.replace(microsecond=dt.microsecond // 1000 * 1000) for dt in common_datetime}
            reference_df = pl.DataFrame({
                'datetime': list(common_datetime_ms)
            }).sort('datetime')
            # 统一为毫秒精度
            reference_df = reference_df.with_columns(
                pl.col('datetime').dt.cast_time_unit('ms')
            )
        else:
            reference_df = pl.DataFrame({'datetime': []})
            reference_df = reference_df.with_columns(
                pl.col('datetime').cast(pl.Datetime('ms'))
            )
        
        # 对齐所有数据源的数据
        aligned_count = 0
        for i, ds in enumerate(self.data_sources):
            if i in valid_datasources and not ds.data.is_empty():
                original_len = len(ds.data)
                
                # 统一时间精度为毫秒，避免精度不匹配问题
                if ds.data['datetime'].dtype.time_unit == 'us':
                    ds.data = ds.data.with_columns(
                        pl.col("datetime").dt.cast_time_unit('ms')
                    )
                
                # 使用join来对齐数据，避免精度问题
                ds.data = reference_df.join(
                    ds.data, 
                    on='datetime', 
                    how='left'
                )
                
                # 按时间排序
                ds.data = ds.data.sort("datetime")
                
                # 根据fill_method填充缺失值
                if fill_method == 'ffill':
                    ds.data = ds.data.fill_null(strategy='forward')
                elif fill_method == 'bfill':
                    ds.data = ds.data.fill_null(strategy='backward')
                elif fill_method == 'none' or fill_method is None:
                    # 不填充，保持null值
                    pass
                
                aligned_count += 1
                
                if self.log_callback:
                    final_len = len(ds.data)
                    self.log_callback(f"Data source {i} ({ds.symbol} {ds.kline_period}): "
                                    f"{original_len} -> {final_len} rows")
            elif not ds.data.is_empty():
                # 无效的数据源（没有datetime列）
                if self.log_callback:
                    self.log_callback(f"Warning: Data source {i} ({ds.symbol} {ds.kline_period}) has no datetime column, skipping alignment")
                    
        if self.log_callback:
            self.log_callback(f"Data alignment completed, aligned {aligned_count} data sources") 


    # 资金方法
    def get_available_capital(self) -> Optional[float]:
        """获取可用资金"""
        return self.available_capital
    
    