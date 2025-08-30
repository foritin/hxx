import polars as pl
import numpy as np
from bidict import bidict
from typing import Dict, Optional, Any, List
from extend.core.backtest_config import TradeConfig


class DataSource:
    """
    数据源类，用于管理单个数据源的数据和交易操作
    """

    def __init__(self, index: int, symbol: str, kline_period: str, data: pl.DataFrame):
        """
        初始化数据源

        Args:
            key: 数据源索引
            symbol: 品种代码，如'rb888'
            kline_period: K线周期，如'1h', 'D'
            adjust_type: 复权类型，'0'表示不复权，'1'表示后复权
            multi_datasource: MultiDataSource实例，用于资金管理
        """
        self.index = index
        self.symbol = symbol
        self.kline_period = kline_period
        self.data = data
        self.current_pos = 0
        self.target_pos = 0
        self.signal_reason = ""
        self.trades = []
        self.current_idx = 0
        self.current_price: Optional[float] = None
        self.current_datetime: Optional[str] = None
        self.pending_orders: List[Dict] = []  # 存储待执行的订单

        # 临时变量
        self.entry_bar: int = 0
        self.entry_price: float = 0.0
        self.entry_volume: float = 0.0

        # 资产信息
        self.capital: float = 0
        self.available_capital: float = 0
        self.comission: float = 0
        self.slippage: float = 0
        self.leverage: int = 0
        self.frozen_capital: float = 0
        self.margin_rate: float = 0
        self.unrealized_pnl: float = 0.0
        self.realized_pnl: float = 0.0

    def get_data(self) -> pl.DataFrame:
        """获取数据"""
        return self.data

    def get_current_price(self) -> Optional[float]:
        """获取当前价格"""
        if self.current_price is not None:
            return self.current_price
        if not self.data.is_empty() and self.current_idx < len(self.data):
            return self.data[self.current_idx, "close"]
        return None

    def get_current_datetime(self) -> Optional[Any]:
        """获取当前日期时间"""
        if self.current_datetime is not None:
            return self.current_datetime
        if not self.data.is_empty() and self.current_idx < len(self.data):
            return self.data[self.current_idx, "datetime"] if "datetime" in self.data.columns else self.data[self.current_idx, "date"]
        return None

    def get_current_pos(self) -> float:
        """获取当前持仓"""
        return self.current_pos

    def _update_pos(self, log_callback=None):
        """更新实际持仓"""
        if self.current_pos != self.target_pos:
            old_pos = self.current_pos
            self.current_pos = self.target_pos
            if log_callback:
                # 添加debug参数检查
                debug_mode = getattr(log_callback, "debug_mode", True)
                if debug_mode:
                    log_callback(f"{self.symbol} {self.kline_period} position changed: {old_pos} -> {self.current_pos}")

    def set_target_pos(self, target_pos: int, log_callback=None):
        """设置目标持仓"""
        self.target_pos = target_pos
        self._update_pos(log_callback)

    def set_signal_reason(self, reason: str):
        """设置交易信号原因"""
        self.signal_reason = reason

    def add_trade(self, action: str, bar_index: int = 0, open_price: float = 0.0, close_price: float = 0.0, volume: float = 0.0, reason: str = "", pnl: float = 0.0, datetime=None):
        """添加交易记录"""
        if datetime is None:
            datetime = self.get_current_datetime()

        self.trades.append({"datetime": datetime, "bar": bar_index, "action": action, "open_price": open_price, "close_price": close_price, "volume": volume, "reason": reason, "pnl": pnl})

    def get_price_by_type(self, order_type="bar_close"):
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
            if order_type == "bar_close":
                if self.current_idx < len(self.data):
                    return self.data[self.current_idx, "close"]
            elif order_type == "next_bar_open":
                if self.current_idx + 1 < len(self.data) and "open" in self.data.columns:
                    return self.data[self.current_idx + 1, "open"]
            elif order_type == "next_bar_close":
                if self.current_idx + 1 < len(self.data):
                    return self.data[self.current_idx + 1, "close"]
            elif order_type == "next_bar_high":
                if self.current_idx + 1 < len(self.data) and "high" in self.data.columns:
                    return self.data[self.current_idx + 1, "high"]
            elif order_type == "next_bar_low":
                if self.current_idx + 1 < len(self.data) and "low" in self.data.columns:
                    return self.data[self.current_idx + 1, "low"]
            elif order_type == "market":
                # 市价单，对于tick数据，可以使用买一卖一价格
                if self.current_idx < len(self.data):
                    if "ask1" in self.data.columns and "bid1" in self.data.columns:
                        # 对于tick数据，市价单买入使用卖一价格(ask1)，卖出使用买一价格(bid1)
                        # 这里返回None，在具体的buy/sell方法中根据买卖方向确定价格
                        return None
                    else:
                        # 对于K线数据，使用收盘价
                        return self.data[self.current_idx, "close"]
        return None

    def _process_pending_orders(self, log_callback=None):
        """处理待执行的订单"""
        if not self.pending_orders:
            return

        # 获取debug模式设置
        debug_mode = getattr(log_callback, "debug_mode", True) if log_callback else True

        orders_to_remove = []
        for i, order in enumerate(self.pending_orders):
            # 获取执行时间
            execution_time = order.get("execution_time", self.current_idx + 1)

            # 获取订单类型（默认为next_bar_open）
            order_type = order.get("order_type", "next_bar_open")

            # 判断是否到达执行时间
            if execution_time <= self.current_idx:
                # 执行订单
                action = order["action"]
                reason = order["reason"]

                # 处理order_percent类型的订单
                if "order_percent" in order:
                    if action == "close long":
                        # 计算平仓数量
                        volume = self.current_pos * order["order_percent"]
                        if volume <= 0:
                            orders_to_remove.append(i)
                            continue
                    elif action == "open short":
                        # 计算开空仓数量
                        frozen_capital = order["order_percent"] * self.available_capital
                        volume = frozen_capital / price if price else 0
                        if volume <= 0:
                            orders_to_remove.append(i)
                            continue
                    else:
                        volume = order.get("volume", 0)
                else:
                    volume = order.get("volume", 0)

                # 根据订单类型获取执行价格
                # 如果已经预先计算了价格，就使用那个价格
                if "price" in order and order["price"] is not None:
                    price = order["price"]
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
                elif action == "close long":
                    if volume is None:
                        volume = max(0, self.current_pos)
                    # 检查是否有多头持仓可平
                    actual_volume = min(volume, max(0, self.current_pos))
                    if actual_volume <= 0:
                        # 没有多头持仓可平，跳过此订单
                        orders_to_remove.append(i)
                        continue
                    # 计算释放的资金
                    released_capital = actual_volume * price
                    self.available_capital += released_capital
                    self.frozen_capital -= released_capital
                    self.target_pos = self.current_pos - actual_volume
                    volume = actual_volume  # 更新volume为实际交易量
                elif action == "open short":
                    frozen_capital = volume * price
                    self.available_capital -= frozen_capital
                    self.frozen_capital += frozen_capital
                    self.target_pos = self.current_pos - volume
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

    def buy(self, order_percent: float = 0.0, reason: str = "", log_callback=None, order_type="bar_close"):
        """
        开多仓

        Args:
            order_percent (float): 下单百分比，0.0表示不下单
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
        debug_mode = getattr(log_callback, "debug_mode", True) if log_callback else True
        if order_type == "bar_close":
            # 立即执行
            price = self.get_current_price()
            if price is None:
                return False
            frozen_capital = order_percent * self.available_capital
            self.available_capital -= frozen_capital
            self.frozen_capital += frozen_capital
            volume = frozen_capital / price
            self.target_pos = self.current_pos + volume
            self.entry_bar = self.current_idx
            self.entry_price = price
            self.entry_volume = volume
            if reason:
                self.set_signal_reason(reason)
            self._update_pos(log_callback)
            self.add_trade(action="open long", open_price=price, volume=volume, reason=reason, bar_index=self.current_idx)
            return True
        else:
            # 添加到待执行订单
            price = self.get_price_by_type(order_type)
            self.pending_orders.append({"action": "open long", "price": price, "reason": reason, "order_percent": order_percent, "order_type": order_type, "execution_time": self.current_idx + 1})
            if log_callback and debug_mode:
                price_str = price if price is not None else "unset"
                log_callback(f"{self.symbol} {self.kline_period} add order: order_type:{order_type}|order_percent:{order_percent * 100:.2fype}|predict price:{price_str}|reason:{reason}")
            return True

    def close_long(self, order_percent: Optional[float] = None, reason: str = "", log_callback=None, order_type="bar_close"):
        """
        平多仓

        Args:
            order_percent (float): 平仓百分比，0.0表示不平仓，1.0表示平掉所有多仓
            reason (str): 交易原因
            log_callback: 日志回调函数
            order_type (str): 订单类型，可选值：
                - 'bar_close': 当前K线收盘价（默认）
                - 'next_bar_open': 下一K线开盘价
                - 'next_bar_close': 下一K线收盘价
                - 'next_bar_high': 下一K线最高价
                - 'next_bar_low': 下一K线最低价
                - 'market': 市价单，按bid1价格成交（卖出用买一价）

        Returns:
            bool: 是否成功下单
        """
        debug_mode = getattr(log_callback, "debug_mode", True) if log_callback else True

        # 检查是否有多头持仓可平
        if self.current_pos <= 0:
            if log_callback and debug_mode:
                log_callback(f"{self.symbol} {self.kline_period} no long position to close")
            return False
        if order_percent is None:
            order_percent = 1.0
        if order_type == "bar_close":
            # 立即执行
            price = self.get_current_price()
            if price is None:
                return False

            # 计算平仓数量
            volume = self.current_pos * order_percent
            if volume <= 0:
                return False

            # 计算释放的资金
            released_capital = self.frozen_capital * order_percent
            self.available_capital += released_capital
            self.frozen_capital -= released_capital

            # 计算盈亏
            pnl = (price - self.entry_price) * volume
            self.realized_pnl += pnl

            # 更新持仓
            self.target_pos = self.current_pos - volume
            if reason:
                self.set_signal_reason(reason)
            self._update_pos(log_callback)

            # 记录交易
            self.add_trade("close long", open_price=self.entry_price, close_price=price, volume=volume, reason=reason, bar_index=self.current_idx, pnl=pnl)

            return True
        else:
            # 添加到待执行订单
            price = self.get_price_by_type(order_type)
            self.pending_orders.append({"action": "close long", "price": price, "reason": reason, "order_percent": order_percent, "order_type": order_type, "execution_time": self.current_idx + 1})
            if log_callback and debug_mode:
                price_str = f"{price:.2f}" if price is not None else "unset"
                log_callback(f"{self.symbol} {self.kline_period} add close long order: order_type:{order_type}|order_percent:{order_percent * 100:.2f}%|predict price:{price_str}|reason:{reason}")
            return True

    def sell(self, order_percent: float = 0.0, reason: str = "", log_callback=None, order_type="bar_close"):
        """
        开空仓

        Args:
            order_percent (float): 下单百分比，0.0表示不下单
            reason (str): 交易原因
            log_callback: 日志回调函数
            order_type (str): 订单类型，可选值：
                - 'bar_close': 当前K线收盘价（默认）
                - 'next_bar_open': 下一K线开盘价
                - 'next_bar_close': 下一K线收盘价
                - 'next_bar_high': 下一K线最高价
                - 'next_bar_low': 下一K线最低价
                - 'market': 市价单，按bid1价格成交（卖出用买一价）

        Returns:
            bool: 是否成功下单
        """
        debug_mode = getattr(log_callback, "debug_mode", True) if log_callback else True
        if order_type == "bar_close":
            # 立即执行
            price = self.get_current_price()
            if price is None:
                return False
            frozen_capital = order_percent * self.available_capital
            self.available_capital -= frozen_capital
            self.frozen_capital += frozen_capital
            volume = frozen_capital / price
            self.target_pos = self.current_pos - volume
            if reason:
                self.set_signal_reason(reason)
            self._update_pos(log_callback)
            self.add_trade("open short", price, volume, reason)
            return True
        else:
            # 添加到待执行订单
            price = self.get_price_by_type(order_type)
            self.pending_orders.append({"action": "open short", "price": price, "reason": reason, "order_percent": order_percent, "order_type": order_type, "execution_time": self.current_idx + 1})
            if log_callback and debug_mode:
                price_str = price if price is not None else "unset"
                log_callback(f"{self.symbol} {self.kline_period} add order: order_type:{order_type}|order_percent:{order_percent * 100:.2fype}|predict price:{price_str}|reason:{reason}")
            return True

    def close_short(self, order_percent: Optional[float] = None, reason: str = "", log_callback=None, order_type="bar_close"):
        """
        平空仓

        Args:
            order_percent (float): 平仓百分比，0.0表示不平仓，1.0表示平掉所有空仓
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
        debug_mode = getattr(log_callback, "debug_mode", True) if log_callback else True

        # 检查是否有空头持仓可平
        if self.current_pos >= 0:
            if log_callback and debug_mode:
                log_callback(f"{self.symbol} {self.kline_period} no short position to close")
            return False
        if order_percent is None:
            order_percent = 1.0
        if order_type == "bar_close":
            # 立即执行
            price = self.get_current_price()
            if price is None:
                return False

            # 计算平仓数量
            volume = abs(self.current_pos) * order_percent
            if volume <= 0:
                return False

            # 计算释放的资金
            released_capital = self.frozen_capital * order_percent
            self.available_capital += released_capital
            self.frozen_capital -= released_capital

            # 计算盈亏
            profit = released_capital - (price * volume)
            self.realized_pnl += profit

            # 更新持仓
            self.target_pos = self.current_pos + volume
            if reason:
                self.set_signal_reason(reason)
            self._update_pos(log_callback)

            # 记录交易
            self.add_trade("close short", price, volume, reason)

            return True
        else:
            # 添加到待执行订单
            price = self.get_price_by_type(order_type)
            self.pending_orders.append({"action": "close short", "price": price, "reason": reason, "order_percent": order_percent, "order_type": order_type, "execution_time": self.current_idx + 1})
            if log_callback and debug_mode:
                price_str = f"{price:.2f}" if price is not None else "unset"
                log_callback(f"{self.symbol} {self.kline_period} add close short order: order_type:{order_type}|order_percent:{order_percent * 100:.2f}%|predict price:{price_str}|reason:{reason}")
            return True

    def reverse_pos(self, reason: str = "", log_callback=None, order_type="bar_close"):
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
        debug_mode = getattr(log_callback, "debug_mode", True) if log_callback else True

        old_pos = self.current_pos

        if order_type == "bar_close":
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
        elif order_type == "market":
            # 市价单，对于tick数据，使用对应价格
            price = None

            # 不同持仓方向使用不同的价格
            if old_pos > 0:  # 平多开空，卖出用买一价格(bid1)，买入用卖一价格(ask1)
                if "bid1" in self.data.columns and self.current_idx < len(self.data):
                    price = self.data[self.current_idx, "bid1"]
                else:
                    price = self.get_current_price()
            elif old_pos < 0:  # 平空开多，买入用卖一价格(ask1)
                if "ask1" in self.data.columns and self.current_idx < len(self.data):
                    price = self.data[self.current_idx, "ask1"]
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

            self.pending_orders.append(
                {
                    "action": action,
                    "volume": volume,
                    "price": price,  # 可能为None，将在执行时重新获取
                    "reason": reason,
                    "order_type": order_type,  # 保存订单类型
                    "execution_time": self.current_idx + 1,  # 在下一K线执行
                }
            )

            if log_callback and debug_mode:
                price_str = f"{price:.2f}" if price is not None else "待确定"
                log_callback(f"{self.symbol} {self.kline_period} 添加待执行订单: {action} {volume}手 订单类型:{order_type} 预计价格:{price_str} 原因:{reason}")

            return True

    def close_all(self, reason: str = "", log_callback=None, order_type="bar_close"):
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
        debug_mode = getattr(log_callback, "debug_mode", True) if log_callback else True

        if self.current_pos > 0:
            # 平多仓
            if order_type == "market":
                # 市价单，对于tick数据，卖出使用买一价格(bid1)
                price = None
                if "bid1" in self.data.columns and self.current_idx < len(self.data):
                    # 使用当前tick的买一价格作为成交价
                    price = self.data[self.current_idx, "bid1"]
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
            if order_type == "market":
                # 市价单，对于tick数据，买入使用卖一价格(ask1)
                price = None
                if "ask1" in self.data.columns and self.current_idx < len(self.data):
                    # 使用当前tick的卖一价格作为成交价
                    price = self.data[self.current_idx, "ask1"]
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
        return self.data["close"] if "close" in self.data.columns else pl.Series()

    def get_open(self) -> pl.Series:
        """获取开盘价序列"""
        return self.data["open"] if "open" in self.data.columns else pl.Series()

    def get_high(self) -> pl.Series:
        """获取最高价序列"""
        return self.data["high"] if "high" in self.data.columns else pl.Series()

    def get_low(self) -> pl.Series:
        """获取最低价序列"""
        return self.data["low"] if "low" in self.data.columns else pl.Series()

    def get_volume(self) -> pl.Series:
        """获取成交量序列"""
        return self.data["volume"] if "volume" in self.data.columns else pl.Series()

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

    def __str__(self):
        import json

        return json.dumps(
            {
                "symbol": self.symbol,
                "kline_period": self.kline_period,
                "data": "skip to avoid printing large data",
                "current_idx": self.current_idx,
                "current_pos": self.current_pos,
                "target_pos": self.target_pos,
                "signal_reason": self.signal_reason,
                "capital": self.capital,
                "available_capital": self.available_capital,
                "frozen_capital": self.frozen_capital,
                "unrealized_pnl": self.unrealized_pnl,
                "realized_pnl": self.realized_pnl,
                "comission": self.comission,
                "slippage": self.slippage,
                "margin_rate": self.margin_rate,
            },
            indent=4,
        )


class MultiDataSource:
    """
    多数据源管理类，用于管理多个数据源和统一资金管理
    """

    def __init__(self):
        """初始化多数据源管理器"""
        self.data_sources: List[DataSource] = []  # 数据源映射表，用于存储所有数据源对象
        self.log_callback = None  # 日志回调函数，用于记录操作日志
        self.double_dict: bidict = bidict({})  # 双向字典，用于数据源索引和名称之间的转换
        # 资金管理相关属性
        # self.total_capital = 0  # 总资金，包括初始资金和盈亏
        # self.available_capital = self.total_capital  # 可用资金，可用于开仓的资金
        # self.commission = trade_config.commission  # 手续费率
        # self.slippage = trade_config.slippage  # 滑点
        # self.total_margin_rate = trade_config.total_margin_rate  # 总保证金率
        # self.trade_type = trade_config.trade_type  # 交易类型，"逐仓或者全仓"
        # self.frozen_capital = 0.0  # 冻结资金总额
        # self.unrealized_pnl = 0.0  # 未实现盈亏总额
        # self.realized_pnl = 0.0  # 已实现盈亏总额

    def allocate_fund(self, trade_config: TradeConfig):
        """根据交易配置分配资金"""
        total_count = self.get_data_sources_count()
        total_capital = trade_config.total_capital
        sep_captital = np.round(total_capital / total_count, 0)
        for ds in self.data_sources:
            ds.capital = sep_captital
            ds.comission = trade_config.commission
            ds.slippage = trade_config.slippage
            ds.margin_rate = trade_config.total_margin_rate
            ds.available_capital = sep_captital
            ds.frozen_capital = 0.0
            ds.unrealized_pnl = 0.0
            ds.realized_pnl = 0.0

    def set_log_callback(self, callback):
        """设置日志回调函数"""
        self.log_callback = callback

    def add_data_source(self, symbol: str, kline_period: str, data: pl.DataFrame) -> int:
        """
        添加数据源

        Args:
            symbol: 品种代码，如'rb888'
            kline_period: K线周期，如'1h', 'D'
            adjust_type: 复权类型，'0'表示不复权，'1'表示后复权
            data: 数据，如果为None则创建空数据源

        Returns:
            数据源索引
        """
        # 将自身MultiDataSource实例传递给DataSource
        datasource_index = len(self.data_sources)
        datasource_key = f"{symbol}_{kline_period}"
        data_source = DataSource(index=datasource_index, symbol=symbol, kline_period=kline_period, data=data)
        self.data_sources.append(data_source)
        self.double_dict[datasource_index] = datasource_key

        return datasource_index

    def get_capital_info(self) -> dict:
        """
        获取总资金信息

        Returns:
            dict: 包含总资金信息的字典
        """
        return {
            "total_capital": self.total_capital,
            "available_capital": self.available_capital,
            "frozen_capital": self.frozen_capital,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_equity": self.total_capital + self.unrealized_pnl + self.realized_pnl,
        }

    def get_datasource_capital_info(self, index: int) -> Optional[dict]:
        """
        获取指定数据源的资金信息

        Args:
            index: 数据源索引

        Returns:
            dict: 包含该数据源资金信息的字典，如果索引无效则返回None
        """
        if index not in self.datasource_info:
            return None

        info = self.datasource_info[index].copy()
        ds = self.data_sources[index]

        # 计算当前持仓价值
        if ds.current_pos != 0 and ds.get_current_price() is not None:
            current_price = ds.get_current_price()
            info["position_value"] = abs(ds.current_pos) * current_price
            info["current_price"] = current_price
            info["current_position"] = ds.current_pos
        else:
            info["position_value"] = 0.0
            info["current_price"] = ds.get_current_price()
            info["current_position"] = ds.current_pos

        return info

    def get_all_datasources_capital_info(self) -> dict:
        """
        获取所有数据源的资金信息

        Returns:
            dict: 包含所有数据源资金信息的字典
        """
        result = {}
        for index in self.datasource_info:
            result[index] = self.get_datasource_capital_info(index)
        return result

    def _calculate_total_unrealized_pnl(self) -> float:
        """
        计算所有数据源的总未实现盈亏

        Returns:
            float: 总未实现盈亏
        """
        total_unrealized_pnl = 0.0

        for index, ds in enumerate(self.data_sources):
            if ds.current_pos != 0 and ds.get_current_price() is not None:
                current_price = ds.get_current_price()
                position_value = abs(ds.current_pos) * current_price

                # 计算该数据源的交易成本（简化处理，实际可能需要更复杂的计算）
                # 这里假设开仓价格存储在datasource_info中
                if "entry_price" in self.datasource_info[index]:
                    entry_price = self.datasource_info[index]["entry_price"]
                    if ds.current_pos > 0:  # 多头持仓
                        unrealized_pnl = (current_price - entry_price) * abs(ds.current_pos)
                    else:  # 空头持仓
                        unrealized_pnl = (entry_price - current_price) * abs(ds.current_pos)
                else:
                    unrealized_pnl = 0.0

                total_unrealized_pnl += unrealized_pnl
                self.datasource_info[index]["unrealized_pnl"] = unrealized_pnl
                self.datasource_info[index]["position_value"] = position_value

        return total_unrealized_pnl

    def _update_capital_after_trade(self, datasource_index: int, action: str, price: float, volume: int):
        """
        交易后更新资金信息

        Args:
            datasource_index: 数据源索引
            action: 交易动作
            price: 成交价格
            volume: 成交数量
        """
        ds = self.data_sources[datasource_index]
        info = self.datasource_info[datasource_index]

        if action in ["开多", "开空"]:
            # 开仓：冻结资金
            required_capital = price * volume
            if self.available_capital >= required_capital:
                self.available_capital -= required_capital
                self.frozen_capital += required_capital
                info["frozen_capital"] += required_capital
                info["entry_price"] = price

                if self.log_callback:
                    self.log_callback(f"{info['symbol']} 开仓冻结资金: {required_capital:.2f}, 可用资金: {self.available_capital:.2f}")

        elif action in ["平多", "平空"]:
            # 平仓：释放资金并计算盈亏
            released_capital = price * volume
            self.frozen_capital -= released_capital
            self.available_capital += released_capital
            info["frozen_capital"] -= released_capital

            # 计算已实现盈亏
            if "entry_price" in info:
                if action == "平多":
                    realized_pnl = (price - info["entry_price"]) * volume
                else:  # 平空
                    realized_pnl = (info["entry_price"] - price) * volume

                self.realized_pnl += realized_pnl
                info["realized_pnl"] += realized_pnl
                self.total_capital += realized_pnl
                self.available_capital += realized_pnl

                if self.log_callback:
                    self.log_callback(f"{info['symbol']} 平仓释放资金: {released_capital:.2f}, 已实现盈亏: {realized_pnl:.2f}")

            # 清除开仓价格
            if ds.current_pos == 0:
                info.pop("entry_price", None)

        elif action in ["平多开空", "平空开多"]:
            # 反手：先平仓再开仓
            old_pos = abs(ds.current_pos)
            released_capital = price * old_pos
            required_capital = price * old_pos

            # 平仓部分
            self.frozen_capital -= released_capital
            info["frozen_capital"] -= released_capital

            # 计算已实现盈亏
            if "entry_price" in info:
                if action == "平多开空":
                    realized_pnl = (price - info["entry_price"]) * old_pos
                else:  # 平空开多
                    realized_pnl = (info["entry_price"] - price) * old_pos

                self.realized_pnl += realized_pnl
                info["realized_pnl"] += realized_pnl
                self.total_capital += realized_pnl
                self.available_capital += realized_pnl

            # 开仓部分
            self.frozen_capital += required_capital
            info["frozen_capital"] += required_capital
            info["entry_price"] = price

            if self.log_callback:
                self.log_callback(f"{info['symbol']} 反手交易，已实现盈亏: {realized_pnl:.2f}")

        # 更新未实现盈亏
        self.unrealized_pnl = self._calculate_total_unrealized_pnl()

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

    def align_data(self, align_index: bool = True, fill_method: Optional[str] = "ffill"):
        """
        对齐所有数据源的数据

        Args:
            align_index: 是否对齐索引
            fill_method: 填充方法，可选值：'ffill', 'bfill', None
        """
        if len(self.data_sources) <= 1:
            return self  # 只有一个或没有数据源，不需要对齐

        if not align_index:
            return self  # 如果不对齐索引，直接返回

        # 收集所有数据源的索引（假设数据框有'datetime'列）
        all_datetime_values = []
        valid_datasources = []

        for i, ds in enumerate(self.data_sources):
            if not ds.data.is_empty() and "datetime" in ds.data.columns:
                datetime_series = ds.data["datetime"]
                if not datetime_series.is_empty():
                    all_datetime_values.append(set(datetime_series.to_list()))
                    valid_datasources.append(i)

        if not all_datetime_values:
            if self.log_callback:
                self.log_callback("No valid data sources found for alignment")
            return self  # 没有有效的数据源

        # 计算所有数据源共有的datetime值
        common_datetime = set.intersection(*all_datetime_values)

        # 统一转换为毫秒精度，避免精度不匹配问题
        common_datetime = {dt.replace(microsecond=dt.microsecond // 1000 * 1000) for dt in common_datetime}

        if not common_datetime:
            if self.log_callback:
                self.log_callback("Warning: No common time points found across all data sources, unable to align data")
            return self

        if self.log_callback:
            self.log_callback(f"Found {len(common_datetime)} common time points, starting alignment")

        # 创建一个参考数据框，包含共同的时间点
        if common_datetime:
            # 统一转换为毫秒精度
            common_datetime_ms = {dt.replace(microsecond=dt.microsecond // 1000 * 1000) for dt in common_datetime}
            reference_df = pl.DataFrame({"datetime": list(common_datetime_ms)}).sort("datetime")
            # 统一为毫秒精度
            reference_df = reference_df.with_columns(pl.col("datetime").dt.cast_time_unit("ms"))
        else:
            reference_df = pl.DataFrame({"datetime": []})
            reference_df = reference_df.with_columns(pl.col("datetime").cast(pl.Datetime("ms")))

        # 对齐所有数据源的数据
        aligned_count = 0
        for i, ds in enumerate(self.data_sources):
            if i in valid_datasources and not ds.data.is_empty():
                original_len = len(ds.data)

                # 统一时间精度为毫秒，避免精度不匹配问题
                if ds.data["datetime"].dtype.time_unit == "us":
                    ds.data = ds.data.with_columns(pl.col("datetime").dt.cast_time_unit("ms"))

                # 使用join来对齐数据，避免精度问题
                ds.data = reference_df.join(ds.data, on="datetime", how="left")

                # 按时间排序
                ds.data = ds.data.sort("datetime")

                # 根据fill_method填充缺失值
                if fill_method == "ffill":
                    ds.data = ds.data.fill_null(strategy="forward")
                elif fill_method == "bfill":
                    ds.data = ds.data.fill_null(strategy="backward")
                elif fill_method == "none" or fill_method is None:
                    # 不填充，保持null值
                    pass

                aligned_count += 1

                if self.log_callback:
                    final_len = len(ds.data)
                    self.log_callback(f"Data source {i} ({ds.symbol} {ds.kline_period}): " f"{original_len} -> {final_len} rows")
            elif not ds.data.is_empty():
                # 无效的数据源（没有datetime列）
                if self.log_callback:
                    self.log_callback(f"Warning: Data source {i} ({ds.symbol} {ds.kline_period}) has no datetime column, skipping alignment")

        if self.log_callback:
            self.log_callback(f"Data alignment completed, aligned {aligned_count} data sources")
        return self
