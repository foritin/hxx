import pandas as pd
import polars as pl
from typing import Dict, List, Any, Optional, Union, Callable
from extend.data.data_source import DataSource
from extend.data.data_source import MultiDataSource


class StrategyAPI:
    """
    策略API核心类，只提供数据访问和交易操作，不包含指标计算
    """

    def __init__(self, context: Dict):
        """
        初始化策略API

        Args:
            context: 策略上下文，包含数据源、日志函数和参数等
        """
        self._context = context
        self._data: MultiDataSource = context["data"]
        self._log = context["log"]
        self._params = context.get("params", {})

    def log(self, message: str):
        """
        记录日志

        Args:
            message: 日志消息
        """
        self._log(message)

    def get_params(self) -> Dict:
        """
        获取策略参数

        Returns:
            策略参数字典
        """
        return self._params

    def get_param(self, key: str, default=None):
        """
        获取指定参数

        Args:
            key: 参数名
            default: 默认值

        Returns:
            参数值，如果不存在则返回默认值
        """
        return self._params.get(key, default)

    def get_data_source(self, datasource_mark: Union[str, int] = 0) -> DataSource:
        """
        获取指定索引的数据源

        Args:
            index: 数据源索引，默认为0（第一个数据源）

        Returns:
            数据源对象，如果索引无效则返回None
        """
        if isinstance(datasource_mark, int):
            index = datasource_mark

        else:
            assert isinstance(datasource_mark, str)
            index = self._data.double_dict.inverse[datasource_mark]

        return self._data[index]

    def get_data_sources_count(self) -> int:
        """
        获取数据源数量

        Returns:
            数据源数量
        """
        return len(self._data)

    def require_data_sources(self, count: int) -> bool:
        """
        确保至少有指定数量的数据源

        Args:
            count: 最少需要的数据源数量

        Returns:
            是否满足要求
        """
        if len(self._data) < count:
            self.log(f"策略需要至少 {count} 个数据源，当前只有 {len(self._data)} 个")
            return False
        return True

    def get_klines(self, datasource_mark: Union[str, int] = 0) -> pl.DataFrame:
        """
        获取指定数据源的K线数据

        Args:
            index: 数据源索引，默认为0（第一个数据源）

        Returns:
            K线数据DataFrame
        """
        ds = self.get_data_source(datasource_mark=datasource_mark)

        return ds.get_klines()

    def get_datetime(self, datasource_mark: Union[str, int] = 0):
        """
        获取当前日期时间

        Args:
            index: 数据源索引，默认为0（第一个数据源）

        Returns:
            当前日期时间
        """
        ds = self.get_data_source(datasource_mark=datasource_mark)

        return ds.get_current_datetime()

    def get_price(self, datasource_mark: Union[str, int] = 0) -> Optional[float]:
        """
        获取当前价格

        Args:
            index: 数据源索引，默认为0（第一个数据源）

        Returns:
            当前价格，如果数据源无效则返回None
        """
        ds = self.get_data_source(datasource_mark=datasource_mark)

        return ds.get_current_price()

    def get_pos(self, datasource_mark: Union[str, int] = 0) -> float:
        """
        获取当前持仓

        Args:
            index: 数据源索引，默认为0（第一个数据源）

        Returns:
            当前持仓，如果数据源无效则返回0
        """
        ds = self.get_data_source(datasource_mark)
        return ds.get_current_pos()

    def get_idx(self, datasource_mark: Union[str, int] = 0) -> int:
        """
        获取当前索引

        Args:
            index: 数据源索引，默认为0（第一个数据源）

        Returns:
            当前索引，如果数据源无效则返回-1
        """
        ds = self.get_data_source(datasource_mark=datasource_mark)
        if ds:
            return ds.current_idx
        return -1

    def get_close(self, datasource_mark: Union[str, int] = 0) -> pl.Series:
        """
        获取收盘价序列

        Args:
            index: 数据源索引，默认为0（第一个数据源）

        Returns:
            收盘价序列
        """
        ds = self.get_data_source(datasource_mark=datasource_mark)
        return ds.get_close()

    def get_open(self, datasource_mark: Union[str, int] = 0) -> pl.Series:
        """
        获取开盘价序列

        Args:
            index: 数据源索引，默认为0（第一个数据源）

        Returns:
            开盘价序列
        """
        ds = self.get_data_source(datasource_mark=datasource_mark)

        return ds.get_open()

    def get_high(self, datasource_mark: Union[str, int] = 0) -> pl.Series:
        """
        获取最高价序列

        Args:
            index: 数据源索引，默认为0（第一个数据源）

        Returns:
            最高价序列
        """
        ds = self.get_data_source(datasource_mark=datasource_mark)
        return ds.get_high()

    def get_low(self, datasource_mark: Union[str, int] = 0) -> pl.Series:
        """
        获取最低价序列

        Args:
            index: 数据源索引，默认为0（第一个数据源）

        Returns:
            最低价序列
        """
        ds = self.get_data_source(datasource_mark=datasource_mark)
        return ds.get_low()

    def get_volume(self, datasource_mark: Union[str, int] = 0) -> pl.Series:
        """
        获取成交量序列

        Args:
            index: 数据源索引，默认为0（第一个数据源）

        Returns:
            成交量序列
        """
        ds = self.get_data_source(datasource_mark=datasource_mark)
        return ds.get_volume()

    def buy(self, datasource_mark: Union[str, int], order_percent: float = 0.0, reason: str = "", order_type: str = "bar_close", index: int = 0):
        """
        买入开仓

        Args:
            order_percent: 交易百分比，默认为0（不交易）
            reason: 交易原因
            order_type: 订单类型，可选值：
                - 'bar_close': 当前K线收盘价（默认）
                - 'next_bar_open': 下一K线开盘价
                - 'next_bar_close': 下一K线收盘价
                - 'next_bar_high': 下一K线最高价
                - 'next_bar_low': 下一K线最低价
                - 'market': 市价单，tick策略中按ask1价格成交（买入用卖一价）
            index: 数据源索引，默认为0（第一个数据源）
        """
        ds = self.get_data_source(datasource_mark=datasource_mark)

        ds.buy(order_percent=order_percent, reason=reason, log_callback=self._log, order_type=order_type)

    def sell(self, datasource_mark: Union[str, int], order_percent: float = 0.0, reason: str = "", order_type: str = "bar_close", index: int = 0):
        """
        卖出开仓

        Args:
            volume: 交易量，默认为全部持仓
            reason: 交易原因
            order_type: 订单类型，可选值：
                - 'bar_close': 当前K线收盘价（默认）
                - 'next_bar_open': 下一K线开盘价
                - 'next_bar_close': 下一K线收盘价
                - 'next_bar_high': 下一K线最高价
                - 'next_bar_low': 下一K线最低价
                - 'market': 市价单，tick策略中按bid1价格成交（卖出用买一价）
            index: 数据源索引，默认为0（第一个数据源）
        """
        ds = self.get_data_source(datasource_mark=datasource_mark)
        ds.sell(order_percent=order_percent, reason=reason, log_callback=self._log, order_type=order_type)

    def close_long(self, datasource_mark: Union[int, str], order_percent: Optional[float] = None, reason: str = "", order_type: str = "bar_close", index: int = 0):
        """
        平多

        Args:
            volume: 交易量，默认为1
            reason: 交易原因
            order_type: 订单类型，可选值：
                - 'bar_close': 当前K线收盘价（默认）
                - 'next_bar_open': 下一K线开盘价
                - 'next_bar_close': 下一K线收盘价
                - 'next_bar_high': 下一K线最高价
                - 'next_bar_low': 下一K线最低价
                - 'market': 市价单，tick策略中按bid1价格成交（卖出用买一价）
            index: 数据源索引，默认为0（第一个数据源）
        """
        ds = self.get_data_source(datasource_mark=datasource_mark)
        ds.close_long(order_percent=order_percent, reason=reason, log_callback=self._log, order_type=order_type)

    def close_short(self, datasource_mark: Union[int, str], order_percent: Optional[float] = None, reason: str = "", order_type: str = "bar_close", index: int = 0):
        """
        买入平仓（平空）

        Args:
            volume: 交易量，默认为全部持仓
            reason: 交易原因
            order_type: 订单类型，可选值：
                - 'bar_close': 当前K线收盘价（默认）
                - 'next_bar_open': 下一K线开盘价
                - 'next_bar_close': 下一K线收盘价
                - 'next_bar_high': 下一K线最高价
                - 'next_bar_low': 下一K线最低价
                - 'market': 市价单，tick策略中按ask1价格成交（买入用卖一价）
            index: 数据源索引，默认为0（第一个数据源）
        """
        ds = self.get_data_source(datasource_mark=datasource_mark)
        ds.close_short(order_percent=order_percent, reason=reason, log_callback=self._log, order_type=order_type)

    # def buytocover(self, volume: Optional[int] = None, reason: str = "", order_type: str = "bar_close", index: int = 0):
    #     """
    #     买入平仓（平空）- 兼容buytocover别名

    #     Args:
    #         volume: 交易量，默认为全部持仓
    #         reason: 交易原因
    #         order_type: 订单类型，同buycover
    #         index: 数据源索引，默认为0（第一个数据源）
    #     """
    #     return self.buycover(volume=volume, reason=reason, order_type=order_type, index=index)

    # def close_all(self, reason: str = "", order_type: str = "bar_close", index: int = 0):
    #     """
    #     平仓所有持仓

    #     Args:
    #         reason: 交易原因
    #         order_type: 订单类型，可选值：
    #             - 'bar_close': 当前K线收盘价（默认）
    #             - 'next_bar_open': 下一K线开盘价
    #             - 'next_bar_close': 下一K线收盘价
    #             - 'next_bar_high': 下一K线最高价
    #             - 'next_bar_low': 下一K线最低价
    #             - 'market': 市价单，tick策略中按对手价成交（买入ask1，卖出bid1）
    #         index: 数据源索引，默认为0（第一个数据源）
    #     """
    #     ds = self.get_data_source(index)
    #     if ds:
    #         ds.close_all(reason=reason, log_callback=self._log, order_type=order_type)

    # def reverse_pos(self, reason: str = "", order_type: str = "bar_close", index: int = 0):
    #     """
    #     反转持仓

    #     Args:
    #         reason: 交易原因
    #         order_type: 订单类型，可选值：
    #             - 'bar_close': 当前K线收盘价（默认）
    #             - 'next_bar_open': 下一K线开盘价
    #             - 'next_bar_close': 下一K线收盘价
    #             - 'next_bar_high': 下一K线最高价
    #             - 'next_bar_low': 下一K线最低价
    #             - 'market': 市价单，tick策略中按对手价成交（买入ask1，卖出bid1）
    #         index: 数据源索引，默认为0（第一个数据源）
    #     """
    #     ds = self.get_data_source(index)
    #     if ds:
    #         ds.reverse_pos(reason=reason, log_callback=self._log, order_type=order_type)

    # def get_tick(self, index: int = 0):
    #     """
    #     获取当前tick的所有字段（Series）
    #     Args:
    #         index: 数据源索引，默认为0
    #     Returns:
    #         当前tick的所有字段（Series），若无数据则返回None
    #     """
    #     ds = self.get_data_source(index)
    #     if ds:
    #         return ds.get_tick()
    #     return None

    # def get_ticks(self, window: int = 100, index: int = 0):
    #     """
    #     获取最近window条tick数据（DataFrame）
    #     Args:
    #         window: 滑窗长度，默认100
    #         index: 数据源索引，默认为0
    #     Returns:
    #         最近window条tick数据（DataFrame），若无数据则返回空DataFrame
    #     """
    #     ds = self.get_data_source(index)
    #     if ds:
    #         return ds.get_ticks(window=window)
    #     return pd.DataFrame()


# 创建策略API工厂函数
def create_strategy_api(context: Dict) -> StrategyAPI:
    """
    从context创建策略API

    Args:
        context: 策略上下文

    Returns:
        策略API对象
    """
    return StrategyAPI(context)
