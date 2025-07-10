import os
import pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests.exceptions
from tools import PathManager
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange, Interval


# 设置pandas显示选项，避免科学记数法
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# 获取全市场股票代码
def get_all_fund_etf_codes():
    df = ak.fund_etf_spot_em()
    code_name_df = df[["代码", "名称"]]
    return list(code_name_df.values)


def get_fund_etf_data(code: str, name: str, adjust: str = "hfq", period: str = "daily", start_date: str = "20150101"):
    """
    Args:
        code: 股票代码
        adjust: 复权方式
        period: 周期

    Returns:
        DataFrame: 股票数据
    """
    column_mapping = {
        "股票代码": "code",
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",  # 注意：单位是"股"
        "成交额": "turnover",  # 注意：单位是"元"
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover_rate",
    }
    df = ak.stock_zh_a_hist(symbol=code, period=period, adjust=adjust, start_date=start_date)
    print(f"成功获取股票 {code} 的数据，共 {len(df)} 条记录")
    if "成交量" in df.columns:
        df["成交量"] = df["成交量"].apply(lambda x: f"{x:,.0f}")

    # 格式化成交额列
    if "成交额" in df.columns:
        df["成交额"] = df["成交额"].apply(lambda x: f"{x:,.2f}")

    data_df = df.rename(columns=column_mapping)
    if data_df.empty:
        return data_df

    data_df["date"] = pd.to_datetime(data_df["date"])
    data_df["name"] = name
    data_df["open"] = data_df["open"].astype(float)
    data_df["close"] = data_df["close"].astype(float)
    data_df["high"] = data_df["high"].astype(float)
    data_df["low"] = data_df["low"].astype(float)
    data_df["volume"] = data_df["volume"].str.replace(",", "").astype(int)
    data_df["turnover"] = data_df["turnover"].str.replace(",", "").astype(float)
    data_df["amplitude"] = data_df["amplitude"].astype(float)
    data_df["pct_change"] = data_df["pct_change"].astype(float)
    data_df["change"] = data_df["change"].astype(float)
    data_df["turnover_rate"] = data_df["turnover_rate"].astype(float)

    return data_df


def split_by_year(df):
    """
    按年切分DataFrame

    Args:
        df: 包含日期列的DataFrame

    Returns:
        dict: 以年份为key，DataFrame为value的字典
    """
    # 确保日期列是datetime类型
    df["date"] = pd.to_datetime(df["date"])

    # 提取年份
    df["year"] = df["date"].dt.year

    # 按年份分组
    yearly_data = {}
    for year in df["year"].unique():
        year_df = df[df["year"] == year].copy()
        # 删除临时添加的年份列
        year_df = year_df.drop("year", axis=1)
        yearly_data[year] = year_df

    return yearly_data


def process():
    db = get_database()
    objects = get_all_fund_etf_codes()
    for object in objects:
        code, name = object
        stock_df = get_fund_etf_data(code, name)
        bar_data_list = []
        if not stock_df.empty:
            for idx, row in stock_df.iterrows():
                bar = BarData(
                    symbol=f"{code}_etfs",
                    exchange=Exchange.LOCAL,
                    datetime=row["date"].to_pydatetime(),
                    interval=Interval.DAILY,
                    open_price=row["open"],
                    high_price=row["high"],
                    low_price=row["low"],
                    close_price=row["close"],
                    volume=row["volume"],
                    turnover=row["turnover"],
                    gateway_name="LOCAL",
                )
                bar_data_list.append(bar)
            db.save_bar_data(bar_data_list)


if __name__ == "__main__":
    # fund_etf_spot_em_df = ak.fund_etf_spot_em()
    # print(fund_etf_spot_em_df)
    # fund_etf_hist_em_df = ak.fund_etf_hist_em(symbol="515790", period="daily", adjust="hfq")
    # print(fund_etf_hist_em_df)
    process()
