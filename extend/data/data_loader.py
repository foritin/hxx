import polars as pl
from typing import Optional, Union
from pathlib import Path


def load_local_data(
    file_path: Union[str, Path], 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> pl.DataFrame:
    """
    加载本地K线数据文件，支持日期范围过滤
    
    Args:
        file_path: 数据文件路径
        start_date: 开始日期 (格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS)
        end_date: 结束日期 (格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS)
    
    Returns:
        pl.DataFrame: 处理后的数据框
        
    Raises:
        ValueError: 文件格式错误或缺少必需列
        FileNotFoundError: 文件不存在
    """
    # 检查文件是否存在
    if not Path(file_path).exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 读取CSV文件，尝试自动解析日期
    try:
        df = pl.read_csv(file_path, try_parse_dates=True)
    except Exception as e:
        raise ValueError(f"读取CSV文件失败: {e}")
    
    # 检查必需列
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"K线数据缺少必需列: {missing_columns}")
    
    # 处理时间列
    df = _process_datetime_column(df)
    
    # 日期范围过滤
    df = _filter_by_date_range(df, start_date, end_date)
    
    # 选择最终列
    final_columns = required_columns + ["datetime"]
    df = df.select(final_columns)
    
    # 按时间排序
    df = df.sort("datetime")
    
    return df


def _process_datetime_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    处理时间列，确保有datetime列
    """
    # 如果已经有datetime列，确保格式正确
    if "datetime" in df.columns:
        # 确保是datetime类型
        if df["datetime"].dtype != pl.Datetime:
            df = df.with_columns(
                pl.col("datetime").str.to_datetime().alias("datetime")
            )
        return df
    
    # 如果有timestamp列，转换为datetime
    if "timestamp" in df.columns:
        # 尝试将时间戳转换为datetime
        try:
            # 如果timestamp是字符串类型，先转换为整数
            if df["timestamp"].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.col("timestamp").cast(pl.Int64).alias("timestamp")
                )
            
            # 将毫秒时间戳转换为datetime
            df = df.with_columns(
                pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime")
            )
            return df
        except Exception as e:
            raise ValueError(f"时间戳转换失败: {e}")
    
    # 如果没有时间列，抛出错误
    raise ValueError("数据中缺少datetime或timestamp列")


def _filter_by_date_range(
    df: pl.DataFrame, 
    start_date: Optional[str], 
    end_date: Optional[str]
) -> pl.DataFrame:
    """
    按日期范围过滤数据
    """
    if not start_date and not end_date:
        return df
    
    filters = []
    
    # 处理开始日期
    if start_date:
        try:
            # 尝试解析为datetime
            if len(start_date) == 10:  # YYYY-MM-DD
                start_dt = pl.col("datetime") >= pl.lit(start_date).str.to_datetime()
            else:  # YYYY-MM-DD HH:MM:SS
                start_dt = pl.col("datetime") >= pl.lit(start_date).str.to_datetime()
            filters.append(start_dt)
        except Exception as e:
            raise ValueError(f"开始日期格式错误: {start_date}, 期望格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS")
    
    # 处理结束日期
    if end_date:
        try:
            # 结束日期包含整天，所以加一天
            if len(end_date) == 10:  # YYYY-MM-DD
                end_dt = pl.col("datetime") < (pl.lit(end_date).str.to_datetime() + pl.duration(days=1))
            else:  # YYYY-MM-DD HH:MM:SS
                end_dt = pl.col("datetime") <= pl.lit(end_date).str.to_datetime()
            filters.append(end_dt)
        except Exception as e:
            raise ValueError(f"结束日期格式错误: {end_date}, 期望格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS")
    
    # 应用过滤器
    if filters:
        df = df.filter(pl.all_horizontal(filters))
    
    return df





if __name__ == '__main__':
    # 测试基本功能
    file_path = "/home/rex/project/hxx/resource/data/swap/btcusdt/30m/btcusdt.csv"
    
    try:
        print("测试1: 加载全部数据")
        df = load_local_data(file_path)
        print(f"数据形状: {df.shape}")
        print(f"数据列: {df.columns}")
        print(f"数据类型: {df.dtypes}")
        print(f"前5行:")
        print(df.head())
        print()
        
        print("测试2: 日期范围过滤")
        df_filtered = load_local_data(file_path, start_date="2024-01-02", end_date="2024-01-03")
        print(f"过滤后数据形状: {df_filtered.shape}")
        print(f"过滤后数据范围: {df_filtered['datetime'].min()} 到 {df_filtered['datetime'].max()}")
        print()
        
        print("测试3: 精确时间范围过滤")
        df_precise = load_local_data(file_path, start_date="2024-01-02 00:00:00", end_date="2024-01-02 12:00:00")
        print(f"精确过滤后数据形状: {df_precise.shape}")
        print(f"精确过滤后数据范围: {df_precise['datetime'].min()} 到 {df_precise['datetime'].max()}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    