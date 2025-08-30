#!/usr/bin/env python3
"""
币安期货数据下载脚本
从币安数据网站下载BTCUSDT的30分钟K线数据并解压到指定目录
"""

import requests
import zipfile
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
import os
import argparse
import time

# 配置常量
CHUNK_SIZE = 8192
MAX_RETRIES = 3
RETRY_DELAY = 5
TIMEOUT = 30
RESUME_ENABLED = True


def get_file_size(file_path: Path) -> int:
    """获取文件大小"""
    if file_path.exists():
        return file_path.stat().st_size
    return 0


def download_file(url: str, save_path: Path, max_retries: int = MAX_RETRIES, 
                 retry_delay: int = RETRY_DELAY, timeout: int = TIMEOUT) -> bool:
    """
    下载文件，支持重试机制
    """
    for attempt in range(max_retries + 1):
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 检查是否支持断点续传
            existing_size = get_file_size(save_path)
            headers = {}
            
            if RESUME_ENABLED and existing_size > 0:
                headers['Range'] = f'bytes={existing_size}-'
                mode = 'ab'  # 追加模式
                print(f"正在续传: {url} (从 {existing_size} 字节开始)")
            else:
                mode = 'wb'  # 写入模式
                print(f"正在下载: {url} (尝试 {attempt + 1}/{max_retries + 1})")
            
            response = requests.get(url, headers=headers, stream=True, timeout=timeout)
            response.raise_for_status()
            
            with open(save_path, mode) as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            
            print(f"下载完成: {save_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                print(f"下载失败: {url}, 错误: {e}")
                print(f"{retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                print(f"下载失败，已达最大重试次数: {url}, 错误: {e}")
                return False
        except Exception as e:
            print(f"下载失败: {url}, 错误: {e}")
            return False
    
    return False


def unzip_file(zip_path: Path, extract_to: Path) -> bool:
    """
    解压文件
    """
    try:
        print(f"正在解压: {zip_path}")
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"解压完成: {extract_to}")
        return True
    except Exception as e:
        print(f"解压失败: {zip_path}, 错误: {e}")
        return False


def generate_date_range(start_date: str, end_date: str) -> List[str]:
    """
    生成日期范围列表
    格式: YYYY-MM-DD
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        date_list = []
        current = start
        while current <= end:
            date_list.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        return date_list
    except ValueError as e:
        print(f"日期格式错误: {e}")
        return []


def get_last_timestamp(output_file: Path) -> Optional[int]:
    """
    获取现有文件的最后时间戳
    """
    if not output_file.exists():
        return None
    
    try:
        df = pd.read_csv(output_file)
        if df.empty:
            return None
        
        # 假设第一列是timestamp
        last_timestamp = df.iloc[-1, 0]
        if pd.notna(last_timestamp):
            return int(float(last_timestamp))
        return None
    except Exception as e:
        print(f"读取现有文件失败: {e}")
        return None


def get_existing_date_ranges(output_file: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    获取现有数据覆盖的日期范围
    """
    if not output_file.exists():
        return None, None
    
    try:
        df = pd.read_csv(output_file)
        if df.empty:
            return None, None
        
        # 假设第一列是timestamp
        timestamps = df.iloc[:, 0].astype(int)
        
        # 转换为日期字符串
        dates = []
        for ts in timestamps:
            dt = datetime.fromtimestamp(ts / 1000)
            dates.append(dt.strftime('%Y-%m-%d'))
        
        # 获取最早和最晚的日期
        unique_dates = sorted(list(set(dates)))
        return unique_dates[0], unique_dates[-1]
        
    except Exception as e:
        print(f"读取现有文件失败: {e}")
        return None, None


def find_missing_dates(target_dates: List[str], existing_start: Optional[str], 
                      existing_end: Optional[str]) -> List[str]:
    """
    找出目标日期范围内缺失的日期
    """
    if existing_start is None or existing_end is None:
        return target_dates
    
    # 将现有日期范围内的所有日期转换为set
    existing_dates = set()
    current = datetime.strptime(existing_start, '%Y-%m-%d')
    end = datetime.strptime(existing_end, '%Y-%m-%d')
    
    while current <= end:
        existing_dates.add(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    # 找出目标日期中不在现有日期集合中的日期
    missing_dates = []
    for date in target_dates:
        if date not in existing_dates:
            missing_dates.append(date)
    
    return missing_dates


def calculate_date_range(start_date: str, end_date: str, symbol: str, interval: str, 
                        incremental: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """
    计算需要下载的日期范围
    """
    if not incremental:
        return start_date, end_date
    
    # 检查现有文件
    output_dir = Path(f"resource/data/swap/{symbol}/{interval}/")
    output_file = output_dir / f"{symbol}.csv"
    
    if not output_file.exists():
        return start_date, end_date
    
    # 获取现有数据的日期范围
    existing_start, existing_end = get_existing_date_ranges(output_file)
    
    if existing_start is None or existing_end is None:
        return start_date, end_date
    
    print(f"检测到现有数据，日期范围: {existing_start} 到 {existing_end}")
    
    # 生成目标日期范围
    target_dates = generate_date_range(start_date, end_date)
    
    # 找出缺失的日期
    missing_dates = find_missing_dates(target_dates, existing_start, existing_end)
    
    if not missing_dates:
        print("没有缺失的数据需要下载")
        return None, None
    
    print(f"发现 {len(missing_dates)} 天的缺失数据需要下载")
    
    # 返回最早和最晚的缺失日期
    return min(missing_dates), max(missing_dates)


def merge_to_single_file(csv_files: List[Path], symbol: str, interval: str) -> None:
    """
    合并CSV文件到单一文件
    """
    if not csv_files:
        return
    
    output_dir = Path(f"resource/data/swap/{symbol}/{interval}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{symbol}.csv"
    
    print(f"正在合并到单一文件: {output_file}")
    
    # 读取所有CSV文件
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"读取文件失败: {csv_file}, 错误: {e}")
    
    if not dfs:
        print("没有成功读取任何CSV文件")
        return
    
    # 合并所有DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 将open_time列重命名为timestamp
    if 'open_time' in merged_df.columns:
        merged_df = merged_df.rename(columns={'open_time': 'timestamp'})
    
    # 按timestamp排序
    merged_df = merged_df.sort_values('timestamp')
    
    # 如果输出文件已存在，读取现有数据并合并
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file)
            # 合并现有数据和新数据
            combined_df = pd.concat([existing_df, merged_df], ignore_index=True)
            # 去重并排序
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
            combined_df = combined_df.sort_values('timestamp')
            merged_df = combined_df
            print(f"已合并现有数据，总共 {len(merged_df)} 行")
        except Exception as e:
            print(f"读取现有文件失败，将创建新文件: {e}")
    
    # 保存合并后的文件
    merged_df.to_csv(output_file, index=False)
    
    print(f"合并完成: {output_file}")
    print(f"总共合并了 {len(csv_files)} 个文件，{len(merged_df)} 行数据")
    
    # 删除原始文件
    for csv_file in csv_files:
        csv_file.unlink()
    print(f"已删除 {len(csv_files)} 个原始CSV文件")


def download_binance_data(symbol: str, interval: str, start_date: str, end_date: str, 
                         clean_zip: bool = True, incremental: bool = True, 
                         max_retries: int = MAX_RETRIES, retry_delay: int = RETRY_DELAY, 
                         timeout: int = TIMEOUT) -> None:
    """
    下载币安期货数据
    """
    # 计算需要下载的日期范围
    actual_start, actual_end = calculate_date_range(
        start_date, end_date, symbol, interval, incremental
    )
    
    if actual_start is None:
        print("没有新数据需要下载")
        return
    
    # 生成完整的日期范围
    target_dates = generate_date_range(start_date, end_date)
    
    # 如果是增量下载，找出缺失的日期
    if incremental:
        output_dir = Path(f"resource/data/swap/{symbol}/{interval}/")
        output_file = output_dir / f"{symbol}.csv"
        
        if output_file.exists():
            existing_start, existing_end = get_existing_date_ranges(output_file)
            dates = find_missing_dates(target_dates, existing_start, existing_end)
        else:
            dates = target_dates
    else:
        dates = target_dates
    
    # 过滤掉在计算范围之外的日期
    if actual_start and actual_end:
        dates = [d for d in dates if actual_start <= d <= actual_end]
    
    if not dates:
        return
    
    # 设置基础URL和输出目录
    base_url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol.upper()}/{interval}/"
    output_dir = Path(f"resource/data/swap/{symbol}/{interval}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载 {len(dates)} 天的数据...")
    
    success_count = 0
    downloaded_csv_files = []
    
    for date in dates:
        # 构建文件名和URL
        zip_filename = f"{symbol.upper()}-{interval}-{date}.zip"
        url = f"{base_url}{zip_filename}"
        
        # 下载路径
        zip_path = output_dir / zip_filename
        
        # 下载文件
        if download_file(url, zip_path, max_retries, retry_delay, timeout):
            success_count += 1
            
            # 解压文件
            if unzip_file(zip_path, output_dir):
                # 找到解压后的CSV文件
                csv_file = output_dir / f"{symbol.upper()}-{interval}-{date}.csv"
                if csv_file.exists():
                    downloaded_csv_files.append(csv_file)
                
                # 清理zip文件
                if clean_zip:
                    zip_path.unlink()
                    print(f"已删除zip文件: {zip_path}")
        else:
            print(f"跳过日期: {date}")
    
    print(f"下载完成! 成功下载 {success_count}/{len(dates)} 天的数据")
    
    # 合并到单一文件
    if downloaded_csv_files:
        print("\n开始合并到单一CSV文件...")
        merge_to_single_file(downloaded_csv_files, symbol, interval)






def main():
    """
    主函数
    """
    print("币安期货数据下载脚本")
    print("=" * 50)
    
    symbol = input("请输入交易对 (如: btcusdt): ").strip().lower()
    interval = input("请输入时间间隔 (如: 30m): ").strip()
    start_date = input("请输入开始日期 (YYYY-MM-DD): ").strip()
    end_date = input("请输入结束日期 (YYYY-MM-DD): ").strip()
    
    clean_zip = input("是否删除zip文件? (y/n): ").strip().lower() == 'y'
    incremental = input("是否增量下载? (y/n): ").strip().lower() == 'y'
    
    download_binance_data(symbol, interval, start_date, end_date, clean_zip, incremental)


if __name__ == "__main__":
    import sys
    
    # 如果有命令行参数，直接执行下载
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='币安期货数据下载脚本')
        parser.add_argument('--symbol', type=str, required=True, help='交易对 (如: btcusdt)')
        parser.add_argument('--interval', type=str, required=True, help='时间间隔 (如: 30m)')
        parser.add_argument('--start', type=str, required=True, help='开始日期 (YYYY-MM-DD)')
        parser.add_argument('--end', type=str, required=True, help='结束日期 (YYYY-MM-DD)')
        parser.add_argument('--keep-zip', action='store_true', help='保留zip文件')
        parser.add_argument('--full-download', action='store_true', help='全量下载，不增量')
        
        args = parser.parse_args()
        
        download_binance_data(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start,
            end_date=args.end,
            clean_zip=not args.keep_zip,
            incremental=not args.full_download
        )
    else:
        main()