# HXX - 多数据源量化回测框架

## 项目简介

HXX 是一个基于 Python 的多数据源量化交易回测框架，支持币安期货数据下载、处理和策略回测。

## 主要特性

- **多数据源支持**: 支持币安期货数据、A股数据等多种数据源
- **高性能数据处理**: 使用 Polars 和 PyArrow 进行高效数据处理
- **模块化设计**: 清晰的模块结构，易于扩展和维护
- **策略回测**: 完整的回测引擎，支持多种交易策略
- **数据管理**: 自动数据下载、缓存和管理

## 项目结构

```
hxx/
├── extend/                 # 核心扩展模块
│   ├── api/               # 策略API接口
│   ├── core/              # 核心回测引擎
│   ├── data/              # 数据处理模块
│   └── utils.py           # 工具函数
├── script/                # 数据下载脚本
│   ├── download_binance_data.py     # 币安数据下载
│   ├── binance_futures_data.py     # 币安期货数据处理
│   └── cn_stock_full_presistent.py # A股数据处理
├── resource/              # 资源文件
│   ├── data/              # 数据存储
│   └── package/           # 依赖包
├── output/                # 输出目录
└── main.py                # 主程序入口
```

## 依赖要求

- Python >= 3.13
- polars >= 1.32.0
- pyarrow >= 21.0.0
- pandas-ta >= 0.3.14b0
- ccxt >= 4.4.98
- akshare >= 1.17.26
- loguru >= 0.7.0
- pydantic >= 2.11.7

## 安装

```bash
# 克隆项目
git clone <repository-url>
cd hxx

# 安装依赖
uv sync
```

## 使用方法

### 下载币安数据

```bash
# 运行币安数据下载脚本
python script/download_binance_data.py
```

### 运行回测

```bash
# 运行主程序
python main.py
```

## 核心模块

### 回测引擎 (extend/core/backtest_engine.py)
- 多数据源回测支持
- 可配置的回测参数
- 自动数据对齐和处理

### 数据管理 (extend/data/)
- 统一的数据加载接口
- 支持多种数据格式
- 数据缓存机制

### 策略API (extend/api/strategy_api.py)
- 标准化的策略接口
- 丰富的交易功能
- 完整的风险管理

## 配置说明

项目支持灵活的配置系统，主要配置项包括：

- 数据源配置
- 回测参数配置
- 品种特定配置
- 调试和日志配置

## 开发指南

1. **添加新数据源**: 在 `extend/data/` 目录下创建新的数据源类
2. **开发策略**: 使用 `extend/api/strategy_api.py` 中的接口
3. **扩展功能**: 在 `extend/core/` 目录下添加新的核心功能

## 许可证

MIT License