from pydantic import BaseModel, ValidationError, Field, field_validator
from typing import List, Optional
from enum import Enum
from datetime import datetime


class SourceType(str, Enum):
    CSV = "csv"
    PARQUET = "parquet"


class TradeType(str, Enum):
    ISOLATED = "isolated"
    CROSS = "cross"


class SymbolType(str, Enum):
    SPOT = "spot"
    SWAP = "swap"


class Period(str, Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    FOUR_HOURS = "4h"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"


class EnhanceBaseModel(BaseModel):

    def from_dict(cls, params_dict: dict):
        return cls(**params_dict)
    
    def to_dict(cls):
        return cls.model_dump()


class BaseConfig(EnhanceBaseModel):
    use_cache: bool = False
    debug: bool = True
    save_data: bool = True
    align_data: bool = True
    fill_method: str = "ffill"
    # Quantstats报告配置
    enable_quantstats_report: bool = True
    include_benchmark: bool = True  # 是否在quantstats报告中包含基线对比（买入持有策略）


class TradeConfig(EnhanceBaseModel):
    total_capital: float = Field(100000, description="Total capital for backtest")
    commission: float = 0.001
    slippage: float = Field(0.000, description="Slippage rate")
    leverage: float = Field(1.0, description="leverage")
    total_margin_rate: float = Field(0.3, description="Total margin rate")
    trade_type: TradeType = TradeType.CROSS


class SingleSymbolConfig(EnhanceBaseModel):
    symbol: str = Field(..., description="Symbol name like btcusdt with lowcase")
    start_date: str = Field(..., description="Start date for backtest, like 2020-01-01", pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., description="End date for backtest, like 2020-12-31", pattern=r"^\d{4}-\d{2}-\d{2}$")
    symbol_type: str = Field(SymbolType.SWAP, description="Symbol type like spot, swap")
    source_type: str = Field(SourceType.CSV, description="Data type for backtest, like csv, parquet, etc.")
    isolated_margin_rate: float = Field(0.3, description="Isolated margin rate")
    periods: List[Period] = Field([Period.FIVE_MINUTES], description="Periods for backtest")

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_validity(cls, v):
        """Validate date is valid (not just format)"""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid date, please ensure date is valid (e.g., 2020-01-01)")
        return v


class StrategyParams(EnhanceBaseModel):
    model_config = {"extra": "allow"}
    
    def __init__(self, **data):
        super().__init__(**data)
    


class OptimizationParams(EnhanceBaseModel):
    model_config = {"extra": "allow"}
    
    def __init__(self, **data):
        super().__init__(**data)
    


class StrategyConfig(BaseModel):
    base_config: BaseConfig
    trade_config: TradeConfig
    symbol_configs: List[SingleSymbolConfig]
    strategy_params: StrategyParams
    optimization_params: OptimizationParams