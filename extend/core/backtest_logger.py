import os
from datetime import datetime
from loguru import logger

class BacktestLogger:
    """回测日志管理器类，负责处理日志记录、日志文件创建等功能"""
    
    def __init__(self, debug_mode=False):
        """初始化日志管理器"""
        self.debug_mode = debug_mode
        self.log_file = None
        self.performance_file = None
        self.logger_id = None
        
    def set_debug_mode(self, debug_mode):
        """设置调试模式
        
        Args:
            debug_mode: 是否开启调试模式
        """
        self.debug_mode = debug_mode
        self._update_logger_config()
    
    def _update_logger_config(self):
        """更新日志配置"""
        # 移除之前的处理器
        if self.logger_id is not None:
            logger.remove(self.logger_id)
        
        # 配置控制台输出
        if self.debug_mode:
            logger.add(
                sink=lambda msg: print(msg, end=""),
                level="INFO",
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                colorize=True
            )
        
        # 配置文件输出
        if self.log_file:
            self.logger_id = logger.add(
                sink=self.log_file,
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="10 MB",
                retention="30 days",
                encoding="utf-8",
                enqueue=True,
                backtrace=True,
                diagnose=True
            )
    
    def prepare_log_file(self, symbols_and_periods):
        """准备日志文件
        
        Args:
            symbols_and_periods: 品种和周期列表
            
        Returns:
            log_file_path: 日志文件路径
        """
        # 创建日志目录
        log_dir = "output/logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbols_str = "_".join([item["symbol"] for item in symbols_and_periods])
        self.log_file = os.path.join(log_dir, f"backtest_{symbols_str}_{timestamp}.log")
        
        # 创建综合绩效报告文件
        results_dir = "output/results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.performance_file = os.path.join(results_dir, f"performance_{symbols_str}_{timestamp}.txt")
        
        # 更新日志配置
        self._update_logger_config()
        
        # 写入日志头
        logger.info(f"多数据源回测日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"回测品种: {symbols_str}")
        logger.info("=" * 80)
            
        return self.log_file
    
    def log_message(self, message):
        """记录日志消息
        
        Args:
            message: 日志消息
        """
        logger.info(message)
    
    def log_debug(self, message):
        """记录调试消息
        
        Args:
            message: 调试消息
        """
        logger.debug(message)
    
    def log_warning(self, message):
        """记录警告消息
        
        Args:
            message: 警告消息
        """
        logger.warning(message)
    
    def log_error(self, message):
        """记录错误消息
        
        Args:
            message: 错误消息
        """
        logger.error(message)
    
    def get_performance_file(self):
        """获取绩效报告文件路径"""
        # 确保绩效报告文件目录存在
        if self.performance_file and not os.path.exists(os.path.dirname(self.performance_file)):
            os.makedirs(os.path.dirname(self.performance_file))
            
        return self.performance_file
    
    def __del__(self):
        """析构函数，清理日志处理器"""
        if self.logger_id is not None:
            logger.remove(self.logger_id)