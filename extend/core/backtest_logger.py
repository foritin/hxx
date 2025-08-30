import os
from datetime import datetime
from loguru import logger


class BacktestLogger:
    """回测日志管理器类，负责处理日志记录、日志文件创建等功能"""

    # 单例实例
    _instance = None

    # 类变量，用于跟踪全局处理器
    _console_handler_id = None
    _file_handler_id = None

    def __new__(cls, debug_mode=False):
        """单例模式，确保只有一个实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.debug_mode = debug_mode
            cls._instance.log_file = None
            cls._instance.performance_file = None
        return cls._instance

    def __init__(self, debug_mode=False):
        """初始化日志管理器"""
        # 更新调试模式
        self.debug_mode = debug_mode

    def set_debug_mode(self, debug_mode):
        """设置调试模式

        Args:
            debug_mode: 是否开启调试模式
        """
        self.debug_mode = debug_mode
        self._update_logger_config()

    def _update_logger_config(self):
        """更新日志配置"""
        # 移除之前的全局处理器
        if BacktestLogger._console_handler_id is not None:
            try:
                logger.remove(BacktestLogger._console_handler_id)
                BacktestLogger._console_handler_id = None
            except ValueError:
                # 处理器可能已经被移除
                pass

        if BacktestLogger._file_handler_id is not None:
            try:
                logger.remove(BacktestLogger._file_handler_id)
                BacktestLogger._file_handler_id = None
            except ValueError:
                # 处理器可能已经被移除
                pass

        # 移除默认的 stderr 处理器（id=0）
        try:
            logger.remove(0)
        except ValueError:
            pass

        # 配置控制台输出（只添加一次）
        if self.debug_mode and BacktestLogger._console_handler_id is None:
            BacktestLogger._console_handler_id = logger.add(
                sink=lambda msg: print(msg, end=""),
                level="INFO",
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                colorize=True,
            )

        # 配置文件输出（只添加一次）
        if self.log_file and BacktestLogger._file_handler_id is None:
            BacktestLogger._file_handler_id = logger.add(
                sink=self.log_file,
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="10 MB",
                retention="30 days",
                encoding="utf-8",
                enqueue=True,
                backtrace=True,
                diagnose=True,
            )

    def log_message(self, message):
        """记录日志消息

        Args:
            message: 日志消息
        """
        # 检查是否已经有处理器配置
        if not (BacktestLogger._console_handler_id or BacktestLogger._file_handler_id):
            # 如果没有处理器，直接打印
            print(message)
        else:
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
        # 清理所有处理器
        if BacktestLogger._console_handler_id is not None:
            try:
                logger.remove(BacktestLogger._console_handler_id)
                BacktestLogger._console_handler_id = None
            except ValueError:
                pass

        if BacktestLogger._file_handler_id is not None:
            try:
                logger.remove(BacktestLogger._file_handler_id)
                BacktestLogger._file_handler_id = None
            except ValueError:
                pass
