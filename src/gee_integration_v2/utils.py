"""
工具函数模块
===========

通用工具函数，如重试装饰器等。

作者: Linus式重构
"""

import time
import logging
import ee
from typing import Callable

logger = logging.getLogger(__name__)


def retry_on_failure(max_retries=3, delay=5, backoff=2):
    """改进的重试装饰器 - 支持指数退避
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟（秒）
        backoff: 退避系数
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # 尝试从self.config获取参数
            if hasattr(args[0], 'config'):
                config = args[0].config
                actual_retries = getattr(config, 'max_retries', max_retries)
                actual_delay = getattr(config, 'retry_delay', delay)
            else:
                actual_retries = max_retries
                actual_delay = delay
            
            last_exception = None
            current_delay = actual_delay
            
            for attempt in range(actual_retries):
                try:
                    return func(*args, **kwargs)
                except (ee.EEException, TimeoutError, ConnectionError) as e:
                    # 只重试网络相关的异常
                    last_exception = e
                    if attempt == actual_retries - 1:
                        logger.error(f"Failed after {actual_retries} attempts: {e}")
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1}/{actual_retries} failed: {e}")
                    logger.info(f"Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff  # 指数退避
                except Exception as e:
                    # 其他异常直接抛出
                    logger.error(f"Non-retryable error: {e}")
                    raise
            
            raise last_exception
        
        return wrapper
    return decorator