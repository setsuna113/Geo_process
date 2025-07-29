"""Decorators for automatic logging and error capture."""

import functools
import time
import inspect
from typing import Callable, Any, Optional, TypeVar, Union
from datetime import datetime

from .structured_logger import get_logger
from .context import stage_context, node_context

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


def log_operation(operation_name: Optional[str] = None, 
                  log_args: bool = False,
                  log_result: bool = False,
                  log_performance: bool = True):
    """Decorator to log operation execution and capture errors.
    
    Args:
        operation_name: Custom operation name (defaults to function name)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_performance: Whether to log performance metrics
        
    Example:
        @log_operation("data_validation")
        def validate_data(data):
            # Function code here
            pass
    """
    def decorator(func: F) -> F:
        # Get operation name
        name = operation_name or func.__name__
        
        # Get logger for the module
        logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            # Build context
            context = {'operation': name}
            
            # Add function arguments if requested
            if log_args:
                # Get argument names
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Filter out large objects
                arg_info = {}
                for arg_name, arg_value in bound_args.arguments.items():
                    if isinstance(arg_value, (str, int, float, bool)):
                        arg_info[arg_name] = arg_value
                    else:
                        arg_info[arg_name] = f"<{type(arg_value).__name__}>"
                
                context['arguments'] = arg_info
            
            try:
                # Log operation start
                logger.info(f"Starting {name}", extra={'context': context})
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log result if requested
                if log_result:
                    if isinstance(result, (str, int, float, bool, list, dict)):
                        context['result'] = result
                    else:
                        context['result'] = f"<{type(result).__name__}>"
                
                # Log performance
                if log_performance:
                    duration = time.time() - start_time
                    logger.log_performance(name, duration, status='success')
                else:
                    logger.info(f"Completed {name}", extra={'context': context})
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log error with full context
                logger.error(
                    f"Failed {name}: {str(e)}", 
                    exc_info=True,
                    extra={
                        'context': context,
                        'performance': {
                            'duration': duration,
                            'status': 'failed',
                            'error_type': type(e).__name__
                        }
                    }
                )
                raise
        
        return wrapper  # type: ignore
    return decorator


def log_stage(stage_name: str, track_progress: bool = True):
    """Decorator for pipeline stages with automatic context.
    
    Args:
        stage_name: Name of the pipeline stage
        track_progress: Whether to track progress updates
        
    Example:
        @log_stage("data_loading")
        def execute(self, context):
            # Stage implementation
            pass
    """
    def decorator(func: F) -> F:
        logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(self, context, *args, **kwargs) -> Any:
            # Use logging context if available
            if hasattr(context, 'logging_context'):
                with context.logging_context.stage(stage_name):
                    return func(self, context, *args, **kwargs)
            else:
                # Fallback without full context
                start_time = time.time()
                
                # Set stage context
                stage_context.set(stage_name)
                
                # Create node ID if possible
                current_node = node_context.get()
                if current_node:
                    new_node = f"{current_node}/{stage_name}"
                else:
                    new_node = f"stage/{stage_name}"
                node_context.set(new_node)
                
                try:
                    logger.info(f"Stage started: {stage_name}")
                    result = func(self, context, *args, **kwargs)
                    
                    duration = time.time() - start_time
                    logger.log_performance(f"stage_{stage_name}", duration, status='completed')
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(
                        f"Stage failed: {stage_name}",
                        exc_info=True,
                        extra={
                            'performance': {
                                'duration': duration,
                                'status': 'failed'
                            }
                        }
                    )
                    raise
                finally:
                    # Clean up context
                    stage_context.set(None)
                    if current_node:
                        node_context.set(current_node)
        
        return wrapper  # type: ignore
    return decorator


def log_checkpoint(checkpoint_name: str):
    """Decorator for checkpoint operations.
    
    Args:
        checkpoint_name: Name of the checkpoint
        
    Example:
        @log_checkpoint("model_state")
        def save_checkpoint(self, data):
            # Save checkpoint
            pass
    """
    def decorator(func: F) -> F:
        logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                logger.info(f"Saving checkpoint: {checkpoint_name}")
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.info(
                    f"Checkpoint saved: {checkpoint_name}",
                    extra={
                        'performance': {
                            'operation': f"checkpoint_{checkpoint_name}",
                            'duration_seconds': duration
                        }
                    }
                )
                
                return result
                
            except Exception as e:
                logger.error(
                    f"Checkpoint failed: {checkpoint_name}",
                    exc_info=True
                )
                raise
        
        return wrapper  # type: ignore
    return decorator


def retry_with_logging(max_attempts: int = 3, 
                      delay: float = 1.0,
                      backoff: float = 2.0,
                      exceptions: tuple = (Exception,)):
    """Decorator to retry operations with exponential backoff and logging.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff: Backoff multiplier for each retry
        exceptions: Tuple of exceptions to catch and retry
        
    Example:
        @retry_with_logging(max_attempts=3, exceptions=(ConnectionError,))
        def connect_to_service():
            # Connection code
            pass
    """
    def decorator(func: F) -> F:
        logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    if attempt > 0:
                        logger.warning(
                            f"Retrying {func.__name__} (attempt {attempt + 1}/{max_attempts})"
                        )
                    
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed: {str(e)}. "
                            f"Retrying in {current_delay}s...",
                            extra={
                                'context': {
                                    'attempt': attempt + 1,
                                    'max_attempts': max_attempts,
                                    'delay': current_delay,
                                    'error_type': type(e).__name__
                                }
                            }
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts",
                            exc_info=True
                        )
            
            if last_exception:
                raise last_exception
                
        return wrapper  # type: ignore
    return decorator