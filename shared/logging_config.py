"""
Centralized logging configuration for the Ensemble Pipeline project.
This module provides consistent logging setup across backend API and frontend.
"""

import logging
import os
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting and handlers.
    
    Args:
        name: Name of the logger (typically module name)
        log_file: Optional log file path (relative to output/logs/)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    
    # Ensure logs directory exists
    log_dir = Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter with emojis for better readability
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        file_path = log_dir / log_file
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_api_logger() -> logging.Logger:
    """Get logger for API backend."""
    return setup_logger('ensemble_api', 'api.log')

def get_frontend_logger() -> logging.Logger:
    """Get logger for frontend."""
    return setup_logger('ensemble_frontend', 'frontend.log')

def get_pipeline_logger() -> logging.Logger:
    """Get logger for pipeline operations."""
    return setup_logger('ensemble_pipeline', 'pipeline.log')

def log_api_call(logger: logging.Logger, method: str, endpoint: str, **kwargs):
    """
    Log an API call with standardized format.
    
    Args:
        logger: Logger instance
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint
        **kwargs: Additional info to log (params, status, time, etc.)
    """
    
    msg_parts = [f"{method} {endpoint}"]
    
    if 'params' in kwargs and kwargs['params']:
        msg_parts.append(f"Params: {kwargs['params']}")
    
    if 'status' in kwargs:
        status_emoji = "✅" if kwargs['status'] < 400 else "❌"
        msg_parts.append(f"Status: {status_emoji} {kwargs['status']}")
    
    if 'time' in kwargs:
        msg_parts.append(f"Time: {kwargs['time']:.3f}s")
    
    if 'data_info' in kwargs:
        msg_parts.append(f"Data: {kwargs['data_info']}")
    
    logger.info(f"🔌 API CALL: {' - '.join(msg_parts)}")

def log_data_operation(logger: logging.Logger, operation: str, data_type: str, **kwargs):
    """
    Log a data operation with standardized format.
    
    Args:
        logger: Logger instance
        operation: Type of operation (load, save, fetch, etc.)
        data_type: Type of data (metrics, predictions, sweep, etc.)
        **kwargs: Additional info (count, source, etc.)
    """
    
    msg_parts = [f"{operation.upper()} {data_type}"]
    
    if 'count' in kwargs:
        msg_parts.append(f"Count: {kwargs['count']}")
    
    if 'source' in kwargs:
        msg_parts.append(f"Source: {kwargs['source']}")
    
    if 'success' in kwargs:
        result_emoji = "✅" if kwargs['success'] else "❌"
        msg_parts.append(f"Result: {result_emoji}")
    
    logger.info(f"📊 DATA: {' - '.join(msg_parts)}")

# Example usage and testing
if __name__ == "__main__":
    # Test the logging setup
    api_logger = get_api_logger()
    frontend_logger = get_frontend_logger()
    pipeline_logger = get_pipeline_logger()
    
    # Test API call logging
    log_api_call(api_logger, "GET", "/results/metrics", status=200, time=0.123, data_info="3 models")
    log_api_call(frontend_logger, "GET", "/results/predictions", status=404, time=0.045)
    
    # Test data operation logging
    log_data_operation(pipeline_logger, "load", "training_data", count=1000, source="CSV", success=True)
    log_data_operation(frontend_logger, "fetch", "sweep_data", source="API", success=False)
    
    print("Logging configuration test completed. Check output/logs/ for log files.") 