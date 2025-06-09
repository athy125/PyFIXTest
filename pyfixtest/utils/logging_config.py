"""
Logging configuration for PyFIXTest.
"""

import logging
import sys
from typing import Optional
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


def setup_logging(level: str = 'INFO', format_string: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        format_string: Custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatter
    formatter = ColoredFormatter(format_string)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Set QuickFIX logging level to WARNING to reduce noise
    logging.getLogger('quickfix').setLevel(logging.WARNING)


def setup_test_logging():
    """Set up logging optimized for testing."""
    setup_logging(
        level='DEBUG',
        format_string='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(name)


def setup_file_logging(log_file: str, level: str = 'DEBUG'):
    """
    Set up file logging in addition to console logging.
    
    Args:
        log_file: Path to log file
        level: Logging level for file
    """
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(getattr(logging, level.upper()))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
