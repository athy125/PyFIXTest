"""
FIX configuration management for different environments and scenarios.
"""

import os
from typing import Dict, Any, Optional
import configparser
import quickfix as fix

from ..utils.logging_config import get_logger


class FIXConfig:
    """
    FIX configuration manager for different environments and testing scenarios.
    
    Handles:
    - Session configuration
    - Connection settings
    - Protocol parameters
    - Environment-specific settings
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.config_data: Dict[str# PyFIXTest - FIX Protocol Testing Library