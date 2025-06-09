"""
PyFIXTest - A lightweight Python library for testing FIX Protocol trading systems.

This library provides tools and utilities for:
- Creating and managing FIX sessions
- Building and validating FIX messages
- Running automated tests for FIX-based trading workflows
- Simulating market conditions and order lifecycles
"""

__version__ = "0.1.0"
__author__ = "Atharva Joshi"
__email__ = "atharvajoshi477@gmail.com"

from .core.fix_engine import FIXEngine
from .core.message_factory import MessageFactory
from .core.session_manager import SessionManager
from .testing.base_test import BaseFIXTest
from .testing.test_helpers import OrderTestHelper, MarketDataTestHelper
from .testing.validators import MessageValidator, OrderValidator
from .testing.assertions import FIXAssertions

__all__ = [
    "FIXEngine",
    "MessageFactory", 
    "SessionManager",
    "BaseFIXTest",
    "OrderTestHelper",
    "MarketDataTestHelper",
    "MessageValidator",
    "OrderValidator",
    "FIXAssertions",
]