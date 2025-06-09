"""
Base test class providing common testing infrastructure for FIX protocol tests.
"""

import unittest
import time
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from ..core.fix_engine import FIXEngine, SessionState
from ..core.message_factory import MessageFactory
from ..config.fix_config import FIXConfig
from ..utils.logging_config import get_logger, setup_test_logging
from .assertions import FIXAssertions


class BaseFIXTest(unittest.TestCase, FIXAssertions):
    """
    Base class for FIX protocol tests.
    
    Provides:
    - FIX engine setup and teardown
    - Common test utilities
    - Message handling helpers
    - Session management
    - Custom assertions for FIX testing
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - called once per test class."""
        setup_test_logging()
        cls.logger = get_logger(cls.__name__)
    
    def setUp(self):
        """Set up individual test - called before each test method."""
        self.logger.info(f"Starting test: {self._testMethodName}")
        
        # Initialize FIX components
        self.config = self._create_test_config()
        self.engine = FIXEngine(self.config)
        self.message_factory = MessageFactory()
        
        # Test state
        self.received_messages: List[Any] = []
        self.test_start_time = time.time()
        
        # Set up message handlers
        self._setup_message_handlers()
    
    def tearDown(self):
        """Clean up after each test method."""
        try:
            if hasattr(self, 'engine') and self.engine:
                self.engine.stop()
                time.sleep(0.5)  # Allow cleanup
            
            test_duration = time.time() - self.test_start_time
            self.logger.info(f"Test {self._testMethodName} completed in {test_duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error in tearDown: {e}")
    
    def _create_test_config(self) -> FIXConfig:
        """
        Create test-specific FIX configuration.
        Override this method to provide custom configuration.
        
        Returns:
            FIXConfig: Test configuration
        """
        config = FIXConfig()
        config.set_defaults_for_testing()
        return config
    
    def _setup_message_handlers(self):
        """Set up default message handlers for tests."""
        # Store all received messages
        def store_message(message, session_id):
            self.received_messages.append({
                'message': message,