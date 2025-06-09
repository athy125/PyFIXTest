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
                'session_id': session_id,
                'timestamp': time.time()
            })
        
        # Register handler for all message types
        common_msg_types = ['8', 'D', 'F', 'G', 'V', 'W', '0', '1']
        for msg_type in common_msg_types:
            self.engine.add_message_handler(msg_type, store_message)
    
    @contextmanager
    def fix_session(self, start_as_initiator: bool = True, wait_for_login: bool = True):
        """
        Context manager for FIX session lifecycle.
        
        Args:
            start_as_initiator: Whether to start as initiator or acceptor
            wait_for_login: Whether to wait for login confirmation
            
        Yields:
            FIXEngine: The active FIX engine
        """
        try:
            # Start engine
            if start_as_initiator:
                success = self.engine.start_initiator()
            else:
                success = self.engine.start_acceptor()
            
            self.assertTrue(success, "Failed to start FIX engine")
            
            # Wait for login if requested
            if wait_for_login:
                login_success = self.engine.wait_for_login(timeout=30.0)
                self.assertTrue(login_success, "Failed to log in within timeout")
            
            yield self.engine
            
        finally:
            # Cleanup is handled in tearDown
            pass
    
    def send_and_wait_for_response(
        self,
        message,
        expected_msg_type: str,
        timeout: float = 10.0
    ):
        """
        Send message and wait for specific response type.
        
        Args:
            message: FIX message to send
            expected_msg_type: Expected response message type
            timeout: Timeout in seconds
            
        Returns:
            Received response message or None
        """
        initial_count = len(self.received_messages)
        
        # Send message
        success = self.engine.send_message(message)
        self.assertTrue(success, "Failed to send message")
        
        # Wait for response
        response = self.engine.wait_for_message(expected_msg_type, timeout)
        
        return response
    
    def wait_for_messages(self, count: int, timeout: float = 10.0) -> bool:
        """
        Wait for specific number of messages to be received.
        
        Args:
            count: Number of messages to wait for
            timeout: Timeout in seconds
            
        Returns:
            bool: True if received expected count within timeout
        """
        start_time = time.time()
        initial_count = len(self.received_messages)
        target_count = initial_count + count
        
        while time.time() - start_time < timeout:
            if len(self.received_messages) >= target_count:
                return True
            time.sleep(0.1)
        
        return False
    
    def get_last_message_of_type(self, msg_type: str):
        """
        Get the last received message of specific type.
        
        Args:
            msg_type: FIX message type
            
        Returns:
            Last message of specified type or None
        """
        for msg_data in reversed(self.received_messages):
            try:
                message = msg_data['message']
                if message.getHeader().getField(35) == msg_type:
                    return message
            except:
                continue
        return None
    
    def clear_received_messages(self):
        """Clear all received messages."""
        self.received_messages.clear()
    
    def simulate_market_conditions(self, symbol: str, price_range: tuple = (100.0, 110.0)):
        """
        Simulate basic market conditions for testing.
        
        Args:
            symbol: Trading symbol
            price_range: Price range (min, max)
        """
        # This is a placeholder for market simulation
        # In real implementation, this could send market data updates
        self.logger.info(f"Simulating market for {symbol} in range {price_range}")