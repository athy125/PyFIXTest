"""
Unit tests for FIX Engine components.

This module provides comprehensive unit tests for the FIX engine,
including engine configuration, application handling, statistics tracking,
and core engine functionality.
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timezone
import tempfile
import os
import shutil

# Import the classes to test
from pyfixtest.core.fix_engine import FIXEngine, SessionState
from pyfixtest.config.fix_config import FIXConfig, SessionConfig, NetworkConfig, SecurityConfig, PerformanceConfig
from pyfixtest.utils.logging_config import get_logger

# Mock QuickFIX if not available
try:
    import quickfix as fix
except ImportError:
    # Create mock QuickFIX module for testing
    fix = MagicMock()
    fix.Application = object
    fix.Message = MagicMock
    fix.SessionID = MagicMock
    fix.Session = MagicMock
    fix.SocketInitiator = MagicMock
    fix.SocketAcceptor = MagicMock
    fix.FileStoreFactory = MagicMock
    fix.FileLogFactory = MagicMock
    fix.FieldNotFound = Exception


class TestFIXEngine(unittest.TestCase):
    """Test FIXEngine core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = FIXConfig()
        self.config.session.sender_comp_id = "TEST_SENDER"
        self.config.session.target_comp_id = "TEST_TARGET"
        self.config.session.begin_string = "FIX.4.4"
        self.config.session.socket_connect_host = "localhost"
        self.config.session.socket_connect_port = 9876
        self.config.session.heartbeat_interval = 30
        self.config.store_path = os.path.join(self.temp_dir, "store")
        self.config.log_path = os.path.join(self.temp_dir, "logs")
        
        # Create directories
        os.makedirs(self.config.store_path, exist_ok=True)
        os.makedirs(self.config.log_path, exist_ok=True)
        
        self.engine = FIXEngine(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'engine') and self.engine:
            try:
                self.engine.stop()
            except:
                pass
        
        # Clean up temp directory
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_engine_initialization(self):
        """Test FIX engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.config, self.config)
        self.assertEqual(self.engine.state, SessionState.DISCONNECTED)
        self.assertIsNone(self.engine.session_id)
        self.assertIsNone(self.engine.application)
        self.assertIsNone(self.engine.initiator)
        self.assertIsNone(self.engine.acceptor)
        self.assertIsInstance(self.engine.message_handlers, dict)
        self.assertIsInstance(self.engine.received_messages, list)
    
    @patch('pyfixtest.core.fix_engine.fix.SocketInitiator')
    @patch('pyfixtest.core.fix_engine.fix.FileLogFactory')
    @patch('pyfixtest.core.fix_engine.fix.FileStoreFactory')
    def test_start_initiator_success(self, mock_store_factory, mock_log_factory, mock_initiator):
        """Test successful initiator startup."""
        # Setup mocks
        mock_initiator_instance = Mock()
        mock_initiator.return_value = mock_initiator_instance
        
        # Test
        result = self.engine.start_initiator()
        
        # Assertions
        self.assertTrue(result)
        self.assertEqual(self.engine.state, SessionState.CONNECTING)
        self.assertIsNotNone(self.engine.application)
        self.assertIsNotNone(self.engine.initiator)
        mock_initiator_instance.start.assert_called_once()
    
    @patch('pyfixtest.core.fix_engine.fix.SocketInitiator')
    def test_start_initiator_failure(self, mock_initiator):
        """Test initiator startup failure."""
        # Setup mock to raise exception
        mock_initiator.side_effect = Exception("Connection failed")
        
        # Test
        result = self.engine.start_initiator()
        
        # Assertions
        self.assertFalse(result)
        self.assertEqual(self.engine.state, SessionState.ERROR)
    
    @patch('pyfixtest.core.fix_engine.fix.SocketAcceptor')
    @patch('pyfixtest.core.fix_engine.fix.FileLogFactory')
    @patch('pyfixtest.core.fix_engine.fix.FileStoreFactory')
    def test_start_acceptor_success(self, mock_store_factory, mock_log_factory, mock_acceptor):
        """Test successful acceptor startup."""
        # Setup mocks
        mock_acceptor_instance = Mock()
        mock_acceptor.return_value = mock_acceptor_instance
        
        # Test
        result = self.engine.start_acceptor()
        
        # Assertions
        self.assertTrue(result)
        self.assertEqual(self.engine.state, SessionState.CONNECTING)
        self.assertIsNotNone(self.engine.application)
        self.assertIsNotNone(self.engine.acceptor)
        mock_acceptor_instance.start.assert_called_once()
    
    @patch('pyfixtest.core.fix_engine.fix.SocketAcceptor')
    def test_start_acceptor_failure(self, mock_acceptor):
        """Test acceptor startup failure."""
        # Setup mock to raise exception
        mock_acceptor.side_effect = Exception("Accept failed")
        
        # Test
        result = self.engine.start_acceptor()
        
        # Assertions
        self.assertFalse(result)
        self.assertEqual(self.engine.state, SessionState.ERROR)
    
    def test_stop_engine(self):
        """Test engine stop functionality."""
        # Setup mocks
        self.engine.initiator = Mock()
        self.engine.acceptor = Mock()
        
        # Test
        self.engine.stop()
        
        # Assertions
        self.engine.initiator.stop.assert_called_once()
        self.engine.acceptor.stop.assert_called_once()
        self.assertEqual(self.engine.state, SessionState.DISCONNECTED)
        self.assertIsNone(self.engine.initiator)
        self.assertIsNone(self.engine.acceptor)
    
    @patch('pyfixtest.core.fix_engine.fix.Session.sendToTarget')
    def test_send_message_success(self, mock_send_to_target):
        """Test successful message sending."""
        # Setup
        mock_message = Mock()
        mock_session_id = Mock()
        self.engine.session_id = mock_session_id
        
        # Test
        result = self.engine.send_message(mock_message)
        
        # Assertions
        self.assertTrue(result)
        mock_send_to_target.assert_called_once_with(mock_message, mock_session_id)
    
    def test_send_message_no_session(self):
        """Test message sending without active session."""
        # Setup
        mock_message = Mock()
        self.engine.session_id = None
        
        # Test
        result = self.engine.send_message(mock_message)
        
        # Assertions
        self.assertFalse(result)
    
    @patch('pyfixtest.core.fix_engine.fix.Session.sendToTarget')
    def test_send_message_exception(self, mock_send_to_target):
        """Test message sending with exception."""
        # Setup
        mock_message = Mock()
        mock_session_id = Mock()
        self.engine.session_id = mock_session_id
        mock_send_to_target.side_effect = Exception("Send failed")
        
        # Test
        result = self.engine.send_message(mock_message)
        
        # Assertions
        self.assertFalse(result)
    
    def test_wait_for_message_found(self):
        """Test waiting for message that exists."""
        # Setup
        mock_message = Mock()
        mock_header = Mock()
        mock_header.getField.return_value = 'D'  # New Order Single
        mock_message.getHeader.return_value = mock_header
        
        self.engine.received_messages = [mock_message]
        
        # Test
        result = self.engine.wait_for_message('D', timeout=1.0)
        
        # Assertions
        self.assertEqual(result, mock_message)
        self.assertEqual(len(self.engine.received_messages), 0)  # Should be removed
    
    def test_wait_for_message_timeout(self):
        """Test waiting for message with timeout."""
        # Test
        result = self.engine.wait_for_message('D', timeout=0.1)
        
        # Assertions
        self.assertIsNone(result)
    
    def test_add_message_handler(self):
        """Test adding message handler."""
        # Setup
        handler = Mock()
        
        # Test
        self.engine.add_message_handler('D', handler)
        
        # Assertions
        self.assertIn('D', self.engine.message_handlers)
        self.assertIn(handler, self.engine.message_handlers['D'])
    
    def test_remove_message_handler(self):
        """Test removing message handler."""
        # Setup
        handler = Mock()
        self.engine.add_message_handler('D', handler)
        
        # Test
        self.engine.remove_message_handler('D', handler)
        
        # Assertions
        self.assertNotIn(handler, self.engine.message_handlers.get('D', []))
    
    def test_get_session_state(self):
        """Test getting session state."""
        # Test initial state
        self.assertEqual(self.engine.get_session_state(), SessionState.DISCONNECTED)
        
        # Test state change
        self.engine.state = SessionState.LOGGED_IN
        self.assertEqual(self.engine.get_session_state(), SessionState.LOGGED_IN)
    
    def test_is_logged_in(self):
        """Test logged in status check."""
        # Test not logged in
        self.assertFalse(self.engine.is_logged_in())
        
        # Test logged in
        self.engine.state = SessionState.LOGGED_IN
        self.assertTrue(self.engine.is_logged_in())
    
    def test_wait_for_login_success(self):
        """Test waiting for login success."""
        # Setup - simulate login after short delay
        def delayed_login():
            time.sleep(0.1)
            self.engine.state = SessionState.LOGGED_IN
        
        login_thread = threading.Thread(target=delayed_login)
        login_thread.start()
        
        # Test
        result = self.engine.wait_for_login(timeout=1.0)
        
        # Cleanup
        login_thread.join()
        
        # Assertions
        self.assertTrue(result)
    
    def test_wait_for_login_timeout(self):
        """Test waiting for login with timeout."""
        # Test
        result = self.engine.wait_for_login(timeout=0.1)
        
        # Assertions
        self.assertFalse(result)


class TestFIXApplication(unittest.TestCase):
    """Test FIXTestApplication functionality."""
    
def setUp(self):
        """Set up test fixtures."""
        from pyfixtest.core.fix_engine import FIXTestApplication
        
        self.engine = Mock()
        self.engine.session_id = None
        self.engine.state = SessionState.DISCONNECTED
        self.engine.received_messages = []
        self.engine.message_handlers = {}
        self.engine._lock = threading.Lock()
        
        self.application = FIXTestApplication(self.engine)
    
def test_on_create(self):
        """Test session creation callback."""
        mock_session_id = Mock()
        
        # Test
        self.application.onCreate(mock_session_id)
        
        # Assertions
        self.assertEqual(self.engine.session_id, mock_session_id)
    
def test_on_logon(self):
        """Test logon callback."""
        mock_session_id = Mock()
        
        # Test
        self.application.onLogon(mock_session_id)
        
        # Assertions
        self.assertEqual(self.engine.state, SessionState.LOGGED_IN)
    
def test_on_logout(self):
        """Test logout callback."""
        mock_session_id = Mock()
        
        # Test
        self.application.onLogout(mock_session_id)
        
        # Assertions
        self.assertEqual(self.engine.state, SessionState.LOGGED_OUT)
    
def test_to_admin(self):
        """Test admin message sending."""
        mock_message = Mock()
        mock_session_id = Mock()
        
        # Test - should not raise exception
        self.application.toAdmin(mock_message, mock_session_id)
    
def test_from_admin(self):
        """Test admin message receiving."""
        mock_message = Mock()
        mock_session_id = Mock()
        
        # Test - should not raise exception
        self.application.fromAdmin(mock_message, mock_session_id)
    
def test_to_app(self):
        """Test application message sending."""
        mock_message = Mock()
        mock_session_id = Mock()
        
        # Test - should not raise exception
        self.application.toApp(mock_message, mock_session_id)
    
def test_from_app(self):
        """Test application message receiving."""
        # Setup
        mock_message = Mock()
        mock_session_id = Mock()
        mock_header = Mock()
        mock_header.getField.return_value = 'D'  # New Order Single
        mock_message.getHeader.return_value = mock_header
        
        # Add message handler
        handler = Mock()
        self.engine.message_handlers['D'] = [handler]
        
        # Test
        self.application.fromApp(mock_message, mock_session_id)
        
        # Assertions
        self.assertIn(mock_message, self.engine.received_messages)
        handler.assert_called_once_with(mock_message, mock_session_id)
    
def test_from_app_with_handler_exception(self):
        """Test application message receiving with handler exception."""
        # Setup
        mock_message = Mock()
        mock_session_id = Mock()
        mock_header = Mock()
        mock_header.getField.return_value = 'D'
        mock_message.getHeader.return_value = mock_header
        
        # Add handler that raises exception
        handler = Mock(side_effect=Exception("Handler error"))
        self.engine.message_handlers['D'] = [handler]
        
        # Test - should not raise exception
        self.application.fromApp(mock_message, mock_session_id)
        
        # Assertions
        self.assertIn(mock_message, self.engine.received_messages)
        handler.assert_called_once()


class TestEngineConfiguration(unittest.TestCase):
    """Test FIX engine configuration handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = FIXConfig()
        self.config.store_path = os.path.join(self.temp_dir, "store")
        self.config.log_path = os.path.join(self.temp_dir, "logs")
        
        # Create directories
        os.makedirs(self.config.store_path, exist_ok=True)
        os.makedirs(self.config.log_path, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_configuration_defaults(self):
        """Test default configuration values."""
        config = FIXConfig()
        
        # Test session defaults
        self.assertEqual(config.session.begin_string, "FIX.4.4")
        self.assertEqual(config.session.heartbeat_interval, 30)
        self.assertEqual(config.session.connection_type, "initiator")
        
        # Test network defaults
        self.assertEqual(config.network.host, "localhost")
        self.assertEqual(config.network.port, 9876)
        self.assertEqual(config.network.connect_timeout, 30)
        
        # Test security defaults
        self.assertFalse(config.security.ssl_enabled)
        self.assertFalse(config.security.authentication_enabled)
        
        # Test performance defaults
        self.assertEqual(config.performance.message_queue_size, 10000)
        self.assertEqual(config.performance.worker_threads, 4)
    
    def test_session_config_to_dict(self):
        """Test session configuration dictionary conversion."""
        session_config = SessionConfig()
        session_config.sender_comp_id = "TEST_SENDER"
        session_config.target_comp_id = "TEST_TARGET"
        session_config.heartbeat_interval = 30
        
        result = session_config.to_dict()
        
        self.assertEqual(result['SenderCompID'], "TEST_SENDER")
        self.assertEqual(result['TargetCompID'], "TEST_TARGET")
        self.assertEqual(result['HeartBtInt'], "30")
    
    def test_network_config_validation(self):
        """Test network configuration validation."""
        network_config = NetworkConfig()
        
        # Test valid configuration
        network_config.port = 9876
        network_config.connect_timeout = 30
        network_config.max_connections = 100
        network_config.min_connections = 1
        
        errors = network_config.validate()
        self.assertEqual(len(errors), 0)
        
        # Test invalid port
        network_config.port = 0
        errors = network_config.validate()
        self.assertIn("Port must be between 1 and 65535", errors)
        
        # Test invalid timeout
        network_config.port = 9876
        network_config.connect_timeout = -1
        errors = network_config.validate()
        self.assertIn("Connect timeout must be positive", errors)
    
    def test_security_config_validation(self):
        """Test security configuration validation."""
        security_config = SecurityConfig()
        
        # Test SSL enabled without certificate
        security_config.ssl_enabled = True
        errors = security_config.validate()
        self.assertIn("SSL certificate file required when SSL is enabled", errors)
        self.assertIn("SSL key file required when SSL is enabled", errors)
        
        # Test authentication enabled without credentials
        security_config.ssl_enabled = False
        security_config.authentication_enabled = True
        errors = security_config.validate()
        self.assertIn("Username required when authentication is enabled", errors)
        self.assertIn("Password required when authentication is enabled", errors)
    
    def test_performance_config_validation(self):
        """Test performance configuration validation."""
        performance_config = PerformanceConfig()
        
        # Test valid configuration
        errors = performance_config.validate()
        self.assertEqual(len(errors), 0)
        
        # Test invalid queue size
        performance_config.message_queue_size = 0
        errors = performance_config.validate()
        self.assertIn("Message queue size must be positive", errors)
        
        # Test invalid thread count
        performance_config.message_queue_size = 1000
        performance_config.worker_threads = 0
        errors = performance_config.validate()
        self.assertIn("Worker threads must be positive", errors)
        
        # Test invalid threshold
        performance_config.worker_threads = 4
        performance_config.cpu_threshold_percent = 150
        errors = performance_config.validate()
        self.assertIn("CPU threshold must be between 0 and 100", errors)
    
    @patch('pyfixtest.core.fix_engine.fix.SessionSettings')
    def test_get_initiator_settings(self, mock_settings):
        """Test generating initiator settings."""
        # Setup
        mock_settings_instance = Mock()
        mock_settings.return_value = mock_settings_instance
        
        config = FIXConfig()
        config.session.sender_comp_id = "TEST_SENDER"
        config.session.target_comp_id = "TEST_TARGET"
        config.session.socket_connect_host = "localhost"
        config.session.socket_connect_port = 9876
        
        # Test
        result = config.get_initiator_settings()
        
        # Assertions
        self.assertEqual(result, mock_settings_instance)
        mock_settings.assert_called_once()
    
    @patch('pyfixtest.core.fix_engine.fix.SessionSettings')
    def test_get_acceptor_settings(self, mock_settings):
        """Test generating acceptor settings."""
        # Setup
        mock_settings_instance = Mock()
        mock_settings.return_value = mock_settings_instance
        
        config = FIXConfig()
        config.session.sender_comp_id = "TEST_SENDER"
        config.session.target_comp_id = "TEST_TARGET"
        config.session.socket_accept_port = 9876
        
        # Test
        result = config.get_acceptor_settings()
        
        # Assertions
        self.assertEqual(result, mock_settings_instance)
        mock_settings.assert_called_once()


class TestEngineStatistics(unittest.TestCase):
    """Test FIX engine statistics and monitoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = FIXConfig()
        self.config.store_path = os.path.join(self.temp_dir, "store")
        self.config.log_path = os.path.join(self.temp_dir, "logs")
        
        os.makedirs(self.config.store_path, exist_ok=True)
        os.makedirs(self.config.log_path, exist_ok=True)
        
        self.engine = FIXEngine(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'engine') and self.engine:
            try:
                self.engine.stop()
            except:
                pass
        
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_message_handler_management(self):
        """Test message handler registration and management."""
        handler1 = Mock()
        handler2 = Mock()
        
        # Test adding handlers
        self.engine.add_message_handler('D', handler1)
        self.engine.add_message_handler('D', handler2)
        self.engine.add_message_handler('8', handler1)
        
        # Verify handlers added
        self.assertEqual(len(self.engine.message_handlers['D']), 2)
        self.assertEqual(len(self.engine.message_handlers['8']), 1)
        
        # Test removing handler
        self.engine.remove_message_handler('D', handler1)
        self.assertEqual(len(self.engine.message_handlers['D']), 1)
        self.assertNotIn(handler1, self.engine.message_handlers['D'])
        
        # Test removing non-existent handler
        self.engine.remove_message_handler('D', handler1)  # Should not error
        self.assertEqual(len(self.engine.message_handlers['D']), 1)
    
    def test_received_messages_management(self):
        """Test received messages storage and retrieval."""
        # Initially empty
        self.assertEqual(len(self.engine.received_messages), 0)
        
        # Add mock messages
        mock_message1 = Mock()
        mock_message2 = Mock()
        
        self.engine.received_messages.append(mock_message1)
        self.engine.received_messages.append(mock_message2)
        
        self.assertEqual(len(self.engine.received_messages), 2)
        self.assertIn(mock_message1, self.engine.received_messages)
        self.assertIn(mock_message2, self.engine.received_messages)
    
    def test_thread_safety(self):
        """Test thread safety of message operations."""
        import threading
        import time
        
        messages_added = []
        handlers_added = []
        
        def add_messages():
            for i in range(10):
                mock_message = Mock()
                mock_message.id = f"msg_{i}"
                with self.engine._lock:
                    self.engine.received_messages.append(mock_message)
                    messages_added.append(mock_message)
                time.sleep(0.001)
        
        def add_handlers():
            for i in range(5):
                handler = Mock()
                handler.id = f"handler_{i}"
                self.engine.add_message_handler(f"TYPE_{i}", handler)
                handlers_added.append(handler)
                time.sleep(0.001)
        
        # Run operations concurrently
        thread1 = threading.Thread(target=add_messages)
        thread2 = threading.Thread(target=add_handlers)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Verify results
        self.assertEqual(len(self.engine.received_messages), 10)
        self.assertEqual(len(handlers_added), 5)
        
        # Verify all handlers were added
        for i, handler in enumerate(handlers_added):
            self.assertIn(handler, self.engine.message_handlers[f"TYPE_{i}"])


class TestEngineIntegration(unittest.TestCase):
    """Integration tests for FIX engine components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = FIXConfig()
        self.config.session.sender_comp_id = "TEST_SENDER"
        self.config.session.target_comp_id = "TEST_TARGET"
        self.config.store_path = os.path.join(self.temp_dir, "store")
        self.config.log_path = os.path.join(self.temp_dir, "logs")
        
        os.makedirs(self.config.store_path, exist_ok=True)
        os.makedirs(self.config.log_path, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_engine_configuration_integration(self):
        """Test engine with different configurations."""
        # Test with minimal configuration
        minimal_config = FIXConfig()
        engine = FIXEngine(minimal_config)
        self.assertIsNotNone(engine)
        
        # Test with SSL configuration
        ssl_config = FIXConfig()
        ssl_config.security.ssl_enabled = True
        ssl_config.security.ssl_cert_file = "/path/to/cert.pem"
        ssl_config.security.ssl_key_file = "/path/to/key.pem"
        ssl_config.store_path = self.config.store_path
        ssl_config.log_path = self.config.log_path
        
        ssl_engine = FIXEngine(ssl_config)
        self.assertIsNotNone(ssl_engine)
        self.assertTrue(ssl_engine.config.security.ssl_enabled)
        
        # Test with authentication configuration
        auth_config = FIXConfig()
        auth_config.security.authentication_enabled = True
        auth_config.security.username = "testuser"
        auth_config.security.password = "testpass"
        auth_config.store_path = self.config.store_path
        auth_config.log_path = self.config.log_path
        
        auth_engine = FIXEngine(auth_config)
        self.assertIsNotNone(auth_engine)
        self.assertTrue(auth_engine.config.security.authentication_enabled)
    
    @patch('pyfixtest.core.fix_engine.fix.SocketInitiator')
    @patch('pyfixtest.core.fix_engine.fix.FileLogFactory')
    @patch('pyfixtest.core.fix_engine.fix.FileStoreFactory')
    def test_full_initiator_lifecycle(self, mock_store_factory, mock_log_factory, mock_initiator):
        """Test complete initiator lifecycle."""
        # Setup mocks
        mock_initiator_instance = Mock()
        mock_initiator.return_value = mock_initiator_instance
        
        engine = FIXEngine(self.config)
        
        # Test startup
        result = engine.start_initiator()
        self.assertTrue(result)
        self.assertEqual(engine.state, SessionState.CONNECTING)
        
        # Simulate login
        engine.state = SessionState.LOGGED_IN
        self.assertTrue(engine.is_logged_in())
        
        # Test message sending
        mock_message = Mock()
        mock_session_id = Mock()
        engine.session_id = mock_session_id
        
        with patch('pyfixtest.core.fix_engine.fix.Session.sendToTarget') as mock_send:
            result = engine.send_message(mock_message)
            self.assertTrue(result)
            mock_send.assert_called_once_with(mock_message, mock_session_id)
        
        # Test shutdown
        engine.stop()
        self.assertEqual(engine.state, SessionState.DISCONNECTED)
        mock_initiator_instance.stop.assert_called_once()


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.getLogger().setLevel(logging.ERROR)  # Reduce noise during testing
    
    # Run tests
    unittest.main(verbosity=2)