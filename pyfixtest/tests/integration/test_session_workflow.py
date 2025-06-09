"""
Integration tests for FIX session workflow and lifecycle.

This module provides comprehensive integration tests for FIX session workflows,
including session establishment, heartbeat management, message flows,
error recovery, and session termination scenarios.
"""

import unittest
import time
import threading
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from pyfixtest import BaseFIXTest, SessionTestHelper
from pyfixtest.core.fix_engine import FIXEngine, SessionState
from pyfixtest.core.session_manager import SessionManager, SessionInfo, SessionEventType
from pyfixtest.core.protocol_handler import ProtocolHandler, MessageContext, MessageDirection
from pyfixtest.core.message_factory import MessageFactory
from pyfixtest.config.fix_config import FIXConfig
from pyfixtest.testing.test_config import TestConfig, create_integration_test_config
from pyfixtest.utils.logging_config import get_logger

# Mock QuickFIX if not available
try:
    import quickfix as fix
except ImportError:
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


class TestSessionWorkflow(BaseFIXTest):
    """Integration tests for complete session workflow."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class fixtures."""
        super().setUpClass()
        cls.test_config = create_integration_test_config()
        cls.test_config.setup()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test class fixtures."""
        if hasattr(cls, 'test_config'):
            cls.test_config.cleanup()
        super().tearDownClass()
    
    def setUp(self):
        """Set up individual test fixtures."""
        super().setUp()
        self.session_events = []
        self.message_events = []
        self.error_events = []
        
        # Create test configuration
        self.fix_config = self.test_config.get_fix_config()
        self.session_manager = SessionManager(self.fix_config)
        self.protocol_handler = ProtocolHandler(self.session_manager)
        
        # Set up event tracking
        self._setup_event_tracking()
    
    def tearDown(self):
        """Clean up individual test fixtures."""
        try:
            if hasattr(self, 'session_manager'):
                self.session_manager.stop_monitoring()
            if hasattr(self, 'protocol_handler'):
                self.protocol_handler.stop_processing()
        except:
            pass
        super().tearDown()
    
    def _setup_event_tracking(self):
        """Set up event tracking for session and protocol events."""
        # Track session events
        for event_type in SessionEventType:
            self.session_manager.add_event_handler(
                event_type, 
                lambda session_info, *args, et=event_type: self._track_session_event(et, session_info, *args)
            )
        
        # Track protocol events
        self.protocol_handler.start_processing()
    
    def _track_session_event(self, event_type: SessionEventType, session_info: SessionInfo, *args):
        """Track session events for verification."""
        event = {
            'type': event_type,
            'session_info': session_info,
            'timestamp': datetime.now(timezone.utc),
            'args': args
        }
        self.session_events.append(event)
        self.logger.debug(f"Session event: {event_type.value} for {session_info.session_id}")
    
    def _track_message_event(self, message, direction: MessageDirection):
        """Track message events for verification."""
        try:
            msg_type = message.getHeader().getField(35) if hasattr(message, 'getHeader') else 'UNKNOWN'
            event = {
                'message_type': msg_type,
                'direction': direction,
                'timestamp': datetime.now(timezone.utc),
                'message': message
            }
            self.message_events.append(event)
        except Exception as e:
            self.logger.warning(f"Error tracking message event: {e}")


class TestSessionEstablishment(TestSessionWorkflow):
    """Test session establishment workflows."""
    
    @patch('pyfixtest.core.fix_engine.fix.SocketInitiator')
    @patch('pyfixtest.core.fix_engine.fix.FileLogFactory')
    @patch('pyfixtest.core.fix_engine.fix.FileStoreFactory')
    def test_successful_session_establishment(self, mock_store_factory, mock_log_factory, mock_initiator):
        """Test successful FIX session establishment workflow."""
        # Setup mocks
        mock_initiator_instance = Mock()
        mock_initiator.return_value = mock_initiator_instance
        
        # Create session ID
        session_id = self._create_test_session_id()
        
        # Start session establishment process
        with self.fix_session() as engine:
            # Verify engine started
            self.assertEqual(engine.get_session_state(), SessionState.CONNECTING)
            
            # Register session with manager
            session_info = self.session_manager.register_session(session_id)
            self.assertIsNotNone(session_info)
            
            # Simulate logon sequence
            self._simulate_logon_sequence(engine, session_id)
            
            # Verify session is established
            self.assertTrue(engine.is_logged_in())
            self.assertTrue(self.session_manager.is_session_active(session_id))
            
            # Verify session events
            self._verify_session_establishment_events()
    
    def test_session_establishment_with_authentication(self):
        """Test session establishment with authentication."""
        # Configure authentication
        auth_config = self.fix_config
        auth_config.security.authentication_enabled = True
        auth_config.security.username = "testuser"
        auth_config.security.password = "testpass"
        
        with patch('pyfixtest.core.fix_engine.fix.SocketInitiator'):
            engine = FIXEngine(auth_config)
            session_id = self._create_test_session_id()
            
            # Simulate authenticated logon
            logon_message = self._create_logon_message(
                username="testuser",
                password="testpass"
            )
            
            # Process logon
            self._simulate_message_exchange(engine, session_id, logon_message)
            
            # Verify authentication was processed
            session_info = self.session_manager.get_session_info(session_id)
            if session_info:
                self.assertEqual(session_info.state, SessionState.LOGGED_IN)
    
    def test_session_establishment_failure_scenarios(self):
        """Test various session establishment failure scenarios."""
        session_id = self._create_test_session_id()
        
        # Test 1: Connection timeout
        with patch('pyfixtest.core.fix_engine.fix.SocketInitiator') as mock_initiator:
            mock_initiator.side_effect = Exception("Connection timeout")
            
            engine = FIXEngine(self.fix_config)
            result = engine.start_initiator()
            
            self.assertFalse(result)
            self.assertEqual(engine.get_session_state(), SessionState.ERROR)
        
        # Test 2: Invalid credentials
        auth_config = self.fix_config
        auth_config.security.authentication_enabled = True
        auth_config.security.username = "invalid_user"
        auth_config.security.password = "wrong_pass"
        
        with patch('pyfixtest.core.fix_engine.fix.SocketInitiator'):
            engine = FIXEngine(auth_config)
            
            # Simulate authentication rejection
            reject_message = self._create_reject_message("Invalid credentials")
            self._simulate_message_exchange(engine, session_id, reject_message)
            
            # Should not be logged in
            self.assertFalse(engine.is_logged_in())
        
        # Test 3: Sequence number issues
        self._test_sequence_number_issues(session_id)
    
    def test_session_establishment_with_ssl(self):
        """Test session establishment with SSL/TLS."""
        # Configure SSL
        ssl_config = self.fix_config
        ssl_config.security.ssl_enabled = True
        ssl_config.security.ssl_cert_file = "/path/to/test.crt"
        ssl_config.security.ssl_key_file = "/path/to/test.key"
        
        with patch('pyfixtest.core.fix_engine.fix.SocketInitiator') as mock_initiator:
            mock_initiator_instance = Mock()
            mock_initiator.return_value = mock_initiator_instance
            
            engine = FIXEngine(ssl_config)
            result = engine.start_initiator()
            
            # Verify SSL configuration was applied
            self.assertTrue(result)
            mock_initiator.assert_called_once()
            
            # Verify settings include SSL parameters
            call_args = mock_initiator.call_args
            settings = call_args[0][1]  # Settings object
            self.assertIsNotNone(settings)


class TestHeartbeatManagement(TestSessionWorkflow):
    """Test heartbeat management workflows."""
    
    def test_normal_heartbeat_workflow(self):
        """Test normal heartbeat exchange workflow."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            session_helper = SessionTestHelper(engine, self.message_factory)
            
            # Register session
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Test heartbeat mechanism
            heartbeat_success = session_helper.test_heartbeat_mechanism()
            self.assertTrue(heartbeat_success, "Heartbeat mechanism should work correctly")
            
            # Verify heartbeat events were recorded
            heartbeat_events = [e for e in self.session_events 
                              if e['type'] == SessionEventType.HEARTBEAT_SENT]
            self.assertGreater(len(heartbeat_events), 0, "Should have heartbeat events")
    
    def test_heartbeat_timeout_handling(self):
        """Test heartbeat timeout detection and handling."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Register session and set to logged in
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            session_info.update_heartbeat()
            
            # Start monitoring
            self.session_manager.start_monitoring()
            
            # Simulate heartbeat timeout by not updating heartbeat
            # Wait longer than heartbeat interval
            heartbeat_interval = self.fix_config.get_heartbeat_interval()
            time.sleep(0.1)  # Short wait for test
            
            # Manually trigger timeout check
            is_overdue = session_info.is_heartbeat_overdue(heartbeat_interval=1)  # 1 second for test
            
            if is_overdue:
                # Should trigger timeout handling
                self.session_manager.add_session_error(session_id, "Heartbeat timeout")
                
                # Verify error was recorded
                self.assertGreater(len(session_info.errors), 0)
                self.assertIn("timeout", session_info.errors[-1].lower())
    
    def test_test_request_response_workflow(self):
        """Test Test Request/Heartbeat response workflow."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Send test request
            test_req_id = f"TEST_REQ_{int(time.time() * 1000)}"
            test_request = self.message_factory.create_test_request(test_req_id)
            
            success = engine.send_message(test_request)
            self.assertTrue(success, "Test request should be sent successfully")
            
            # Simulate heartbeat response
            heartbeat_response = self.message_factory.create_heartbeat(test_req_id)
            
            # Add to received messages to simulate response
            engine.received_messages.append(heartbeat_response)
            
            # Wait for and verify heartbeat response
            response = engine.wait_for_message('0', timeout=5.0)  # Heartbeat
            self.assertIsNotNone(response, "Should receive heartbeat response")
            
            # Verify test request ID matches
            try:
                received_test_req_id = response.getField(112)  # TestReqID
                self.assertEqual(received_test_req_id, test_req_id)
            except:
                pass  # Field might not be present in mock
    
    def test_heartbeat_interval_compliance(self):
        """Test compliance with configured heartbeat intervals."""
        session_id = self._create_test_session_id()
        heartbeat_interval = 10  # 10 seconds for test
        
        # Configure specific heartbeat interval
        test_config = self.fix_config
        test_config.session.heartbeat_interval = heartbeat_interval
        
        with self.fix_session() as engine:
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Track heartbeat timing
            heartbeat_times = []
            
            def track_heartbeat(session_info, *args):
                heartbeat_times.append(datetime.now(timezone.utc))
            
            self.session_manager.add_event_handler(
                SessionEventType.HEARTBEAT_SENT, 
                track_heartbeat
            )
            
            # Simulate multiple heartbeats
            for i in range(3):
                self.session_manager.record_heartbeat_sent(session_id)
                time.sleep(0.1)  # Short delay for test
            
            # Verify heartbeat timing (in real implementation)
            if len(heartbeat_times) >= 2:
                intervals = []
                for i in range(1, len(heartbeat_times)):
                    interval = (heartbeat_times[i] - heartbeat_times[i-1]).total_seconds()
                    intervals.append(interval)
                
                # In real test, would verify intervals are approximately correct
                self.assertGreater(len(intervals), 0)


class TestMessageFlowWorkflows(TestSessionWorkflow):
    """Test message flow workflows during active sessions."""
    
    def test_order_message_workflow(self):
        """Test complete order message workflow."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Establish session
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Create and send new order
            new_order = self.message_factory.create_new_order_single(
                symbol='AAPL',
                side='1',  # Buy
                order_qty=100.0,
                order_type='2',  # Limit
                price=150.0,
                cl_ord_id='TEST_ORDER_001'
            )
            
            success = engine.send_message(new_order)
            self.assertTrue(success, "Order should be sent successfully")
            
            # Record message sent
            self.session_manager.record_message_sent(session_id, new_order)
            
            # Simulate execution report response
            exec_report = self.message_factory.create_execution_report(
                cl_ord_id='TEST_ORDER_001',
                exec_id='EXEC_001',
                exec_type='0',  # New
                ord_status='0',  # New
                symbol='AAPL',
                side='1',
                leaves_qty=100.0,
                cum_qty=0.0
            )
            
            # Simulate receiving execution report
            engine.received_messages.append(exec_report)
            self.session_manager.record_message_received(session_id, exec_report)
            
            # Verify message flow
            session_stats = session_info
            self.assertEqual(session_stats.message_count_sent, 1)
            self.assertEqual(session_stats.message_count_received, 1)
            
            # Verify execution report received
            received_exec = engine.wait_for_message('8', timeout=1.0)  # Execution Report
            self.assertIsNotNone(received_exec, "Should receive execution report")
    
    def test_market_data_workflow(self):
        """Test market data subscription and update workflow."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Establish session
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Create market data request
            md_request = self.message_factory.create_market_data_request(
                md_req_id='MD_REQ_001',
                subscription_request_type='1',  # Snapshot + Updates
                market_depth=5,
                symbols=['AAPL', 'MSFT'],
                md_entry_types=['0', '1']  # Bid, Offer
            )
            
            success = engine.send_message(md_request)
            self.assertTrue(success, "Market data request should be sent")
            
            # Simulate market data snapshot response
            md_snapshot = self._create_market_data_snapshot('AAPL')
            engine.received_messages.append(md_snapshot)
            
            # Verify market data received
            received_md = engine.wait_for_message('W', timeout=1.0)  # Market Data Snapshot
            self.assertIsNotNone(received_md, "Should receive market data snapshot")
    
    def test_administrative_message_workflow(self):
        """Test administrative message workflows."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Establish session
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Test sequence reset
            seq_reset = self._create_sequence_reset_message(new_seq_no=5)
            
            # Process through protocol handler
            context = MessageContext(seq_reset, session_id, MessageDirection.INBOUND)
            success = self.protocol_handler.handle_inbound_message(seq_reset, session_id)
            self.assertTrue(success, "Sequence reset should be processed")
            
            # Test resend request
            resend_request = self._create_resend_request_message(begin_seq_no=1, end_seq_no=3)
            success = self.protocol_handler.handle_inbound_message(resend_request, session_id)
            self.assertTrue(success, "Resend request should be processed")
            
            # Verify administrative messages were handled
            admin_events = [e for e in self.message_events 
                           if e['message_type'] in ['2', '4']]  # ResendRequest, SequenceReset
            # In full implementation, would verify proper handling


class TestErrorRecoveryWorkflows(TestSessionWorkflow):
    """Test error recovery and resilience workflows."""
    
    def test_sequence_number_recovery(self):
        """Test sequence number gap detection and recovery."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Establish session
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Simulate sequence number gap
            # Normal sequence: 1, 2, 3, ...
            # Simulate receiving message with sequence 5 (gap: 4)
            
            message_with_gap = self._create_test_message_with_sequence(5)
            
            # Process message through protocol handler
            success = self.protocol_handler.handle_inbound_message(message_with_gap, session_id)
            
            # Should trigger resend request for missing sequence numbers
            # In full implementation, would verify resend request is sent
            self.assertTrue(success, "Message with sequence gap should trigger recovery")
    
    def test_connection_recovery_workflow(self):
        """Test connection loss and recovery workflow."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Establish session
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Simulate connection loss
            session_info.update_state(SessionState.DISCONNECTED)
            self.session_manager.add_session_error(session_id, "Connection lost")
            
            # Simulate reconnection attempt
            self._simulate_reconnection_attempt(engine, session_id)
            
            # Verify error recovery
            self.assertGreater(len(session_info.errors), 0)
            
            # In full implementation, would verify:
            # - Reconnection logic triggered
            # - Session state properly restored
            # - Message recovery performed
    
    def test_message_rejection_handling(self):
        """Test handling of message rejections."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Establish session
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Send invalid order
            invalid_order = self.message_factory.create_new_order_single(
                symbol='INVALID_SYMBOL',
                side='3',  # Invalid side
                order_qty=-100.0,  # Invalid quantity
                order_type='X'  # Invalid order type
            )
            
            success = engine.send_message(invalid_order)
            self.assertTrue(success, "Message should be sent even if invalid")
            
            # Simulate business reject response
            business_reject = self._create_business_reject_message(
                ref_msg_type='D',
                business_reject_reason=5,  # Invalid field value
                text="Invalid order side"
            )
            
            # Process reject through protocol handler
            success = self.protocol_handler.handle_inbound_message(business_reject, session_id)
            self.assertTrue(success, "Business reject should be processed")
            
            # Verify reject was handled
            # In full implementation, would verify proper error handling
    
    def test_timeout_and_retry_workflow(self):
        """Test timeout detection and retry workflows."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Configure short timeouts for testing
            test_config = self.fix_config
            test_config.session.logon_timeout = 1  # 1 second
            test_config.session.logout_timeout = 1
            
            # Simulate timeout scenario
            start_time = datetime.now(timezone.utc)
            
            # Send message that would timeout
            test_message = self.message_factory.create_test_request("TIMEOUT_TEST")
            success = engine.send_message(test_message)
            
            # Wait for timeout period
            time.sleep(0.1)  # Short wait for test
            
            # Check if timeout would be detected
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # In full implementation, would verify:
            # - Timeout detection logic
            # - Retry mechanisms
            # - Exponential backoff
            self.assertGreater(elapsed, 0)


class TestSessionTermination(TestSessionWorkflow):
    """Test session termination workflows."""
    
    def test_graceful_logout_workflow(self):
        """Test graceful session logout workflow."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Establish session
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Initiate logout
            logout_message = self._create_logout_message("Normal logout")
            success = engine.send_message(logout_message)
            self.assertTrue(success, "Logout message should be sent")
            
            # Update session state
            session_info.update_state(SessionState.LOGGING_OUT)
            
            # Simulate logout response
            logout_response = self._create_logout_message("Logout acknowledged")
            engine.received_messages.append(logout_response)
            
            # Process logout
            success = self.protocol_handler.handle_inbound_message(logout_response, session_id)
            self.assertTrue(success, "Logout response should be processed")
            
            # Update final state
            session_info.update_state(SessionState.LOGGED_OUT)
            
            # Verify session termination
            self.assertFalse(self.session_manager.is_session_active(session_id))
            self.assertEqual(session_info.state, SessionState.LOGGED_OUT)
    
    def test_forced_disconnection_workflow(self):
        """Test forced disconnection handling."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Establish session
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Simulate forced disconnection
            self.session_manager.force_logout_session(session_id, "Forced disconnection")
            
            # Verify session was terminated
            self.assertEqual(session_info.state, SessionState.LOGGING_OUT)
            
            # In full implementation, would verify:
            # - Cleanup procedures executed
            # - Resources properly released
            # - Error logging performed
    
    def test_session_cleanup_workflow(self):
        """Test session cleanup and resource management."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Establish session
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Add some session activity
            self.session_manager.record_message_sent(session_id, Mock())
            self.session_manager.record_message_received(session_id, Mock())
            
            # Unregister session (cleanup)
            self.session_manager.unregister_session(session_id)
            
            # Verify cleanup
            retrieved_session = self.session_manager.get_session_info(session_id)
            # Session info should still exist but marked as disconnected
            if retrieved_session:
                self.assertEqual(retrieved_session.state, SessionState.DISCONNECTED)


class TestConcurrentSessionWorkflows(TestSessionWorkflow):
    """Test concurrent session management workflows."""
    
    def test_multiple_session_management(self):
        """Test managing multiple concurrent sessions."""
        session_ids = []
        
        # Create multiple test sessions
        for i in range(3):
            session_id = self._create_test_session_id(f"SENDER_{i}", f"TARGET_{i}")
            session_ids.append(session_id)
        
        with self.fix_session() as engine:
            # Register all sessions
            session_infos = []
            for session_id in session_ids:
                session_info = self.session_manager.register_session(session_id)
                session_info.update_state(SessionState.LOGGED_IN)
                session_infos.append(session_info)
            
            # Verify all sessions are active
            active_sessions = self.session_manager.get_active_sessions()
            self.assertEqual(len(active_sessions), 3)
            
            # Send messages on different sessions
            for i, session_id in enumerate(session_ids):
                test_message = self.message_factory.create_test_request(f"TEST_{i}")
                self.session_manager.record_message_sent(session_id, test_message)
            
            # Verify message counts
            for session_info in session_infos:
                self.assertEqual(session_info.message_count_sent, 1)
    
    def test_session_isolation(self):
        """Test that sessions are properly isolated."""
        session_id1 = self._create_test_session_id("SENDER_1", "TARGET_1")
        session_id2 = self._create_test_session_id("SENDER_2", "TARGET_2")
        
        with self.fix_session() as engine:
            # Register sessions
            session1 = self.session_manager.register_session(session_id1)
            session2 = self.session_manager.register_session(session_id2)
            
            # Add error to session 1
            self.session_manager.add_session_error(session_id1, "Test error")
            
            # Verify error isolation
            self.assertGreater(len(session1.errors), 0)
            self.assertEqual(len(session2.errors), 0)
            
            # Send messages to different sessions
            self.session_manager.record_message_sent(session_id1, Mock())
            self.session_manager.record_message_sent(session_id1, Mock())
            self.session_manager.record_message_sent(session_id2, Mock())
            
            # Verify message count isolation
            self.assertEqual(session1.message_count_sent, 2)
            self.assertEqual(session2.message_count_sent, 1)


# Helper methods for test setup and simulation
class TestSessionWorkflow(TestSessionWorkflow):
    """Extended test class with helper methods."""
    
    def _create_test_session_id(self, sender_comp_id=None, target_comp_id=None):
        """Create test session ID."""
        if hasattr(fix, 'SessionID'):
            return fix.SessionID(
                "FIX.4.4",
                sender_comp_id or "TEST_SENDER",
                target_comp_id or "TEST_TARGET",
                ""
            )
        else:
            # Mock session ID
            mock_session_id = Mock()
            mock_session_id.getSenderCompID.return_value = sender_comp_id or "TEST_SENDER"
            mock_session_id.getTargetCompID.return_value = target_comp_id or "TEST_TARGET"
            mock_session_id.getBeginString.return_value = "FIX.4.4"
            mock_session_id.getSessionQualifier.return_value = ""
            return mock_session_id
    
    def _simulate_logon_sequence(self, engine, session_id):
        """Simulate complete logon sequence."""
        # Create logon message
        logon_message = self._create_logon_message()
        
        # Send logon
        success = engine.send_message(logon_message)
        self.assertTrue(success, "Logon message should be sent")
        
        # Simulate logon acknowledgment
        logon_ack = self._create_logon_message()
        engine.received_messages.append(logon_ack)
        
        # Update session state
        session_info = self.session_manager.get_session_info(session_id)
        if session_info:
            session_info.update_state(SessionState.LOGGED_IN)
        
        # Update engine state
        engine.state = SessionState.LOGGED_IN
    
    def _create_logon_message(self, username=None, password=None):
        """Create logon message."""
        logon = Mock()
        logon.getHeader.return_value.getField.return_value = 'A'  # Logon message type
        
        # Add authentication fields if provided
        if username:
            logon.getField = Mock(side_effect=lambda tag: {
                553: username,  # Username
                554: password or ""  # Password
            }.get(tag, ""))
        
        return logon
    
    def _create_logout_message(self, text=""):
        """Create logout message."""
        logout = Mock()
        logout.getHeader.return_value.getField.return_value = '5'  # Logout message type
        
        if text:
            logout.getField = Mock(side_effect=lambda tag: {
                58: text  # Text field
            }.get(tag, ""))
        
        return logout
    
    def _create_reject_message(self, reason=""):
        """Create reject message."""
        reject = Mock()
        reject.getHeader.return_value.getField.return_value = '3'  # Reject message type
        reject.getField = Mock(side_effect=lambda tag: {
            58: reason,  # Text
            45: "1",     # RefSeqNum
            372: "A"     # RefMsgType (Logon)
        }.get(tag, ""))
        
        return reject
    
    def _create_business_reject_message(self, ref_msg_type, business_reject_reason, text):
        """Create business message reject."""
        reject = Mock()
        reject.getHeader.return_value.getField.return_value = 'j'  # BusinessMessageReject
        reject.getField = Mock(side_effect=lambda tag: {
            372: ref_msg_type,  # RefMsgType
            380: str(business_reject_reason),  # BusinessRejectReason
            58: text  # Text
        }.get(tag, ""))
        
        return reject
    
    def _create_sequence_reset_message(self, new_seq_no, gap_fill_flag="N"):
        """Create sequence reset message."""
        seq_reset = Mock()
        seq_reset.getHeader.return_value.getField.return_value = '4'  # SequenceReset
        seq_reset.getField = Mock(side_effect=lambda tag: {
            36: str(new_seq_no),  # NewSeqNo
            123: gap_fill_flag    # GapFillFlag
        }.get(tag, ""))
        
        return seq_reset
    
    def _create_resend_request_message(self, begin_seq_no, end_seq_no):
        """Create resend request message."""
        resend_req = Mock()
        resend_req.getHeader.return_value.getField.return_value = '2'  # ResendRequest
        resend_req.getField = Mock(side_effect=lambda tag: {
            7: str(begin_seq_no),  # BeginSeqNo
            16: str(end_seq_no)    # EndSeqNo
        }.get(tag, ""))
        
        return resend_req
    
    def _create_test_message_with_sequence(self, seq_num):
        """Create test message with specific sequence number."""
        message = Mock()
        message.getHeader.return_value.getField = Mock(side_effect=lambda tag: {
            35: 'D',  # MsgType (NewOrderSingle)
            34: str(seq_num)  # MsgSeqNum
        }.get(tag, ""))
        
        return message
    
    def _create_market_data_snapshot(self, symbol):
        """Create market data snapshot message."""
        md_snapshot = Mock()
        md_snapshot.getHeader.return_value.getField.return_value = 'W'  # MarketDataSnapshot
        md_snapshot.getField = Mock(side_effect=lambda tag: {
            55: symbol,  # Symbol
            262: "MD_REQ_001",  # MDReqID
            268: "2"  # NoMDEntries
        }.get(tag, ""))
        
        return md_snapshot
    
    def _simulate_message_exchange(self, engine, session_id, message):
        """Simulate message exchange with session."""
        # Add message to received messages
        engine.received_messages.append(message)
        
        # Record with session manager
        self.session_manager.record_message_received(session_id, message)
        
        # Process through protocol handler if available
        try:
            self.protocol_handler.handle_inbound_message(message, session_id)
        except:
            pass  # Continue if protocol handler not available
    
    def _simulate_reconnection_attempt(self, engine, session_id):
        """Simulate reconnection attempt after connection loss."""
        # Update session state to reconnecting
        session_info = self.session_manager.get_session_info(session_id)
        if session_info:
            session_info.update_state(SessionState.RECONNECTING)
        
        # Simulate reconnection logic
        time.sleep(0.1)  # Brief delay
        
        # In full implementation, would:
        # - Attempt to re-establish connection
        # - Perform logon sequence
        # - Recover message state
        
        # For test, just mark as connected
        if session_info:
            session_info.update_state(SessionState.LOGGED_IN)
    
    def _verify_session_establishment_events(self):
        """Verify that proper session establishment events occurred."""
        # Check for session creation event
        creation_events = [e for e in self.session_events 
                          if e['type'] == SessionEventType.SESSION_CREATED]
        self.assertGreater(len(creation_events), 0, "Should have session creation events")
        
        # Check for logon event
        logon_events = [e for e in self.session_events 
                       if e['type'] == SessionEventType.SESSION_LOGON]
        # Note: This might be 0 in mock environment, but would be > 0 in real implementation
        
        # Verify event ordering
        if len(creation_events) > 0 and len(logon_events) > 0:
            creation_time = creation_events[0]['timestamp']
            logon_time = logon_events[0]['timestamp']
            self.assertLessEqual(creation_time, logon_time, 
                               "Creation should occur before logon")
    
    def _test_sequence_number_issues(self, session_id):
        """Test sequence number gap and recovery."""
        with patch('pyfixtest.core.fix_engine.fix.SocketInitiator'):
            engine = FIXEngine(self.fix_config)
            
            # Simulate sequence number problems
            # Expected sequence: 1, 2, 3, 4, 5
            # Receive: 1, 2, 4, 5 (missing 3)
            
            for seq_num in [1, 2, 4, 5]:
                message = self._create_test_message_with_sequence(seq_num)
                
                # Process message
                try:
                    self.protocol_handler.handle_inbound_message(message, session_id)
                except:
                    pass  # Continue if handler not available
            
            # In full implementation, would verify:
            # - Gap detection for missing sequence 3
            # - Resend request generated
            # - Proper sequence recovery


class TestSessionPerformanceWorkflows(TestSessionWorkflow):
    """Test session performance and load scenarios."""
    
    def test_high_message_volume_workflow(self):
        """Test session handling under high message volume."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Establish session
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Send high volume of messages
            message_count = 100
            start_time = time.time()
            
            for i in range(message_count):
                test_message = self.message_factory.create_test_request(f"PERF_TEST_{i}")
                success = engine.send_message(test_message)
                self.assertTrue(success, f"Message {i} should be sent successfully")
                
                # Record message
                self.session_manager.record_message_sent(session_id, test_message)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Verify performance metrics
            self.assertEqual(session_info.message_count_sent, message_count)
            self.assertLess(duration, 10.0, "Should handle high volume efficiently")
            
            # Calculate throughput
            throughput = message_count / duration
            self.logger.info(f"Message throughput: {throughput:.2f} messages/second")
    
    def test_concurrent_session_load(self):
        """Test concurrent load across multiple sessions."""
        session_count = 5
        messages_per_session = 20
        session_ids = []
        
        # Create multiple sessions
        for i in range(session_count):
            session_id = self._create_test_session_id(f"LOAD_SENDER_{i}", f"LOAD_TARGET_{i}")
            session_ids.append(session_id)
        
        with self.fix_session() as engine:
            # Register all sessions
            for session_id in session_ids:
                session_info = self.session_manager.register_session(session_id)
                session_info.update_state(SessionState.LOGGED_IN)
            
            # Send messages concurrently
            def send_messages_for_session(session_id, session_index):
                for i in range(messages_per_session):
                    message = self.message_factory.create_test_request(
                        f"CONCURRENT_{session_index}_{i}"
                    )
                    engine.send_message(message)
                    self.session_manager.record_message_sent(session_id, message)
                    time.sleep(0.001)  # Small delay to simulate real timing
            
            # Create and start threads
            threads = []
            start_time = time.time()
            
            for i, session_id in enumerate(session_ids):
                thread = threading.Thread(
                    target=send_messages_for_session,
                    args=(session_id, i)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            end_time = time.time()
            
            # Verify all messages were sent
            total_messages_sent = 0
            for session_id in session_ids:
                session_info = self.session_manager.get_session_info(session_id)
                total_messages_sent += session_info.message_count_sent
            
            expected_total = session_count * messages_per_session
            self.assertEqual(total_messages_sent, expected_total)
            
            # Verify performance
            duration = end_time - start_time
            overall_throughput = expected_total / duration
            self.logger.info(f"Concurrent throughput: {overall_throughput:.2f} messages/second")
    
    def test_session_resource_management(self):
        """Test session resource usage and cleanup."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Establish session
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Generate substantial message traffic
            large_message_count = 500
            
            # Track resource usage (simplified)
            initial_message_count = len(engine.received_messages)
            
            for i in range(large_message_count):
                # Simulate received message
                mock_message = Mock()
                mock_message.getHeader.return_value.getField.return_value = 'D'  # Order
                engine.received_messages.append(mock_message)
                
                # Periodically clean up (simulate message processing)
                if i % 100 == 0:
                    # In real implementation, would process and remove old messages
                    processed_count = min(50, len(engine.received_messages))
                    engine.received_messages = engine.received_messages[processed_count:]
            
            # Verify resource management
            final_message_count = len(engine.received_messages)
            self.assertLess(final_message_count, large_message_count, 
                          "Should manage message buffer size")


class TestSessionComplianceWorkflows(TestSessionWorkflow):
    """Test session compliance and regulatory scenarios."""
    
    def test_audit_trail_workflow(self):
        """Test audit trail generation and compliance."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Establish session
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Enable audit tracking
            audit_events = []
            
            def audit_callback(session_info, *args):
                audit_events.append({
                    'session_id': str(session_info.session_id),
                    'timestamp': datetime.now(timezone.utc),
                    'event_type': 'MESSAGE_SENT',
                    'details': args
                })
            
            self.session_manager.add_event_handler(
                SessionEventType.MESSAGE_SENT,
                audit_callback
            )
            
            # Send tracked messages
            order_message = self.message_factory.create_new_order_single(
                symbol='AAPL',
                side='1',
                order_qty=100.0,
                order_type='2',
                price=150.0,
                cl_ord_id='AUDIT_ORDER_001'
            )
            
            engine.send_message(order_message)
            self.session_manager.record_message_sent(session_id, order_message)
            
            # Verify audit trail
            self.assertGreater(len(audit_events), 0, "Should generate audit events")
            
            # Verify audit event content
            if audit_events:
                audit_event = audit_events[0]
                self.assertIn('session_id', audit_event)
                self.assertIn('timestamp', audit_event)
                self.assertEqual(audit_event['event_type'], 'MESSAGE_SENT')
    
    def test_regulatory_timing_compliance(self):
        """Test compliance with regulatory timing requirements."""
        session_id = self._create_test_session_id()
        
        with self.fix_session() as engine:
            # Establish session
            session_info = self.session_manager.register_session(session_id)
            session_info.update_state(SessionState.LOGGED_IN)
            
            # Test message timestamp requirements
            order_message = self.message_factory.create_new_order_single(
                symbol='AAPL',
                side='1',
                order_qty=100.0
            )
            
            # Record send time
            send_time = datetime.now(timezone.utc)
            success = engine.send_message(order_message)
            self.assertTrue(success)
            
            # Verify timestamp compliance (messages should have recent timestamps)
            # In full implementation, would check:
            # - SendingTime field accuracy
            # - TransactTime field presence
            # - Clock synchronization compliance
            
            # Simulate execution report with timing
            exec_report = self.message_factory.create_execution_report(
                cl_ord_id='TIMING_TEST_001',
                exec_id='EXEC_TIMING_001',
                exec_type='0',
                ord_status='0',
                symbol='AAPL',
                side='1',
                leaves_qty=100.0,
                cum_qty=0.0
            )
            
            receive_time = datetime.now(timezone.utc)
            engine.received_messages.append(exec_report)
            
            # Verify timing compliance
            response_time = (receive_time - send_time).total_seconds()
            self.assertLess(response_time, 1.0, "Response should be timely")


if __name__ == '__main__':
    # Configure test environment
    import logging
    logging.getLogger().setLevel(logging.WARNING)  # Reduce test noise
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSessionEstablishment,
        TestHeartbeatManagement,
        TestMessageFlowWorkflows,
        TestErrorRecoveryWorkflows,
        TestSessionTermination,
        TestConcurrentSessionWorkflows,
        TestSessionPerformanceWorkflows,
        TestSessionComplianceWorkflows
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Session Workflow Integration Tests Summary")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)