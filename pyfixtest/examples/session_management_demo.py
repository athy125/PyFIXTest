#!/usr/bin/env python3
"""
Session management demonstration.
"""

import unittest
import time
from pyfixtest import BaseFIXTest, SessionTestHelper


class SessionManagementTest(BaseFIXTest):
    """Example of session management testing."""
    
    def test_session_lifecycle(self):
        """Test complete session lifecycle."""
        print("Testing session lifecycle...")
        
        # Test engine startup
        success = self.engine.start_initiator()
        self.assertTrue(success, "Failed to start FIX engine")
        print("FIX engine started")
        
        # Test login process
        login_success = self.engine.wait_for_login(timeout=30.0)
        self.assertTrue(login_success, "Failed to log in")
        self.assertSessionLoggedIn(self.engine)
        print("Session logged in successfully")
        
        # Test heartbeat mechanism
        session_helper = SessionTestHelper(self.engine, self.message_factory)
        heartbeat_ok = session_helper.test_heartbeat_mechanism()
        self.assertTrue(heartbeat_ok, "Heartbeat test failed")
        print("Heartbeat mechanism working correctly")
        
        # Test sending messages during active session
        test_message = self.message_factory.create_test_request("TEST_123")
        send_success = self.engine.send_message(test_message)
        self.assertTrue(send_success, "Failed to send test request")
        
        # Wait for heartbeat response
        heartbeat = self.engine.wait_for_message('0', timeout=10.0)
        if heartbeat:
            self.assertHeartbeatValid(heartbeat, expected_test_req_id="TEST_123")
            print("Received heartbeat response to test request")
        
        print("Session lifecycle test completed!")
    
    def test_session_states(self):
        """Test different session states."""
        print("Testing session states...")
        
        # Initial state should be disconnected
        initial_state = self.engine.get_session_state()
        print(f"Initial state: {initial_state}")
        
        # Start engine and check connecting state
        self.engine.start_initiator()
        time.sleep(1)  # Allow state transition
        
        connecting_state = self.engine.get_session_state()
        print(f"After start: {connecting_state}")
        
        # Wait for login and check logged in state
        if self.engine.wait_for_login(timeout=30.0):
            logged_in_state = self.engine.get_session_state()
            print(f"After login: {logged_in_state}")
            self.assertTrue(self.engine.is_logged_in())
        
        print("Session state testing completed!")


if __name__ == '__main__':
    unittest.main(verbosity=2)