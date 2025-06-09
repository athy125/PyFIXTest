"""
FIX Engine wrapper providing high-level interface for FIX protocol operations.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import quickfix as fix

from ..utils.logging_config import get_logger
from ..config.fix_config import FIXConfig


class SessionState(Enum):
    """FIX session states."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    LOGGED_IN = "LOGGED_IN"
    LOGGED_OUT = "LOGGED_OUT"
    ERROR = "ERROR"


class FIXEngine:
    """
    High-level FIX engine wrapper for testing FIX-based trading systems.
    
    Provides simplified interface for:
    - Session management
    - Message sending/receiving
    - Event handling
    - Connection monitoring
    """
    
    def __init__(self, config: FIXConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.state = SessionState.DISCONNECTED
        self.session_id = None
        self.application = None
        self.initiator = None
        self.acceptor = None
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.received_messages: List[fix.Message] = []
        self._lock = threading.Lock()
        
    def start_initiator(self) -> bool:
        """
        Start FIX initiator (client) connection.
        
        Returns:
            bool: True if started successfully
        """
        try:
            self.application = FIXTestApplication(self)
            settings = self.config.get_initiator_settings()
            
            store_factory = fix.FileStoreFactory(settings)
            log_factory = fix.FileLogFactory(settings)
            
            self.initiator = fix.SocketInitiator(
                self.application, store_factory, settings, log_factory
            )
            
            self.initiator.start()
            self.state = SessionState.CONNECTING
            self.logger.info("FIX initiator started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start initiator: {e}")
            self.state = SessionState.ERROR
            return False
    
    def start_acceptor(self) -> bool:
        """
        Start FIX acceptor (server) connection.
        
        Returns:
            bool: True if started successfully
        """
        try:
            self.application = FIXTestApplication(self)
            settings = self.config.get_acceptor_settings()
            
            store_factory = fix.FileStoreFactory(settings)
            log_factory = fix.FileLogFactory(settings)
            
            self.acceptor = fix.SocketAcceptor(
                self.application, store_factory, settings, log_factory
            )
            
            self.acceptor.start()
            self.state = SessionState.CONNECTING
            self.logger.info("FIX acceptor started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start acceptor: {e}")
            self.state = SessionState.ERROR
            return False
    
    def stop(self):
        """Stop FIX engine and cleanup resources."""
        try:
            if self.initiator:
                self.initiator.stop()
                self.initiator = None
            
            if self.acceptor:
                self.acceptor.stop()
                self.acceptor = None
                
            self.state = SessionState.DISCONNECTED
            self.logger.info("FIX engine stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping FIX engine: {e}")
    
    def send_message(self, message: fix.Message, session_id: Optional[str] = None) -> bool:
        """
        Send FIX message to counterparty.
        
        Args:
            message: FIX message to send
            session_id: Optional session ID, uses default if None
            
        Returns:
            bool: True if sent successfully
        """
        try:
            target_session = session_id or self.session_id
            if target_session:
                fix.Session.sendToTarget(message, target_session)
                self.logger.debug(f"Sent message: {message}")
                return True
            else:
                self.logger.error("No active session to send message")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    def wait_for_message(self, msg_type: str, timeout: float = 10.0) -> Optional[fix.Message]:
        """
        Wait for specific message type to be received.
        
        Args:
            msg_type: FIX message type to wait for
            timeout: Timeout in seconds
            
        Returns:
            Received message or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                for msg in self.received_messages:
                    if msg.getHeader().getField(35) == msg_type:
                        self.received_messages.remove(msg)
                        return msg
            time.sleep(0.1)
        
        return None
    
    def add_message_handler(self, msg_type: str, handler: Callable):
        """Add handler for specific message type."""
        if msg_type not in self.message_handlers:
            self.message_handlers[msg_type] = []
        self.message_handlers[msg_type].append(handler)
    
    def remove_message_handler(self, msg_type: str, handler: Callable):
        """Remove message handler."""
        if msg_type in self.message_handlers:
            try:
                self.message_handlers[msg_type].remove(handler)
            except ValueError:
                pass
    
    def get_session_state(self) -> SessionState:
        """Get current session state."""
        return self.state
    
    def is_logged_in(self) -> bool:
        """Check if session is logged in."""
        return self.state == SessionState.LOGGED_IN
    
    def wait_for_login(self, timeout: float = 30.0) -> bool:
        """
        Wait for session to log in.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            bool: True if logged in within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_logged_in():
                return True
            time.sleep(0.5)
        
        return False


class FIXTestApplication(fix.Application):
    """FIX application for testing purposes."""
    
    def __init__(self, engine: FIXEngine):
        super().__init__()
        self.engine = engine
        self.logger = get_logger(__name__)
    
    def onCreate(self, sessionID):
        self.logger.info(f"Session created: {sessionID}")
        self.engine.session_id = sessionID
    
    def onLogon(self, sessionID):
        self.logger.info(f"Session logged on: {sessionID}")
        self.engine.state = SessionState.LOGGED_IN
    
    def onLogout(self, sessionID):
        self.logger.info(f"Session logged out: {sessionID}")
        self.engine.state = SessionState.LOGGED_OUT
    
    def toAdmin(self, message, sessionID):
        self.logger.debug(f"Admin message to send: {message}")
    
    def fromAdmin(self, message, sessionID):
        self.logger.debug(f"Admin message received: {message}")
    
    def toApp(self, message, sessionID):
        self.logger.debug(f"Application message to send: {message}")
    
    def fromApp(self, message, sessionID):
        self.logger.debug(f"Application message received: {message}")
        
        # Store received message
        with self.engine._lock:
            self.engine.received_messages.append(message)
        
        # Call registered handlers
        try:
            msg_type = message.getHeader().getField(35)
            if msg_type in self.engine.message_handlers:
                for handler in self.engine.message_handlers[msg_type]:
                    try:
                        handler(message, sessionID)
                    except Exception as e:
                        self.logger.error(f"Error in message handler: {e}")
        except Exception as e:
            self.logger.error(f"Error processing received message: {e}")