"""
FIX session management and lifecycle handling.
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from datetime import datetime, timezone
import quickfix as fix

from ..utils.logging_config import get_logger
from ..utils.time_utils import get_utc_timestamp, is_timestamp_recent
from ..config.fix_config import FIXConfig


class SessionState(Enum):
    """FIX session states."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    LOGGING_IN = "LOGGING_IN"
    LOGGED_IN = "LOGGED_IN"
    LOGGING_OUT = "LOGGING_OUT"
    LOGGED_OUT = "LOGGED_OUT"
    ERROR = "ERROR"
    RECONNECTING = "RECONNECTING"


class SessionEventType(Enum):
    """Session event types."""
    SESSION_CREATED = "SESSION_CREATED"
    SESSION_LOGON = "SESSION_LOGON"
    SESSION_LOGOUT = "SESSION_LOGOUT"
    SESSION_REJECTED = "SESSION_REJECTED"
    MESSAGE_SENT = "MESSAGE_SENT"
    MESSAGE_RECEIVED = "MESSAGE_RECEIVED"
    HEARTBEAT_SENT = "HEARTBEAT_SENT"
    HEARTBEAT_RECEIVED = "HEARTBEAT_RECEIVED"
    ERROR_OCCURRED = "ERROR_OCCURRED"


class SessionInfo:
    """Information about a FIX session."""
    
    def __init__(self, session_id: fix.SessionID):
        self.session_id = session_id
        self.state = SessionState.DISCONNECTED
        self.created_time = datetime.now(timezone.utc)
        self.login_time: Optional[datetime] = None
        self.logout_time: Optional[datetime] = None
        self.last_heartbeat_time: Optional[datetime] = None
        self.message_count_sent = 0
        self.message_count_received = 0
        self.sequence_number_sent = 1
        self.sequence_number_received = 1
        self.errors: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    def update_state(self, new_state: SessionState):
        """Update session state with timestamp."""
        old_state = self.state
        self.state = new_state
        
        if new_state == SessionState.LOGGED_IN:
            self.login_time = datetime.now(timezone.utc)
        elif new_state == SessionState.LOGGED_OUT:
            self.logout_time = datetime.now(timezone.utc)
    
    def add_error(self, error_msg: str):
        """Add error to session history."""
        self.errors.append(f"{get_utc_timestamp()}: {error_msg}")
    
    def increment_sent_count(self):
        """Increment sent message count."""
        self.message_count_sent += 1
        self.sequence_number_sent += 1
    
    def increment_received_count(self):
        """Increment received message count."""
        self.message_count_received += 1
        self.sequence_number_received += 1
    
    def update_heartbeat(self):
        """Update last heartbeat timestamp."""
        self.last_heartbeat_time = datetime.now(timezone.utc)
    
    def get_session_duration(self) -> Optional[float]:
        """Get session duration in seconds."""
        if self.login_time:
            end_time = self.logout_time or datetime.now(timezone.utc)
            return (end_time - self.login_time).total_seconds()
        return None
    
    def is_heartbeat_overdue(self, heartbeat_interval: int = 30) -> bool:
        """Check if heartbeat is overdue."""
        if not self.last_heartbeat_time:
            return False
        
        elapsed = (datetime.now(timezone.utc) - self.last_heartbeat_time).total_seconds()
        return elapsed > heartbeat_interval * 2  # Allow 2x interval before considering overdue


class SessionManager:
    """
    Manages FIX sessions and their lifecycle.
    
    Provides:
    - Session creation and destruction
    - State management
    - Event handling
    - Statistics tracking
    - Health monitoring
    """
    
    def __init__(self, config: FIXConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Session tracking
        self.sessions: Dict[str, SessionInfo] = {}
        self.active_sessions: Dict[str, fix.SessionID] = {}
        
        # Event handling
        self.event_handlers: Dict[SessionEventType, List[Callable]] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        
        # Statistics
        self.total_sessions_created = 0
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.start_time = datetime.now(timezone.utc)
        
        # Configuration
        self.heartbeat_interval = config.get_heartbeat_interval()
        self.reconnect_interval = config.get_reconnect_interval()
        self.max_reconnect_attempts = config.get_max_reconnect_attempts()
    
    def start_monitoring(self):
        """Start session monitoring thread."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitor_sessions,
                daemon=True
            )
            self._monitoring_thread.start()
            self.logger.info("Session monitoring started")
    
    def stop_monitoring(self):
        """Stop session monitoring thread."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
            self.logger.info("Session monitoring stopped")
    
    def register_session(self, session_id: fix.SessionID) -> SessionInfo:
        """
        Register a new session.
        
        Args:
            session_id: FIX session ID
            
        Returns:
            SessionInfo: Created session info object
        """
        with self._lock:
            session_key = self._get_session_key(session_id)
            
            if session_key in self.sessions:
                self.logger.warning(f"Session {session_key} already registered")
                return self.sessions[session_key]
            
            session_info = SessionInfo(session_id)
            self.sessions[session_key] = session_info
            self.active_sessions[session_key] = session_id
            self.total_sessions_created += 1
            
            self.logger.info(f"Registered session: {session_key}")
            self._fire_event(SessionEventType.SESSION_CREATED, session_info)
            
            return session_info
    
    def unregister_session(self, session_id: fix.SessionID):
        """Unregister a session."""
        with self._lock:
            session_key = self._get_session_key(session_id)
            
            if session_key in self.sessions:
                session_info = self.sessions[session_key]
                session_info.update_state(SessionState.DISCONNECTED)
                
                del self.active_sessions[session_key]
                self.logger.info(f"Unregistered session: {session_key}")
    
    def get_session_info(self, session_id: fix.SessionID) -> Optional[SessionInfo]:
        """Get session information."""
        session_key = self._get_session_key(session_id)
        return self.sessions.get(session_key)
    
    def update_session_state(self, session_id: fix.SessionID, state: SessionState):
        """Update session state."""
        with self._lock:
            session_info = self.get_session_info(session_id)
            if session_info:
                old_state = session_info.state
                session_info.update_state(state)
                
                self.logger.debug(f"Session {self._get_session_key(session_id)} "
                                f"state changed: {old_state} -> {state}")
                
                # Fire appropriate events
                if state == SessionState.LOGGED_IN:
                    self._fire_event(SessionEventType.SESSION_LOGON, session_info)
                elif state == SessionState.LOGGED_OUT:
                    self._fire_event(SessionEventType.SESSION_LOGOUT, session_info)
                elif state == SessionState.ERROR:
                    self._fire_event(SessionEventType.ERROR_OCCURRED, session_info)
    
    def record_message_sent(self, session_id: fix.SessionID, message: fix.Message):
        """Record sent message statistics."""
        with self._lock:
            session_info = self.get_session_info(session_id)
            if session_info:
                session_info.increment_sent_count()
                self.total_messages_sent += 1
                
                self._fire_event(SessionEventType.MESSAGE_SENT, session_info, message)
    
    def record_message_received(self, session_id: fix.SessionID, message: fix.Message):
        """Record received message statistics."""
        with self._lock:
            session_info = self.get_session_info(session_id)
            if session_info:
                session_info.increment_received_count()
                self.total_messages_received += 1
                
                # Check for heartbeat
                try:
                    msg_type = message.getHeader().getField(35)
                    if msg_type == '0':  # Heartbeat
                        session_info.update_heartbeat()
                        self._fire_event(SessionEventType.HEARTBEAT_RECEIVED, session_info)
                except:
                    pass
                
                self._fire_event(SessionEventType.MESSAGE_RECEIVED, session_info, message)
    
    def record_heartbeat_sent(self, session_id: fix.SessionID):
        """Record heartbeat sent."""
        session_info = self.get_session_info(session_id)
        if session_info:
            session_info.update_heartbeat()
            self._fire_event(SessionEventType.HEARTBEAT_SENT, session_info)
    
    def add_session_error(self, session_id: fix.SessionID, error_msg: str):
        """Add error to session."""
        session_info = self.get_session_info(session_id)
        if session_info:
            session_info.add_error(error_msg)
            session_info.update_state(SessionState.ERROR)
            self.logger.error(f"Session {self._get_session_key(session_id)} error: {error_msg}")
    
    def get_active_sessions(self) -> List[SessionInfo]:
        """Get list of active sessions."""
        with self._lock:
            return [
                self.sessions[key] 
                for key in self.active_sessions.keys()
                if key in self.sessions
            ]
    
    def get_session_by_target(self, target_comp_id: str) -> Optional[SessionInfo]:
        """Get session by target company ID."""
        for session_info in self.sessions.values():
            if session_info.session_id.getTargetCompID() == target_comp_id:
                return session_info
        return None
    
    def is_session_active(self, session_id: fix.SessionID) -> bool:
        """Check if session is active."""
        session_info = self.get_session_info(session_id)
        return (session_info and 
                session_info.state in [SessionState.LOGGED_IN, SessionState.CONNECTED])
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get overall session statistics."""
        with self._lock:
            active_count = len(self.active_sessions)
            logged_in_count = sum(
                1 for info in self.sessions.values()
                if info.state == SessionState.LOGGED_IN
            )
            
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            
            return {
                'total_sessions_created': self.total_sessions_created,
                'active_sessions': active_count,
                'logged_in_sessions': logged_in_count,
                'total_messages_sent': self.total_messages_sent,
                'total_messages_received': self.total_messages_received,
                'uptime_seconds': uptime,
                'messages_per_second_sent': self.total_messages_sent / max(uptime, 1),
                'messages_per_second_received': self.total_messages_received / max(uptime, 1),
            }
    
    def get_session_health_report(self) -> Dict[str, Any]:
        """Get session health report."""
        with self._lock:
            healthy_sessions = []
            unhealthy_sessions = []
            
            for session_info in self.sessions.values():
                session_data = {
                    'session_id': str(session_info.session_id),
                    'state': session_info.state.value,
                    'duration': session_info.get_session_duration(),
                    'messages_sent': session_info.message_count_sent,
                    'messages_received': session_info.message_count_received,
                    'errors': len(session_info.errors),
                    'last_heartbeat': session_info.last_heartbeat_time,
                }
                
                if (session_info.state == SessionState.LOGGED_IN and
                    not session_info.is_heartbeat_overdue(self.heartbeat_interval) and
                    len(session_info.errors) < 5):
                    healthy_sessions.append(session_data)
                else:
                    session_data['issues'] = []
                    
                    if session_info.state != SessionState.LOGGED_IN:
                        session_data['issues'].append(f"Not logged in: {session_info.state.value}")
                    
                    if session_info.is_heartbeat_overdue(self.heartbeat_interval):
                        session_data['issues'].append("Heartbeat overdue")
                    
                    if len(session_info.errors) >= 5:
                        session_data['issues'].append(f"High error count: {len(session_info.errors)}")
                    
                    unhealthy_sessions.append(session_data)
            
            return {
                'healthy_sessions': healthy_sessions,
                'unhealthy_sessions': unhealthy_sessions,
                'overall_health': 'HEALTHY' if not unhealthy_sessions else 'DEGRADED',
                'timestamp': get_utc_timestamp(),
            }
    
    def add_event_handler(self, event_type: SessionEventType, handler: Callable):
        """Add event handler for session events."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: SessionEventType, handler: Callable):
        """Remove event handler."""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    def reset_session_stats(self, session_id: fix.SessionID):
        """Reset statistics for a session."""
        session_info = self.get_session_info(session_id)
        if session_info:
            session_info.message_count_sent = 0
            session_info.message_count_received = 0
            session_info.errors.clear()
            self.logger.info(f"Reset stats for session {self._get_session_key(session_id)}")
    
    def force_logout_session(self, session_id: fix.SessionID, reason: str = "Forced logout"):
        """Force logout a session."""
        try:
            fix.Session.lookupSession(session_id).logout(reason)
            self.update_session_state(session_id, SessionState.LOGGING_OUT)
            self.logger.info(f"Forced logout session {self._get_session_key(session_id)}: {reason}")
        except Exception as e:
            self.logger.error(f"Failed to force logout session: {e}")
    
    def disconnect_all_sessions(self):
        """Disconnect all active sessions."""
        with self._lock:
            for session_id in list(self.active_sessions.values()):
                self.force_logout_session(session_id, "Manager shutdown")
    
    def _get_session_key(self, session_id: fix.SessionID) -> str:
        """Generate unique key for session."""
        return f"{session_id.getSenderCompID()}_{session_id.getTargetCompID()}_{session_id.getSessionQualifier()}"
    
    def _fire_event(self, event_type: SessionEventType, session_info: SessionInfo, *args):
        """Fire session event to registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(session_info, *args)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {e}")
    
    def _monitor_sessions(self):
        """Background monitoring of sessions."""
        while self._monitoring_active:
            try:
                with self._lock:
                    for session_info in self.sessions.values():
                        # Check for heartbeat timeouts
                        if session_info.is_heartbeat_overdue(self.heartbeat_interval):
                            self.logger.warning(
                                f"Heartbeat overdue for session {self._get_session_key(session_info.session_id)}"
                            )
                        
                        # Check for stale sessions
                        if (session_info.state == SessionState.CONNECTING and
                            (datetime.now(timezone.utc) - session_info.created_time).total_seconds() > 60):
                            session_info.update_state(SessionState.ERROR)
                            session_info.add_error("Connection timeout")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in session monitoring: {e}")
                time.sleep(5)


class SessionFactory:
    """Factory for creating FIX sessions."""
    
    def __init__(self, config: FIXConfig, session_manager: SessionManager):
        self.config = config
        self.session_manager = session_manager
        self.logger = get_logger(__name__)
    
    def create_session_id(
        self,
        sender_comp_id: str,
        target_comp_id: str,
        session_qualifier: str = ""
    ) -> fix.SessionID:
        """Create FIX session ID."""
        begin_string = self.config.get_begin_string()
        return fix.SessionID(begin_string, sender_comp_id, target_comp_id, session_qualifier)
    
    def create_session_settings(
        self,
        session_id: fix.SessionID,
        connection_type: str = "initiator",
        **kwargs
    ) -> fix.SessionSettings:
        """Create session settings."""
        settings = fix.SessionSettings()
        
        # Default settings
        default_dict = fix.Dictionary()
        default_dict.setString("BeginString", session_id.getBeginString())
        default_dict.setString("ConnectionType", connection_type)
        default_dict.setString("HeartBtInt", str(self.config.get_heartbeat_interval()))
        default_dict.setString("FileStorePath", self.config.get_store_path())
        default_dict.setString("FileLogPath", self.config.get_log_path())
        
        # Add custom settings
        for key, value in kwargs.items():
            default_dict.setString(key, str(value))
        
        settings.set(default_dict)
        
        # Session-specific settings
        session_dict = fix.Dictionary()
        session_dict.setString("SenderCompID", session_id.getSenderCompID())
        session_dict.setString("TargetCompID", session_id.getTargetCompID())
        
        if session_id.getSessionQualifier():
            session_dict.setString("SessionQualifier", session_id.getSessionQualifier())
        
        settings.set(session_id, session_dict)
        
        return settings
    
    def validate_session_config(self, session_id: fix.SessionID) -> bool:
        """Validate session configuration."""
        try:
            # Check required components
            if not session_id.getSenderCompID():
                raise ValueError("SenderCompID is required")
            
            if not session_id.getTargetCompID():
                raise ValueError("TargetCompID is required")
            
            if not session_id.getBeginString():
                raise ValueError("BeginString is required")
            
            # Check FIX version support
            supported_versions = ["FIX.4.2", "FIX.4.4", "FIXT.1.1"]
            if session_id.getBeginString() not in supported_versions:
                raise ValueError(f"Unsupported FIX version: {session_id.getBeginString()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Session validation failed: {e}")
            return False