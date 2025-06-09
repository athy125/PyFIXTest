"""
FIX protocol message handlers and processing logic.
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, timezone
from enum import Enum
import quickfix as fix

from ..utils.logging_config import get_logger
from ..utils.time_utils import get_utc_timestamp, parse_fix_time
from ..utils.message_utils import MessageUtils
from ..validators.validators import MessageValidator, OrderValidator
from .session_manager import SessionManager, SessionEventType


class MessageDirection(Enum):
    """Message direction for tracking."""
    INBOUND = "INBOUND"
    OUTBOUND = "OUTBOUND"


class MessagePriority(Enum):
    """Message processing priority."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ProcessingResult(Enum):
    """Message processing result."""
    SUCCESS = "SUCCESS"
    REJECTED = "REJECTED"
    ERROR = "ERROR"
    PENDING = "PENDING"


class MessageContext:
    """Context information for message processing."""
    
    def __init__(
        self,
        message: fix.Message,
        session_id: fix.SessionID,
        direction: MessageDirection,
        timestamp: Optional[datetime] = None
    ):
        self.message = message
        self.session_id = session_id
        self.direction = direction
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.priority = MessagePriority.NORMAL
        self.metadata: Dict[str, Any] = {}
        self.processing_time: Optional[float] = None
        self.result: Optional[ProcessingResult] = None
        self.error_message: Optional[str] = None
        
        # Extract message type
        try:
            self.msg_type = message.getHeader().getField(35)
        except:
            self.msg_type = "UNKNOWN"
    
    def set_priority(self, priority: MessagePriority):
        """Set message priority."""
        self.priority = priority
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to context."""
        self.metadata[key] = value
    
    def set_result(self, result: ProcessingResult, error_msg: Optional[str] = None):
        """Set processing result."""
        self.result = result
        self.error_message = error_msg
    
    def start_processing(self):
        """Mark start of processing."""
        self.processing_start_time = time.time()
    
    def end_processing(self):
        """Mark end of processing."""
        if hasattr(self, 'processing_start_time'):
            self.processing_time = time.time() - self.processing_start_time


class ProtocolHandler:
    """
    Main FIX protocol handler for processing messages.
    
    Provides:
    - Message routing and dispatch
    - Protocol-level validation
    - Session management integration
    - Error handling and recovery
    - Performance monitoring
    """
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.logger = get_logger(__name__)
        self.message_utils = MessageUtils()
        self.message_validator = MessageValidator()
        self.order_validator = OrderValidator()
        
        # Message handlers by type
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.default_handlers: List[Callable] = []
        
        # Processing queues
        self.processing_queue: List[MessageContext] = []
        self.processing_thread: Optional[threading.Thread] = None
        self.processing_active = False
        self._queue_lock = threading.Lock()
        
        # Statistics
        self.message_stats: Dict[str, int] = {}
        self.processing_times: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        
        # Configuration
        self.max_queue_size = 10000
        self.processing_timeout = 30.0
        self.validation_enabled = True
        
        # Pre-register standard handlers
        self._register_standard_handlers()
    
    def start_processing(self):
        """Start message processing thread."""
        if not self.processing_active:
            self.processing_active = True
            self.processing_thread = threading.Thread(
                target=self._process_messages,
                daemon=True
            )
            self.processing_thread.start()
            self.logger.info("Protocol handler started")
    
    def stop_processing(self):
        """Stop message processing."""
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            self.logger.info("Protocol handler stopped")
    
    def handle_inbound_message(
        self,
        message: fix.Message,
        session_id: fix.SessionID
    ) -> bool:
        """
        Handle inbound FIX message.
        
        Args:
            message: Received FIX message
            session_id: Session that received the message
            
        Returns:
            bool: True if handled successfully
        """
        try:
            context = MessageContext(message, session_id, MessageDirection.INBOUND)
            
            # Immediate validation for critical messages
            if self.validation_enabled:
                if not self._validate_message_basic(context):
                    return False
            
            # Record with session manager
            self.session_manager.record_message_received(session_id, message)
            
            # Queue for processing
            return self._queue_message(context)
            
        except Exception as e:
            self.logger.error(f"Error handling inbound message: {e}")
            return False
    
    def handle_outbound_message(
        self,
        message: fix.Message,
        session_id: fix.SessionID
    ) -> bool:
        """
        Handle outbound FIX message.
        
        Args:
            message: Message to send
            session_id: Target session
            
        Returns:
            bool: True if handled successfully
        """
        try:
            context = MessageContext(message, session_id, MessageDirection.OUTBOUND)
            
            # Pre-send validation
            if self.validation_enabled:
                if not self._validate_message_basic(context):
                    return False
            
            # Add sending time
            self.message_utils.add_sending_time(message)
            
            # Record with session manager
            self.session_manager.record_message_sent(session_id, message)
            
            # Send message
            fix.Session.sendToTarget(message, session_id)
            
            # Queue for post-send processing
            return self._queue_message(context)
            
        except Exception as e:
            self.logger.error(f"Error handling outbound message: {e}")
            return False
    
    def register_message_handler(self, msg_type: str, handler: Callable):
        """
        Register handler for specific message type.
        
        Args:
            msg_type: FIX message type (e.g., 'D', '8', 'V')
            handler: Handler function that takes (MessageContext) -> bool
        """
        if msg_type not in self.message_handlers:
            self.message_handlers[msg_type] = []
        self.message_handlers[msg_type].append(handler)
        self.logger.debug(f"Registered handler for message type {msg_type}")
    
    def unregister_message_handler(self, msg_type: str, handler: Callable):
        """Unregister message handler."""
        if msg_type in self.message_handlers:
            try:
                self.message_handlers[msg_type].remove(handler)
            except ValueError:
                pass
    
    def register_default_handler(self, handler: Callable):
        """Register default handler for unhandled messages."""
        self.default_handlers.append(handler)
    
    def send_heartbeat(self, session_id: fix.SessionID, test_req_id: Optional[str] = None):
        """Send heartbeat message."""
        try:
            heartbeat = fix.Message()
            header = heartbeat.getHeader()
            header.setField(35, '0')  # MsgType = Heartbeat
            
            if test_req_id:
                heartbeat.setField(112, test_req_id)  # TestReqID
            
            self.session_manager.record_heartbeat_sent(session_id)
            return self.handle_outbound_message(heartbeat, session_id)
            
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {e}")
            return False
    
    def send_test_request(self, session_id: fix.SessionID, test_req_id: str):
        """Send test request message."""
        try:
            test_request = fix.Message()
            header = test_request.getHeader()
            header.setField(35, '1')  # MsgType = TestRequest
            test_request.setField(112, test_req_id)  # TestReqID
            
            return self.handle_outbound_message(test_request, session_id)
            
        except Exception as e:
            self.logger.error(f"Error sending test request: {e}")
            return False
    
    def send_reject(
        self,
        session_id: fix.SessionID,
        ref_seq_num: int,
        reason: str,
        ref_msg_type: Optional[str] = None
    ):
        """Send reject message."""
        try:
            reject = fix.Message()
            header = reject.getHeader()
            header.setField(35, '3')  # MsgType = Reject
            reject.setField(45, str(ref_seq_num))  # RefSeqNum
            reject.setField(58, reason)  # Text
            
            if ref_msg_type:
                reject.setField(372, ref_msg_type)  # RefMsgType
            
            return self.handle_outbound_message(reject, session_id)
            
        except Exception as e:
            self.logger.error(f"Error sending reject: {e}")
            return False
    
    def send_business_reject(
        self,
        session_id: fix.SessionID,
        ref_msg_type: str,
        business_reject_reason: int,
        text: str,
        ref_id: Optional[str] = None
    ):
        """Send business message reject."""
        try:
            reject = fix.Message()
            header = reject.getHeader()
            header.setField(35, 'j')  # MsgType = BusinessMessageReject
            reject.setField(372, ref_msg_type)  # RefMsgType
            reject.setField(380, str(business_reject_reason))  # BusinessRejectReason
            reject.setField(58, text)  # Text
            
            if ref_id:
                reject.setField(379, ref_id)  # BusinessRejectRefID
            
            return self.handle_outbound_message(reject, session_id)
            
        except Exception as e:
            self.logger.error(f"Error sending business reject: {e}")
            return False
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get message processing statistics."""
        with self._queue_lock:
            stats = {
                'queue_size': len(self.processing_queue),
                'message_counts': dict(self.message_stats),
                'error_counts': dict(self.error_counts),
                'average_processing_times': {},
                'total_processed': sum(self.message_stats.values()),
                'total_errors': sum(self.error_counts.values()),
            }
            
            # Calculate average processing times
            for msg_type, times in self.processing_times.items():
                if times:
                    stats['average_processing_times'][msg_type] = {
                        'average_ms': sum(times) / len(times) * 1000,
                        'min_ms': min(times) * 1000,
                        'max_ms': max(times) * 1000,
                        'count': len(times)
                    }
            
            return stats
    
    def reset_statistics(self):
        """Reset processing statistics."""
        with self._queue_lock:
            self.message_stats.clear()
            self.processing_times.clear()
            self.error_counts.clear()
            self.logger.info("Protocol handler statistics reset")
    
    def _queue_message(self, context: MessageContext) -> bool:
        """Queue message for processing."""
        with self._queue_lock:
            if len(self.processing_queue) >= self.max_queue_size:
                self.logger.warning("Processing queue full, dropping message")
                return False
            
            # Set priority based on message type
            self._set_message_priority(context)
            
            # Insert based on priority
            self._insert_by_priority(context)
            
            return True
    
    def _set_message_priority(self, context: MessageContext):
        """Set message priority based on type and content."""
        critical_types = ['0', '1', '2', '4', '5']  # Heartbeat, TestRequest, ResendRequest, etc.
        high_priority_types = ['D', 'F', 'G', '8']  # Orders and executions
        
        if context.msg_type in critical_types:
            context.set_priority(MessagePriority.CRITICAL)
        elif context.msg_type in high_priority_types:
            context.set_priority(MessagePriority.HIGH)
        else:
            context.set_priority(MessagePriority.NORMAL)
    
    def _insert_by_priority(self, context: MessageContext):
        """Insert message in queue based on priority."""
        inserted = False
        for i, existing in enumerate(self.processing_queue):
            if context.priority.value > existing.priority.value:
                self.processing_queue.insert(i, context)
                inserted = True
                break
        
        if not inserted:
            self.processing_queue.append(context)
    
    def _process_messages(self):
        """Main message processing loop."""
        while self.processing_active:
            try:
                context = None
                
                # Get next message from queue
                with self._queue_lock:
                    if self.processing_queue:
                        context = self.processing_queue.pop(0)
                
                if context:
                    self._process_single_message(context)
                else:
                    time.sleep(0.01)  # Brief sleep when queue is empty
                    
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {e}")
                time.sleep(0.1)
    
    def _process_single_message(self, context: MessageContext):
        """Process a single message."""
        try:
            context.start_processing()
            
            # Update statistics
            self._update_message_stats(context.msg_type)
            
            # Validate message if enabled
            if self.validation_enabled and context.direction == MessageDirection.INBOUND:
                if not self._validate_message_full(context):
                    context.set_result(ProcessingResult.REJECTED, "Validation failed")
                    return
            
            # Route to appropriate handlers
            success = self._route_message(context)
            
            if success:
                context.set_result(ProcessingResult.SUCCESS)
            else:
                context.set_result(ProcessingResult.ERROR, "Handler processing failed")
                self._update_error_stats(context.msg_type)
            
        except Exception as e:
            self.logger.error(f"Error processing message {context.msg_type}: {e}")
            context.set_result(ProcessingResult.ERROR, str(e))
            self._update_error_stats(context.msg_type)
        
        finally:
            context.end_processing()
            if context.processing_time:
                self._update_processing_time(context.msg_type, context.processing_time)
    
    def _route_message(self, context: MessageContext) -> bool:
        """Route message to appropriate handlers."""
        handlers = self.message_handlers.get(context.msg_type, [])
        
        if not handlers:
            handlers = self.default_handlers
        
        success = True
        for handler in handlers:
            try:
                if not handler(context):
                    success = False
            except Exception as e:
                self.logger.error(f"Handler error for {context.msg_type}: {e}")
                success = False
        
        return success
    
    def _validate_message_basic(self, context: MessageContext) -> bool:
        """Perform basic message validation."""
        try:
            # Validate message structure
            self.message_validator.validate_message_structure(context.message)
            
            # Check for required fields based on message type
            self.message_validator.validate_required_fields(context.message, context.msg_type)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Basic validation failed for {context.msg_type}: {e}")
            self.session_manager.add_session_error(context.session_id, f"Validation error: {e}")
            return False
    
    def _validate_message_full(self, context: MessageContext) -> bool:
        """Perform full message validation."""
        try:
            # Basic validation
            if not self._validate_message_basic(context):
                return False
            
            # Order-specific validation
            if context.msg_type in ['D', 'F', 'G', '8']:
                self.order_validator.validate_execution_report(context.message)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Full validation failed for {context.msg_type}: {e}")
            return False
    
    def _register_standard_handlers(self):
        """Register handlers for standard FIX messages."""
        # Heartbeat handler
        self.register_message_handler('0', self._handle_heartbeat)
        
        # Test Request handler
        self.register_message_handler('1', self._handle_test_request)
        
        # Resend Request handler
        self.register_message_handler('2', self._handle_resend_request)
        
        # Reject handler
        self.register_message_handler('3', self._handle_reject)
        
        # Sequence Reset handler
        self.register_message_handler('4', self._handle_sequence_reset)
        
        # Logout handler
        self.register_message_handler('5', self._handle_logout)
        
        # Execution Report handler
        self.register_message_handler('8', self._handle_execution_report)
        
        # Business Message Reject handler
        self.register_message_handler('j', self._handle_business_reject)
    
    def _handle_heartbeat(self, context: MessageContext) -> bool:
        """Handle heartbeat message."""
        try:
            if context.direction == MessageDirection.INBOUND:
                # Check if this is a response to our test request
                try:
                    test_req_id = context.message.getField(112)
                    self.logger.debug(f"Received heartbeat response for TestReqID: {test_req_id}")
                except:
                    self.logger.debug("Received heartbeat")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling heartbeat: {e}")
            return False
    
    def _handle_test_request(self, context: MessageContext) -> bool:
        """Handle test request message."""
        try:
            if context.direction == MessageDirection.INBOUND:
                # Send heartbeat response
                test_req_id = context.message.getField(112)
                self.send_heartbeat(context.session_id, test_req_id)
                self.logger.debug(f"Responded to test request: {test_req_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling test request: {e}")
            return False
    
    def _handle_resend_request(self, context: MessageContext) -> bool:
        """Handle resend request message."""
        try:
            if context.direction == MessageDirection.INBOUND:
                begin_seq_no = int(context.message.getField(7))
                end_seq_no = int(context.message.getField(16))
                
                self.logger.info(f"Resend request: {begin_seq_no} to {end_seq_no}")
                
                # Handle resend logic here
                # This would typically involve replaying messages from the store
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling resend request: {e}")
            return False
    
    def _handle_reject(self, context: MessageContext) -> bool:
        """Handle reject message."""
        try:
            if context.direction == MessageDirection.INBOUND:
                ref_seq_num = context.message.getField(45)
                text = context.message.getField(58)
                
                self.logger.warning(f"Received reject for seq {ref_seq_num}: {text}")
                self.session_manager.add_session_error(
                    context.session_id,
                    f"Message rejected - seq: {ref_seq_num}, reason: {text}"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling reject: {e}")
            return False
    
    def _handle_sequence_reset(self, context: MessageContext) -> bool:
        """Handle sequence reset message."""
        try:
            if context.direction == MessageDirection.INBOUND:
                new_seq_no = int(context.message.getField(36))
                gap_fill_flag = context.message.getField(123) if context.message.isSetField(123) else 'N'
                
                self.logger.info(f"Sequence reset to {new_seq_no}, gap fill: {gap_fill_flag}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling sequence reset: {e}")
            return False
    
    def _handle_logout(self, context: MessageContext) -> bool:
        """Handle logout message."""
        try:
            if context.direction == MessageDirection.INBOUND:
                text = ""
                try:
                    text = context.message.getField(58)
                except:
                    pass
                
                self.logger.info(f"Received logout: {text}")
                self.session_manager.update_session_state(
                    context.session_id, 
                    self.session_manager.SessionState.LOGGED_OUT
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling logout: {e}")
            return False
    
    def _handle_execution_report(self, context: MessageContext) -> bool:
        """Handle execution report message."""
        try:
            if context.direction == MessageDirection.INBOUND:
                cl_ord_id = context.message.getField(11)
                exec_type = context.message.getField(150)
                ord_status = context.message.getField(39)
                
                self.logger.info(f"Execution report - Order: {cl_ord_id}, "
                               f"ExecType: {exec_type}, Status: {ord_status}")
                
                # Validate execution report
                self.order_validator.validate_execution_report(context.message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling execution report: {e}")
            return False
    
    def _handle_business_reject(self, context: MessageContext) -> bool:
        """Handle business message reject."""
        try:
            if context.direction == MessageDirection.INBOUND:
                ref_msg_type = context.message.getField(372)
                business_reject_reason = context.message.getField(380)
                text = context.message.getField(58)
                
                self.logger.warning(f"Business reject - MsgType: {ref_msg_type}, "
                                  f"Reason: {business_reject_reason}, Text: {text}")
                
                self.session_manager.add_session_error(
                    context.session_id,
                    f"Business reject - {ref_msg_type}: {text}"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling business reject: {e}")
            return False
    
    def _update_message_stats(self, msg_type: str):
        """Update message statistics."""
        if msg_type not in self.message_stats:
            self.message_stats[msg_type] = 0
        self.message_stats[msg_type] += 1
    
    def _update_error_stats(self, msg_type: str):
        """Update error statistics."""
        if msg_type not in self.error_counts:
            self.error_counts[msg_type] = 0
        self.error_counts[msg_type] += 1
    
    def _update_processing_time(self, msg_type: str, processing_time: float):
        """Update processing time statistics."""
        if msg_type not in self.processing_times:
            self.processing_times[msg_type] = []
        
        times = self.processing_times[msg_type]
        times.append(processing_time)
        
        # Keep only last 100 times for each message type
        if len(times) > 100:
            times.pop(0)


class MessageRouter:
    """
    Advanced message routing with filtering and transformation capabilities.
    """
    
    def __init__(self, protocol_handler: ProtocolHandler):
        self.protocol_handler = protocol_handler
        self.logger = get_logger(__name__)
        
        # Routing rules
        self.routing_rules: List[Dict[str, Any]] = []
        self.filters: Dict[str, Callable] = {}
        self.transformers: Dict[str, Callable] = {}
    
    def add_routing_rule(
        self,
        condition: Callable[[MessageContext], bool],
        handler: Callable[[MessageContext], bool],
        priority: int = 0
    ):
        """Add message routing rule."""
        rule = {
            'condition': condition,
            'handler': handler,
            'priority': priority
        }
        
        self.routing_rules.append(rule)
        self.routing_rules.sort(key=lambda x: x['priority'], reverse=True)
    
    def add_message_filter(self, name: str, filter_func: Callable[[MessageContext], bool]):
        """Add message filter."""
        self.filters[name] = filter_func
    
    def add_message_transformer(self, name: str, transform_func: Callable[[MessageContext], MessageContext]):
        """Add message transformer."""
        self.transformers[name] = transform_func
    
    def route_message(self, context: MessageContext) -> bool:
        """Route message through rules and filters."""
        try:
            # Apply filters first
            for filter_name, filter_func in self.filters.items():
                if not filter_func(context):
                    self.logger.debug(f"Message filtered out by {filter_name}")
                    return False
            
            # Apply transformers
            for transformer_name, transform_func in self.transformers.items():
                try:
                    context = transform_func(context)
                except Exception as e:
                    self.logger.error(f"Error in transformer {transformer_name}: {e}")
                    return False
            
            # Apply routing rules
            for rule in self.routing_rules:
                try:
                    if rule['condition'](context):
                        return rule['handler'](context)
                except Exception as e:
                    self.logger.error(f"Error in routing rule: {e}")
                    continue
            
            # No rule matched, use default handler
            return True
            
        except Exception as e:
            self.logger.error(f"Error routing message: {e}")
            return False