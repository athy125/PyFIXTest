"""
Validators for FIX messages and trading workflows.
"""

import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import quickfix as fix

from ..utils.logging_config import get_logger


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class MessageValidator:
    """
    Validator for FIX message structure and content.
    
    Provides validation for:
    - Message format compliance
    - Required field presence
    - Field value constraints
    - Business logic rules
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.required_fields = self._load_required_fields()
    
    def validate_message_structure(self, message: fix.Message) -> bool:
        """
        Validate basic FIX message structure.
        
        Args:
            message: FIX message to validate
            
        Returns:
            bool: True if structure is valid
            
        Raises:
            ValidationError: If structure is invalid
        """
        try:
            # Check if message has header
            header = message.getHeader()
            if not header:
                raise ValidationError("Message missing header")
            
            # Check message type
            msg_type = header.getField(35)
            if not msg_type:
                raise ValidationError("Message missing MsgType field")
            
            # Check begin string
            begin_string = header.getField(8)
            if not begin_string:
                raise ValidationError("Message missing BeginString field")
            
            # Check body length
            body_length = header.getField(9)
            if not body_length:
                raise ValidationError("Message missing BodyLength field")
            
            return True
            
        except Exception as e:
            raise ValidationError(f"Message structure validation failed: {e}")
    
    def validate_required_fields(self, message: fix.Message, msg_type: str) -> bool:
        """
        Validate that all required fields are present.
        
        Args:
            message: FIX message to validate
            msg_type: Message type
            
        Returns:
            bool: True if all required fields present
            
        Raises:
            ValidationError: If required fields missing
        """
        if msg_type not in self.required_fields:
            self.logger.warning(f"No required field rules for message type {msg_type}")
            return True
        
        missing_fields = []
        required = self.required_fields[msg_type]
        
        for field_tag in required:
            try:
                message.getField(field_tag)
            except:
                missing_fields.append(field_tag)
        
        if missing_fields:
            raise ValidationError(f"Missing required fields: {missing_fields}")
        
        return True
    
    def validate_field_values(self, message: fix.Message, rules: Dict[int, Any]) -> bool:
        """
        Validate field values against custom rules.
        
        Args:
            message: FIX message to validate
            rules: Dictionary of field_tag -> validation_rule
            
        Returns:
            bool: True if all field values are valid
            
        Raises:
            ValidationError: If field values are invalid
        """
        for field_tag, rule in rules.items():
            try:
                field_value = message.getField(field_tag)
                
                if isinstance(rule, list):
                    # Value must be in allowed list
                    if field_value not in rule:
                        raise ValidationError(
                            f"Field {field_tag} value '{field_value}' not in allowed values: {rule}"
                        )
                
                elif isinstance(rule, dict):
                    # Complex validation rule
                    if 'min' in rule:
                        if float(field_value) < rule['min']:
                            raise ValidationError(
                                f"Field {field_tag} value {field_value} below minimum {rule['min']}"
                            )
                    
                    if 'max' in rule:
                        if float(field_value) > rule['max']:
                            raise ValidationError(
                                f"Field {field_tag} value {field_value} above maximum {rule['max']}"
                            )
                    
                    if 'pattern' in rule:
                        if not re.match(rule['pattern'], field_value):
                            raise ValidationError(
                                f"Field {field_tag} value '{field_value}' doesn't match pattern"
                            )
                
                elif callable(rule):
                    # Custom validation function
                    if not rule(field_value):
                        raise ValidationError(
                            f"Field {field_tag} value '{field_value}' failed custom validation"
                        )
            
            except fix.FieldNotFound:
                # Field not present, skip validation
                continue
        
        return True
    
    def validate_order_message(self, message: fix.Message) -> bool:
        """
        Validate order-specific business rules.
        
        Args:
            message: Order message to validate
            
        Returns:
            bool: True if order is valid
            
        Raises:
            ValidationError: If order validation fails
        """
        try:
            msg_type = message.getHeader().getField(35)
            
            if msg_type == 'D':  # New Order Single
                return self._validate_new_order_single(message)
            elif msg_type == 'F':  # Order Cancel Request
                return self._validate_order_cancel_request(message)
            elif msg_type == 'G':  # Order Cancel/Replace Request
                return self._validate_order_cancel_replace_request(message)
            else:
                self.logger.warning(f"No order validation rules for message type {msg_type}")
                return True
        
        except Exception as e:
            raise ValidationError(f"Order validation failed: {e}")
    
    def _validate_new_order_single(self, message: fix.Message) -> bool:
        """Validate New Order Single message."""
        # Check required fields
        required_fields = [11, 55, 54, 60, 40, 38]  # ClOrdID, Symbol, Side, TransactTime, OrdType, OrderQty
        
        for field_tag in required_fields:
            try:
                message.getField(field_tag)
            except:
                raise ValidationError(f"NewOrderSingle missing required field {field_tag}")
        
        # Validate order type and price
        ord_type = message.getField(40)
        if ord_type == '2':  # Limit order
            try:
                price = float(message.getField(44))
                if price <= 0:
                    raise ValidationError("Limit order price must be positive")
            except:
                raise ValidationError("Limit order missing price field")
        
        # Validate quantity
        try:
            qty = float(message.getField(38))
            if qty <= 0:
                raise ValidationError("Order quantity must be positive")
        except:
            raise ValidationError("Invalid order quantity")
        
        # Validate side
        side = message.getField(54)
        if side not in ['1', '2']:  # Buy, Sell
            raise ValidationError(f"Invalid order side: {side}")
        
        return True
    
    def _validate_order_cancel_request(self, message: fix.Message) -> bool:
        """Validate Order Cancel Request message."""
        required_fields = [11, 41, 55, 54]  # ClOrdID, OrigClOrdID, Symbol, Side
        
        for field_tag in required_fields:
            try:
                message.getField(field_tag)
            except:
                raise ValidationError(f"OrderCancelRequest missing required field {field_tag}")
        
        return True
    
    def _validate_order_cancel_replace_request(self, message: fix.Message) -> bool:
        """Validate Order Cancel/Replace Request message."""
        required_fields = [11, 41, 55, 54, 38]  # ClOrdID, OrigClOrdID, Symbol, Side, OrderQty
        
        for field_tag in required_fields:
            try:
                message.getField(field_tag)
            except:
                raise ValidationError(f"OrderCancelReplaceRequest missing required field {field_tag}")
        
        return True
    
    def _load_required_fields(self) -> Dict[str, List[int]]:
        """Load required fields mapping for different message types."""
        return {
            'D': [11, 55, 54, 60, 40, 38],  # New Order Single
            'F': [11, 41, 55, 54, 60],       # Order Cancel Request
            'G': [11, 41, 55, 54, 60, 38],   # Order Cancel/Replace Request
            '8': [11, 17, 150, 39, 55, 54, 151, 14, 60],  # Execution Report
            'V': [262, 263, 264],            # Market Data Request
            'W': [55, 268],                  # Market Data Snapshot
        }


class OrderValidator:
    """
    Validator for order lifecycle and workflow validation.
    
    Provides validation for:
    - Order state transitions
    - Fill validation
    - Risk checks
    - Compliance rules
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.valid_transitions = self._load_valid_state_transitions()
        self.order_states: Dict[str, str] = {}
        self.order_fills: Dict[str, List[Dict]] = {}
    
    def validate_order_state_transition(
        self,
        cl_ord_id: str,
        old_status: str,
        new_status: str
    ) -> bool:
        """
        Validate order state transition is legal.
        
        Args:
            cl_ord_id: Client order ID
            old_status: Previous order status
            new_status: New order status
            
        Returns:
            bool: True if transition is valid
            
        Raises:
            ValidationError: If transition is invalid
        """
        if old_status not in self.valid_transitions:
            raise ValidationError(f"Unknown order status: {old_status}")
        
        valid_next_states = self.valid_transitions[old_status]
        
        if new_status not in valid_next_states:
            raise ValidationError(
                f"Invalid state transition for order {cl_ord_id}: "
                f"{old_status} -> {new_status}. Valid transitions: {valid_next_states}"
            )
        
        # Update tracked state
        self.order_states[cl_ord_id] = new_status
        
        return True
    
    def validate_execution_report(self, exec_report: fix.Message) -> bool:
        """
        Validate execution report for consistency and business rules.
        
        Args:
            exec_report: Execution report message
            
        Returns:
            bool: True if execution report is valid
            
        Raises:
            ValidationError: If execution report is invalid
        """
        try:
            cl_ord_id = exec_report.getField(11)
            exec_type = exec_report.getField(150)
            ord_status = exec_report.getField(39)
            
            # Validate basic fields
            symbol = exec_report.getField(55)
            side = exec_report.getField(54)
            
            # Validate quantities
            leaves_qty = float(exec_report.getField(151))
            cum_qty = float(exec_report.getField(14))
            
            if leaves_qty < 0:
                raise ValidationError("LeavesQty cannot be negative")
            
            if cum_qty < 0:
                raise ValidationError("CumQty cannot be negative")
            
            # Validate state transition
            if cl_ord_id in self.order_states:
                old_status = self.order_states[cl_ord_id]
                self.validate_order_state_transition(cl_ord_id, old_status, ord_status)
            else:
                self.order_states[cl_ord_id] = ord_status
            
            # Validate fill details if this is a trade
            if exec_type in ['F', '4']:  # Trade or Trade Correct
                self._validate_fill_details(exec_report)
            
            return True
            
        except Exception as e:
            raise ValidationError(f"Execution report validation failed: {e}")
    
    def validate_fill_consistency(self, cl_ord_id: str, fills: List[Dict]) -> bool:
        """
        Validate that fills are consistent with order and each other.
        
        Args:
            cl_ord_id: Client order ID
            fills: List of fill dictionaries
            
        Returns:
            bool: True if fills are consistent
            
        Raises:
            ValidationError: If fills are inconsistent
        """
        if not fills:
            return True
        
        total_qty = sum(fill['qty'] for fill in fills)
        total_value = sum(fill['qty'] * fill['price'] for fill in fills)
        
        # Check for negative quantities
        for fill in fills:
            if fill['qty'] <= 0:
                raise ValidationError(f"Fill quantity must be positive: {fill}")
            if fill['price'] <= 0:
                raise ValidationError(f"Fill price must be positive: {fill}")
        
        # Check fill timestamps are in order
        timestamps = [fill['timestamp'] for fill in fills]
        if timestamps != sorted(timestamps):
            raise ValidationError("Fill timestamps are not in chronological order")
        
        # Store fills for tracking
        self.order_fills[cl_ord_id] = fills
        
        return True
    
    def validate_risk_limits(
        self,
        order_data: Dict,
        position_limits: Optional[Dict] = None,
        credit_limits: Optional[Dict] = None
    ) -> bool:
        """
        Validate order against risk limits.
        
        Args:
            order_data: Order information
            position_limits: Position limit rules
            credit_limits: Credit limit rules
            
        Returns:
            bool: True if order passes risk checks
            
        Raises:
            ValidationError: If order violates risk limits
        """
        symbol = order_data.get('symbol')
        quantity = order_data.get('quantity', 0)
        price = order_data.get('price', 0)
        side = order_data.get('side')
        
        # Position limit checks
        if position_limits and symbol in position_limits:
            max_position = position_limits[symbol]
            if quantity > max_position:
                raise ValidationError(
                    f"Order quantity {quantity} exceeds position limit {max_position} for {symbol}"
                )
        
        # Credit limit checks
        if credit_limits:
            order_value = quantity * price
            max_order_value = credit_limits.get('max_order_value', float('inf'))
            
            if order_value > max_order_value:
                raise ValidationError(
                    f"Order value {order_value} exceeds credit limit {max_order_value}"
                )
        
        # Basic sanity checks
        if quantity <= 0:
            raise ValidationError("Order quantity must be positive")
        
        if price <= 0 and order_data.get('order_type') == 'LIMIT':
            raise ValidationError("Limit order price must be positive")
        
        if side not in ['1', '2', 'BUY', 'SELL']:
            raise ValidationError(f"Invalid order side: {side}")
        
        return True
    
    def get_order_state(self, cl_ord_id: str) -> Optional[str]:
        """Get current state of order."""
        return self.order_states.get(cl_ord_id)
    
    def get_order_fills(self, cl_ord_id: str) -> List[Dict]:
        """Get fills for order."""
        return self.order_fills.get(cl_ord_id, [])
    
    def _validate_fill_details(self, exec_report: fix.Message):
        """Validate fill-specific details in execution report."""
        try:
            last_qty = float(exec_report.getField(32))
            last_px = float(exec_report.getField(31))
            
            if last_qty <= 0:
                raise ValidationError("LastQty must be positive for fills")
            
            if last_px <= 0:
                raise ValidationError("LastPx must be positive for fills")
            
        except fix.FieldNotFound:
            raise ValidationError("Fill execution missing LastQty or LastPx")
    
    def _load_valid_state_transitions(self) -> Dict[str, List[str]]:
        """Load valid order state transitions."""
        return {
            # FIX order status transitions
            'A': ['0', '1', '2', '4', '8'],  # PendingNew -> New, PartiallyFilled, Filled, Canceled, Rejected
            '0': ['1', '2', '4', '6', '8'],  # New -> PartiallyFilled, Filled, Canceled, PendingCancel, Rejected
            '1': ['1', '2', '4', '6'],       # PartiallyFilled -> PartiallyFilled, Filled, Canceled, PendingCancel
            '2': ['2'],                      # Filled -> Filled (no transitions)
            '4': ['4'],                      # Canceled -> Canceled (no transitions)
            '6': ['1', '2', '4', '8'],       # PendingCancel -> PartiallyFilled, Filled, Canceled, Rejected
            '8': ['8'],                      # Rejected -> Rejected (no transitions)
            'E': ['0', '1', '2', '4', '8'],  # PendingReplace -> New, PartiallyFilled, Filled, Canceled, Rejected
        }


class WorkflowValidator:
    """
    Validator for complex trading workflows and scenarios.
    
    Provides validation for:
    - Multi-leg strategies
    - Time-based workflows
    - Cross-order dependencies
    - Market data consistency
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.workflow_state: Dict[str, Any] = {}
        self.workflow_rules: Dict[str, Any] = {}
    
    def validate_order_sequence(
        self,
        orders: List[Dict],
        sequence_rules: Dict[str, Any]
    ) -> bool:
        """
        Validate sequence of orders follows business rules.
        
        Args:
            orders: List of order dictionaries
            sequence_rules: Rules for order sequencing
            
        Returns:
            bool: True if sequence is valid
            
        Raises:
            ValidationError: If sequence violates rules
        """
        if not orders:
            return True
        
        # Check timing constraints
        if 'min_interval' in sequence_rules:
            min_interval = sequence_rules['min_interval']
            
            for i in range(1, len(orders)):
                time_diff = orders[i]['timestamp'] - orders[i-1]['timestamp']
                if time_diff < min_interval:
                    raise ValidationError(
                        f"Order sequence violates minimum interval: {time_diff} < {min_interval}"
                    )
        
        # Check quantity constraints
        if 'max_total_quantity' in sequence_rules:
            total_qty = sum(order['quantity'] for order in orders)
            max_qty = sequence_rules['max_total_quantity']
            
            if total_qty > max_qty:
                raise ValidationError(
                    f"Total order quantity {total_qty} exceeds maximum {max_qty}"
                )
        
        # Check side constraints
        if 'same_side_only' in sequence_rules and sequence_rules['same_side_only']:
            sides = set(order['side'] for order in orders)
            if len(sides) > 1:
                raise ValidationError("All orders in sequence must have same side")
        
        return True
    
    def validate_market_data_consistency(
        self,
        market_data: List[Dict],
        consistency_rules: Dict[str, Any]
    ) -> bool:
        """
        Validate market data for consistency and sanity.
        
        Args:
            market_data: List of market data updates
            consistency_rules: Rules for market data consistency
            
        Returns:
            bool: True if market data is consistent
            
        Raises:
            ValidationError: If market data is inconsistent
        """
        if not market_data:
            return True
        
        for data in market_data:
            # Check bid/offer spread
            if 'bid_price' in data and 'offer_price' in data:
                bid = data['bid_price']
                offer = data['offer_price']
                
                if bid >= offer:
                    raise ValidationError(f"Invalid spread: bid {bid} >= offer {offer}")
                
                # Check maximum spread
                if 'max_spread_pct' in consistency_rules:
                    spread_pct = (offer - bid) / bid * 100
                    max_spread = consistency_rules['max_spread_pct']
                    
                    if spread_pct > max_spread:
                        raise ValidationError(
                            f"Spread {spread_pct:.2f}% exceeds maximum {max_spread}%"
                        )
            
            # Check price movement limits
            if 'max_price_move_pct' in consistency_rules:
                # This would require comparing with previous prices
                pass
        
        return True
    
    def validate_strategy_execution(
        self,
        strategy_id: str,
        executed_orders: List[Dict],
        expected_orders: List[Dict],
        tolerance: float = 0.01
    ) -> bool:
        """
        Validate that strategy was executed according to plan.
        
        Args:
            strategy_id: Strategy identifier
            executed_orders: Actually executed orders
            expected_orders: Expected orders per strategy
            tolerance: Price tolerance for comparison
            
        Returns:
            bool: True if strategy executed correctly
            
        Raises:
            ValidationError: If strategy execution deviated from plan
        """
        if len(executed_orders) != len(expected_orders):
            raise ValidationError(
                f"Strategy {strategy_id}: Expected {len(expected_orders)} orders, "
                f"executed {len(executed_orders)}"
            )
        
        for i, (executed, expected) in enumerate(zip(executed_orders, expected_orders)):
            # Check symbol
            if executed['symbol'] != expected['symbol']:
                raise ValidationError(
                    f"Order {i}: Symbol mismatch - executed {executed['symbol']}, "
                    f"expected {expected['symbol']}"
                )
            
            # Check side
            if executed['side'] != expected['side']:
                raise ValidationError(
                    f"Order {i}: Side mismatch - executed {executed['side']}, "
                    f"expected {expected['side']}"
                )
            
            # Check quantity
            if abs(executed['quantity'] - expected['quantity']) > tolerance:
                raise ValidationError(
                    f"Order {i}: Quantity mismatch - executed {executed['quantity']}, "
                    f"expected {expected['quantity']}"
                )
            
            # Check price (if applicable)
            if 'price' in both expected and executed:
                price_diff = abs(executed['price'] - expected['price'])
                if price_diff > tolerance:
                    raise ValidationError(
                        f"Order {i}: Price mismatch - executed {executed['price']}, "
                        f"expected {expected['price']}"
                    )
        
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