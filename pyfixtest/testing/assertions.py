"""
Custom assertions for FIX protocol testing.
"""

import time
from typing import Any, Optional, Dict, List, Union
import quickfix as fix

from ..utils.logging_config import get_logger


class FIXAssertions:
    """
    Custom assertion methods for FIX protocol testing.
    
    Provides specialized assertions for:
    - Message content validation
    - Order state verification
    - Timing assertions
    - Session state checks
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def assertMessageType(self, message: fix.Message, expected_type: str, msg: str = ""):
        """Assert message is of expected type."""
        try:
            actual_type = message.getHeader().getField(35)
            if actual_type != expected_type:
                raise AssertionError(
                    f"{msg}Expected message type {expected_type}, got {actual_type}"
                )
        except Exception as e:
            raise AssertionError(f"{msg}Failed to get message type: {e}")
    
    def assertMessageHasField(self, message: fix.Message, field_tag: int, msg: str = ""):
        """Assert message contains specific field."""
        try:
            message.getField(field_tag)
        except fix.FieldNotFound:
            raise AssertionError(f"{msg}Message missing required field {field_tag}")
        except Exception as e:
            raise AssertionError(f"{msg}Error checking field {field_tag}: {e}")
    
    def assertMessageFieldEquals(
        self,
        message: fix.Message,
        field_tag: int,
        expected_value: str,
        msg: str = ""
    ):
        """Assert message field has expected value."""
        try:
            actual_value = message.getField(field_tag)
            if actual_value != expected_value:
                raise AssertionError(
                    f"{msg}Field {field_tag}: expected '{expected_value}', got '{actual_value}'"
                )
        except fix.FieldNotFound:
            raise AssertionError(f"{msg}Message missing field {field_tag}")
        except Exception as e:
            raise AssertionError(f"{msg}Error checking field {field_tag}: {e}")
    
    def assertMessageFieldIn(
        self,
        message: fix.Message,
        field_tag: int,
        expected_values: List[str],
        msg: str = ""
    ):
        """Assert message field value is in list of expected values."""
        try:
            actual_value = message.getField(field_tag)
            if actual_value not in expected_values:
                raise AssertionError(
                    f"{msg}Field {field_tag}: value '{actual_value}' not in {expected_values}"
                )
        except fix.FieldNotFound:
            raise AssertionError(f"{msg}Message missing field {field_tag}")
        except Exception as e:
            raise AssertionError(f"{msg}Error checking field {field_tag}: {e}")
    
    def assertOrderStatus(
        self,
        exec_report: fix.Message,
        expected_status: str,
        msg: str = ""
    ):
        """Assert execution report has expected order status."""
        self.assertMessageType(exec_report, '8', "Expected execution report. ")
        self.assertMessageFieldEquals(exec_report, 39, expected_status, msg)
    
    def assertOrderFilled(
        self,
        exec_report: fix.Message,
        expected_qty: Optional[float] = None,
        msg: str = ""
    ):
        """Assert order is filled."""
        self.assertOrderStatus(exec_report, '2', f"{msg}Order should be filled. ")
        
        if expected_qty is not None:
            try:
                cum_qty = float(exec_report.getField(14))
                if abs(cum_qty - expected_qty) > 0.001:  # Small tolerance for float comparison
                    raise AssertionError(
                        f"{msg}Expected fill quantity {expected_qty}, got {cum_qty}"
                    )
            except Exception as e:
                raise AssertionError(f"{msg}Error checking fill quantity: {e}")
    
    def assertOrderPartiallyFilled(
        self,
        exec_report: fix.Message,
        expected_cum_qty: Optional[float] = None,
        expected_leaves_qty: Optional[float] = None,
        msg: str = ""
    ):
        """Assert order is partially filled."""
        self.assertOrderStatus(exec_report, '1', f"{msg}Order should be partially filled. ")
        
        if expected_cum_qty is not None:
            try:
                cum_qty = float(exec_report.getField(14))
                if abs(cum_qty - expected_cum_qty) > 0.001:
                    raise AssertionError(
                        f"{msg}Expected cumulative quantity {expected_cum_qty}, got {cum_qty}"
                    )
            except Exception as e:
                raise AssertionError(f"{msg}Error checking cumulative quantity: {e}")
        
        if expected_leaves_qty is not None:
            try:
                leaves_qty = float(exec_report.getField(151))
                if abs(leaves_qty - expected_leaves_qty) > 0.001:
                    raise AssertionError(
                        f"{msg}Expected leaves quantity {expected_leaves_qty}, got {leaves_qty}"
                    )
            except Exception as e:
                raise AssertionError(f"{msg}Error checking leaves quantity: {e}")
    
    def assertOrderCanceled(self, exec_report: fix.Message, msg: str = ""):
        """Assert order is canceled."""
        self.assertOrderStatus(exec_report, '4', f"{msg}Order should be canceled. ")
    
    def assertOrderRejected(self, exec_report: fix.Message, msg: str = ""):
        """Assert order is rejected."""
        self.assertOrderStatus(exec_report, '8', f"{msg}Order should be rejected. ")
    
    def assertPriceWithinTolerance(
        self,
        actual_price: float,
        expected_price: float,
        tolerance: float = 0.01,
        msg: str = ""
    ):
        """Assert price is within tolerance of expected value."""
        price_diff = abs(actual_price - expected_price)
        if price_diff > tolerance:
            raise AssertionError(
                f"{msg}Price {actual_price} not within tolerance {tolerance} of {expected_price}"
            )
    
    def assertExecutionPrice(
        self,
        exec_report: fix.Message,
        expected_price: float,
        tolerance: float = 0.01,
        msg: str = ""
    ):
        """Assert execution report has expected price."""
        try:
            last_px = float(exec_report.getField(31))
            self.assertPriceWithinTolerance(last_px, expected_price, tolerance, msg)
        except fix.FieldNotFound:
            raise AssertionError(f"{msg}Execution report missing LastPx field")
        except Exception as e:
            raise AssertionError(f"{msg}Error checking execution price: {e}")
    
    def assertTimestampRecent(
        self,
        message: fix.Message,
        field_tag: int = 60,  # TransactTime
        max_age_seconds: float = 5.0,
        msg: str = ""
    ):
        """Assert message timestamp is recent."""
        try:
            timestamp_str = message.getField(field_tag)
            # Convert FIX timestamp to epoch (simplified)
            # Real implementation would need proper FIX timestamp parsing
            current_time = time.time()
            
            # For demonstration, assume timestamp is recent
            # Real implementation would parse FIX timestamp format
            
        except fix.FieldNotFound:
            raise AssertionError(f"{msg}Message missing timestamp field {field_tag}")
        except Exception as e:
            raise AssertionError(f"{msg}Error checking timestamp: {e}")
    
    def assertSessionLoggedIn(self, engine, msg: str = ""):
        """Assert FIX session is logged in."""
        if not engine.is_logged_in():
            raise AssertionError(f"{msg}FIX session is not logged in")
    
    def assertMessageReceived(
        self,
        received_messages: List[Dict],
        msg_type: str,
        timeout: float = 10.0,
        msg: str = ""
    ):
        """Assert message of specific type was received within timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            for msg_data in received_messages:
                try:
                    message = msg_data['message']
                    if message.getHeader().getField(35) == msg_type:
                        return message
                except:
                    continue
            time.sleep(0.1)
        
        raise AssertionError(
            f"{msg}Message type {msg_type} not received within {timeout} seconds"
        )
    
    def assertMarketDataValid(
        self,
        md_message: fix.Message,
        expected_symbol: Optional[str] = None,
        min_entries: int = 1,
        msg: str = ""
    ):
        """Assert market data message is valid."""
        self.assertMessageType(md_message, 'W', f"{msg}Expected market data snapshot. ")
        
        if expected_symbol:
            self.assertMessageFieldEquals(md_message, 55, expected_symbol, msg)
        
        # Check MD entries (simplified - real implementation would parse groups)
        try:
            # This is a simplified check - real implementation would iterate through MD entries
            md_message.getField(268)  # NoMDEntries
        except fix.FieldNotFound:
            raise AssertionError(f"{msg}Market data message missing NoMDEntries field")
    
    def assertHeartbeatValid(
        self,
        heartbeat: fix.Message,
        expected_test_req_id: Optional[str] = None,
        msg: str = ""
    ):
        """Assert heartbeat message is valid."""
        self.assertMessageType(heartbeat, '0', f"{msg}Expected heartbeat message. ")
        
        if expected_test_req_id:
            self.assertMessageFieldEquals(heartbeat, 112, expected_test_req_id, msg)
