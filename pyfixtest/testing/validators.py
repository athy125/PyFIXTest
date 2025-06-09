"""
Test validators for FIX trading system testing framework.

This module provides specialized validators for testing scenarios, including
enhanced validation, test-specific assertions, and compliance checking.
"""

import re
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
import quickfix as fix

from ..utils.logging_config import get_logger
from ..utils.time_utils import get_utc_timestamp, parse_fix_time, time_difference_seconds
from ..validators.validators import ValidationError, MessageValidator, OrderValidator


class TestValidationLevel(Enum):
    """Test validation levels."""
    BASIC = "BASIC"
    STANDARD = "STANDARD"
    STRICT = "STRICT"
    COMPLIANCE = "COMPLIANCE"


class TestValidationResult:
    """Result of test validation."""
    
    def __init__(self, passed: bool = True, message: str = ""):
        self.passed = passed
        self.message = message
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.details: Dict[str, Any] = {}
        self.timestamp = datetime.now(timezone.utc)
    
    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.passed = False
    
    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)
    
    def add_detail(self, key: str, value: Any):
        """Add validation detail."""
        self.details[key] = value
    
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.passed and len(self.errors) == 0
    
    def get_summary(self) -> str:
        """Get validation summary."""
        if self.is_valid():
            return f"PASSED: {self.message}"
        else:
            return f"FAILED: {self.message} - Errors: {len(self.errors)}, Warnings: {len(self.warnings)}"


class TestMessageValidator(MessageValidator):
    """
    Enhanced message validator for testing scenarios.
    
    Extends the base MessageValidator with test-specific validations,
    performance tracking, and detailed reporting.
    """
    
    def __init__(self, validation_level: TestValidationLevel = TestValidationLevel.STANDARD):
        super().__init__()
        self.validation_level = validation_level
        self.logger = get_logger(__name__)
        
        # Test-specific tracking
        self.validation_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        self.error_patterns: Dict[str, int] = {}
        
        # Validation rules by level
        self.validation_rules = self._load_validation_rules()
        
        # Custom test rules
        self.custom_rules: Dict[str, Callable] = {}
        self.test_expectations: Dict[str, Dict[str, Any]] = {}
    
    def validate_test_message(
        self,
        message: fix.Message,
        expected_fields: Optional[Dict[int, Any]] = None,
        forbidden_fields: Optional[List[int]] = None,
        custom_checks: Optional[List[Callable]] = None
    ) -> TestValidationResult:
        """
        Comprehensive test message validation.
        
        Args:
            message: FIX message to validate
            expected_fields: Expected field values
            forbidden_fields: Fields that should not be present
            custom_checks: Custom validation functions
            
        Returns:
            TestValidationResult: Detailed validation result
        """
        start_time = time.time()
        result = TestValidationResult(message="Test message validation")
        
        try:
            msg_type = message.getHeader().getField(35)
            result.add_detail("message_type", msg_type)
            
            # Basic structure validation
            if not self._validate_basic_structure(message, result):
                return result
            
            # Level-specific validation
            if not self._validate_by_level(message, result):
                return result
            
            # Expected fields validation
            if expected_fields and not self._validate_expected_fields(message, expected_fields, result):
                return result
            
            # Forbidden fields validation
            if forbidden_fields and not self._validate_forbidden_fields(message, forbidden_fields, result):
                return result
            
            # Custom checks
            if custom_checks:
                for check in custom_checks:
                    try:
                        if not check(message):
                            result.add_error(f"Custom validation check failed: {check.__name__}")
                    except Exception as e:
                        result.add_error(f"Custom check error: {e}")
            
            # Test-specific business rules
            self._validate_test_business_rules(message, result)
            
            # Performance and timing checks
            self._validate_timing_requirements(message, result)
            
        except Exception as e:
            result.add_error(f"Validation exception: {e}")
        
        finally:
            # Record performance
            validation_time = time.time() - start_time
            self._record_validation_performance(msg_type, validation_time)
            result.add_detail("validation_time_ms", validation_time * 1000)
            
            # Store validation history
            self._record_validation_history(message, result)
        
        return result
    
    def validate_message_sequence(
        self,
        messages: List[fix.Message],
        expected_sequence: Optional[List[str]] = None,
        timing_constraints: Optional[Dict[str, float]] = None
    ) -> TestValidationResult:
        """
        Validate sequence of messages for test scenarios.
        
        Args:
            messages: List of messages in sequence
            expected_sequence: Expected message type sequence
            timing_constraints: Timing requirements between messages
            
        Returns:
            TestValidationResult: Sequence validation result
        """
        result = TestValidationResult(message="Message sequence validation")
        
        try:
            if not messages:
                result.add_error("Empty message sequence")
                return result
            
            msg_types = []
            timestamps = []
            
            # Extract message types and timestamps
            for i, msg in enumerate(messages):
                try:
                    msg_type = msg.getHeader().getField(35)
                    msg_types.append(msg_type)
                    
                    # Try to get timestamp
                    try:
                        timestamp_str = msg.getField(52)  # SendingTime
                        timestamp = parse_fix_time(timestamp_str)
                        timestamps.append(timestamp)
                    except:
                        timestamps.append(None)
                        
                except Exception as e:
                    result.add_error(f"Failed to process message {i}: {e}")
            
            result.add_detail("message_types", msg_types)
            result.add_detail("message_count", len(messages))
            
            # Validate expected sequence
            if expected_sequence:
                if msg_types != expected_sequence:
                    result.add_error(f"Sequence mismatch. Expected: {expected_sequence}, Got: {msg_types}")
                else:
                    result.add_detail("sequence_match", True)
            
            # Validate timing constraints
            if timing_constraints and timestamps:
                self._validate_sequence_timing(timestamps, timing_constraints, result)
            
            # Validate sequence integrity
            self._validate_sequence_integrity(messages, result)
            
        except Exception as e:
            result.add_error(f"Sequence validation error: {e}")
        
        return result
    
    def validate_performance_requirements(
        self,
        operation_name: str,
        execution_time: float,
        requirements: Dict[str, float]
    ) -> TestValidationResult:
        """
        Validate performance requirements for test operations.
        
        Args:
            operation_name: Name of the operation
            execution_time: Actual execution time in seconds
            requirements: Performance requirements
            
        Returns:
            TestValidationResult: Performance validation result
        """
        result = TestValidationResult(message=f"Performance validation for {operation_name}")
        
        execution_time_ms = execution_time * 1000
        result.add_detail("execution_time_ms", execution_time_ms)
        result.add_detail("operation_name", operation_name)
        
        # Check maximum execution time
        if "max_time_ms" in requirements:
            max_time = requirements["max_time_ms"]
            if execution_time_ms > max_time:
                result.add_error(f"Execution time {execution_time_ms:.2f}ms exceeds maximum {max_time}ms")
            else:
                result.add_detail("max_time_check", "PASSED")
        
        # Check minimum execution time (for detecting caching issues)
        if "min_time_ms" in requirements:
            min_time = requirements["min_time_ms"]
            if execution_time_ms < min_time:
                result.add_warning(f"Execution time {execution_time_ms:.2f}ms below minimum {min_time}ms")
        
        # Check percentile requirements
        if "percentile_requirements" in requirements:
            # This would require historical data
            result.add_detail("percentile_check", "NOT_IMPLEMENTED")
        
        return result
    
    def add_custom_rule(self, rule_name: str, rule_func: Callable[[fix.Message], bool]):
        """Add custom validation rule."""
        self.custom_rules[rule_name] = rule_func
    
    def set_test_expectations(self, msg_type: str, expectations: Dict[str, Any]):
        """Set expectations for specific message type in tests."""
        self.test_expectations[msg_type] = expectations
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics for test reporting."""
        total_validations = len(self.validation_history)
        passed_validations = sum(1 for v in self.validation_history if v['result'].is_valid())
        
        stats = {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'failed_validations': total_validations - passed_validations,
            'success_rate': passed_validations / max(total_validations, 1) * 100,
            'error_patterns': dict(self.error_patterns),
            'average_validation_times': {},
        }
        
        # Calculate average validation times
        for msg_type, times in self.performance_metrics.items():
            if times:
                stats['average_validation_times'][msg_type] = {
                    'average_ms': sum(times) / len(times) * 1000,
                    'min_ms': min(times) * 1000,
                    'max_ms': max(times) * 1000,
                    'count': len(times)
                }
        
        return stats
    
    def _validate_basic_structure(self, message: fix.Message, result: TestValidationResult) -> bool:
        """Validate basic message structure."""
        try:
            self.validate_message_structure(message)
            result.add_detail("structure_check", "PASSED")
            return True
        except ValidationError as e:
            result.add_error(f"Structure validation failed: {e}")
            return False
    
    def _validate_by_level(self, message: fix.Message, result: TestValidationResult) -> bool:
        """Validate based on validation level."""
        try:
            msg_type = message.getHeader().getField(35)
            
            if self.validation_level == TestValidationLevel.BASIC:
                # Only basic structure checks
                pass
                
            elif self.validation_level == TestValidationLevel.STANDARD:
                # Standard validation with required fields
                self.validate_required_fields(message, msg_type)
                
            elif self.validation_level == TestValidationLevel.STRICT:
                # Strict validation with field value checks
                self.validate_required_fields(message, msg_type)
                rules = self.validation_rules.get(msg_type, {})
                if rules:
                    self.validate_field_values(message, rules)
                    
            elif self.validation_level == TestValidationLevel.COMPLIANCE:
                # Full compliance validation
                self.validate_required_fields(message, msg_type)
                rules = self.validation_rules.get(msg_type, {})
                if rules:
                    self.validate_field_values(message, rules)
                self._validate_compliance_rules(message, result)
            
            result.add_detail("level_validation", self.validation_level.value)
            return True
            
        except ValidationError as e:
            result.add_error(f"Level validation failed: {e}")
            return False
    
    def _validate_expected_fields(
        self,
        message: fix.Message,
        expected_fields: Dict[int, Any],
        result: TestValidationResult
    ) -> bool:
        """Validate expected field values."""
        mismatches = []
        
        for field_tag, expected_value in expected_fields.items():
            try:
                actual_value = message.getField(field_tag)
                
                if isinstance(expected_value, (list, tuple)):
                    # Value should be in list
                    if actual_value not in expected_value:
                        mismatches.append(f"Field {field_tag}: expected one of {expected_value}, got {actual_value}")
                elif callable(expected_value):
                    # Custom validation function
                    if not expected_value(actual_value):
                        mismatches.append(f"Field {field_tag}: custom validation failed for value {actual_value}")
                else:
                    # Exact value match
                    if str(actual_value) != str(expected_value):
                        mismatches.append(f"Field {field_tag}: expected {expected_value}, got {actual_value}")
                        
            except fix.FieldNotFound:
                mismatches.append(f"Field {field_tag}: expected {expected_value}, but field not found")
        
        if mismatches:
            for mismatch in mismatches:
                result.add_error(mismatch)
            return False
        
        result.add_detail("expected_fields_check", "PASSED")
        return True
    
    def _validate_forbidden_fields(
        self,
        message: fix.Message,
        forbidden_fields: List[int],
        result: TestValidationResult
    ) -> bool:
        """Validate that forbidden fields are not present."""
        found_forbidden = []
        
        for field_tag in forbidden_fields:
            if message.isSetField(field_tag):
                found_forbidden.append(field_tag)
        
        if found_forbidden:
            result.add_error(f"Forbidden fields present: {found_forbidden}")
            return False
        
        result.add_detail("forbidden_fields_check", "PASSED")
        return True
    
    def _validate_test_business_rules(self, message: fix.Message, result: TestValidationResult):
        """Apply test-specific business rules."""
        try:
            msg_type = message.getHeader().getField(35)
            
            # Apply custom rules
            for rule_name, rule_func in self.custom_rules.items():
                try:
                    if not rule_func(message):
                        result.add_error(f"Custom rule '{rule_name}' failed")
                except Exception as e:
                    result.add_error(f"Custom rule '{rule_name}' error: {e}")
            
            # Apply test expectations
            if msg_type in self.test_expectations:
                expectations = self.test_expectations[msg_type]
                
                # Check field presence expectations
                if "required_fields" in expectations:
                    for field_tag in expectations["required_fields"]:
                        if not message.isSetField(field_tag):
                            result.add_error(f"Expected field {field_tag} not present")
                
                # Check field value expectations
                if "field_values" in expectations:
                    for field_tag, expected_value in expectations["field_values"].items():
                        try:
                            actual_value = message.getField(field_tag)
                            if str(actual_value) != str(expected_value):
                                result.add_error(f"Field {field_tag}: expected {expected_value}, got {actual_value}")
                        except:
                            result.add_error(f"Expected field {field_tag} with value {expected_value} not found")
            
        except Exception as e:
            result.add_warning(f"Business rule validation error: {e}")
    
    def _validate_timing_requirements(self, message: fix.Message, result: TestValidationResult):
        """Validate timing-related requirements."""
        try:
            # Check for SendingTime field
            if message.isSetField(52):
                sending_time_str = message.getField(52)
                sending_time = parse_fix_time(sending_time_str)
                current_time = datetime.now(timezone.utc)
                
                # Check if timestamp is reasonable (not too old or in future)
                time_diff = abs((current_time - sending_time).total_seconds())
                
                if time_diff > 300:  # 5 minutes
                    result.add_warning(f"SendingTime differs by {time_diff:.1f} seconds from current time")
                
                result.add_detail("sending_time_check", "PASSED")
                result.add_detail("time_difference_seconds", time_diff)
            
        except Exception as e:
            result.add_warning(f"Timing validation error: {e}")
    
    def _validate_sequence_timing(
        self,
        timestamps: List[Optional[datetime]],
        constraints: Dict[str, float],
        result: TestValidationResult
    ):
        """Validate timing constraints in message sequence."""
        valid_timestamps = [t for t in timestamps if t is not None]
        
        if len(valid_timestamps) < 2:
            result.add_warning("Insufficient timestamps for timing validation")
            return
        
        # Check maximum time between messages
        if "max_interval_seconds" in constraints:
            max_interval = constraints["max_interval_seconds"]
            
            for i in range(1, len(valid_timestamps)):
                interval = (valid_timestamps[i] - valid_timestamps[i-1]).total_seconds()
                if interval > max_interval:
                    result.add_error(f"Message interval {interval:.2f}s exceeds maximum {max_interval}s")
        
        # Check minimum time between messages
        if "min_interval_seconds" in constraints:
            min_interval = constraints["min_interval_seconds"]
            
            for i in range(1, len(valid_timestamps)):
                interval = (valid_timestamps[i] - valid_timestamps[i-1]).total_seconds()
                if interval < min_interval:
                    result.add_warning(f"Message interval {interval:.2f}s below minimum {min_interval}s")
        
        # Check total sequence time
        if "max_total_time_seconds" in constraints:
            total_time = (valid_timestamps[-1] - valid_timestamps[0]).total_seconds()
            max_total = constraints["max_total_time_seconds"]
            
            if total_time > max_total:
                result.add_error(f"Total sequence time {total_time:.2f}s exceeds maximum {max_total}s")
    
    def _validate_sequence_integrity(self, messages: List[fix.Message], result: TestValidationResult):
        """Validate integrity of message sequence."""
        try:
            # Check for duplicate ClOrdIDs in order messages
            cl_ord_ids = []
            for msg in messages:
                if msg.isSetField(11):  # ClOrdID
                    cl_ord_id = msg.getField(11)
                    if cl_ord_id in cl_ord_ids:
                        result.add_warning(f"Duplicate ClOrdID found: {cl_ord_id}")
                    cl_ord_ids.append(cl_ord_id)
            
            # Check for proper order state transitions
            order_states = {}
            for msg in messages:
                try:
                    msg_type = msg.getHeader().getField(35)
                    if msg_type == '8' and msg.isSetField(11):  # Execution Report
                        cl_ord_id = msg.getField(11)
                        ord_status = msg.getField(39)
                        
                        if cl_ord_id in order_states:
                            # Validate state transition
                            old_status = order_states[cl_ord_id]
                            if not self._is_valid_state_transition(old_status, ord_status):
                                result.add_error(f"Invalid state transition for {cl_ord_id}: {old_status} -> {ord_status}")
                        
                        order_states[cl_ord_id] = ord_status
                except:
                    continue
            
            result.add_detail("sequence_integrity_check", "COMPLETED")
            
        except Exception as e:
            result.add_warning(f"Sequence integrity validation error: {e}")
    
    def _validate_compliance_rules(self, message: fix.Message, result: TestValidationResult):
        """Validate regulatory compliance rules."""
        try:
            msg_type = message.getHeader().getField(35)
            
            # MiFID II compliance checks for orders
            if msg_type == 'D':  # New Order Single
                self._validate_mifid_compliance(message, result)
            
            # Best execution checks
            if msg_type == '8':  # Execution Report
                self._validate_execution_compliance(message, result)
            
            result.add_detail("compliance_check", "COMPLETED")
            
        except Exception as e:
            result.add_warning(f"Compliance validation error: {e}")
    
    def _validate_mifid_compliance(self, message: fix.Message, result: TestValidationResult):
        """Validate MiFID II compliance for orders."""
        # Check for required MiFID fields (simplified)
        mifid_fields = [
            # Add MiFID II required fields
        ]
        
        for field_tag in mifid_fields:
            if not message.isSetField(field_tag):
                result.add_warning(f"MiFID II: Missing field {field_tag}")
    
    def _validate_execution_compliance(self, message: fix.Message, result: TestValidationResult):
        """Validate execution compliance."""
        # Check execution timing
        if message.isSetField(60):  # TransactTime
            # Validate execution timestamp is reasonable
            pass
    
    def _is_valid_state_transition(self, old_status: str, new_status: str) -> bool:
        """Check if order state transition is valid."""
        valid_transitions = {
            'A': ['0', '1', '2', '4', '8'],  # PendingNew
            '0': ['1', '2', '4', '6', '8'],  # New
            '1': ['1', '2', '4', '6'],       # PartiallyFilled
            '2': ['2'],                      # Filled
            '4': ['4'],                      # Canceled
            '6': ['1', '2', '4', '8'],       # PendingCancel
            '8': ['8'],                      # Rejected
        }
        
        return new_status in valid_transitions.get(old_status, [])
    
    def _record_validation_performance(self, msg_type: str, validation_time: float):
        """Record validation performance metrics."""
        if msg_type not in self.performance_metrics:
            self.performance_metrics[msg_type] = []
        
        times = self.performance_metrics[msg_type]
        times.append(validation_time)
        
        # Keep only last 100 measurements
        if len(times) > 100:
            times.pop(0)
    
    def _record_validation_history(self, message: fix.Message, result: TestValidationResult):
        """Record validation in history."""
        try:
            msg_type = message.getHeader().getField(35)
            
            history_entry = {
                'timestamp': result.timestamp,
                'message_type': msg_type,
                'result': result,
                'validation_level': self.validation_level.value
            }
            
            self.validation_history.append(history_entry)
            
            # Update error patterns
            for error in result.errors:
                if error not in self.error_patterns:
                    self.error_patterns[error] = 0
                self.error_patterns[error] += 1
            
            # Keep only last 1000 entries
            if len(self.validation_history) > 1000:
                self.validation_history.pop(0)
                
        except Exception as e:
            self.logger.warning(f"Failed to record validation history: {e}")
    
    def _load_validation_rules(self) -> Dict[str, Dict[int, Any]]:
        """Load validation rules based on level."""
        return {
            'D': {  # New Order Single
                54: ['1', '2'],  # Side: Buy/Sell
                40: ['1', '2', '3', '4'],  # OrdType
                38: {'min': 0},  # OrderQty
                44: {'min': 0},  # Price (when applicable)
            },
            '8': {  # Execution Report
                39: ['0', '1', '2', '4', '6', '8', 'A', 'E'],  # OrdStatus
                150: ['0', '4', 'F'],  # ExecType
                14: {'min': 0},  # CumQty
                151: {'min': 0},  # LeavesQty
            },
            'F': {  # Order Cancel Request
                54: ['1', '2'],  # Side
            },
            'G': {  # Order Cancel/Replace Request
                54: ['1', '2'],  # Side
                38: {'min': 0},  # OrderQty
            }
        }


class TestOrderValidator(OrderValidator):
    """
    Enhanced order validator for testing scenarios.
    
    Extends OrderValidator with test-specific order lifecycle validation,
    scenario testing, and detailed reporting capabilities.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        
        # Test-specific tracking
        self.test_orders: Dict[str, Dict[str, Any]] = {}
        self.scenario_results: Dict[str, List[TestValidationResult]] = {}
        self.performance_tracking: Dict[str, List[float]] = {}
    
    def validate_order_scenario(
        self,
        scenario_name: str,
        orders: List[Dict[str, Any]],
        expected_outcomes: Dict[str, Any]
    ) -> TestValidationResult:
        """
        Validate complete order scenario.
        
        Args:
            scenario_name: Name of the test scenario
            orders: List of order data
            expected_outcomes: Expected scenario outcomes
            
        Returns:
            TestValidationResult: Scenario validation result
        """
        result = TestValidationResult(message=f"Order scenario validation: {scenario_name}")
        
        try:
            # Validate individual orders
            for i, order_data in enumerate(orders):
                order_result = self._validate_single_order_in_scenario(order_data, i, result)
                if not order_result:
                    result.add_error(f"Order {i} validation failed")
            
            # Validate scenario outcomes
            self._validate_scenario_outcomes(orders, expected_outcomes, result)
            
            # Store scenario result
            if scenario_name not in self.scenario_results:
                self.scenario_results[scenario_name] = []
            self.scenario_results[scenario_name].append(result)
            
        except Exception as e:
            result.add_error(f"Scenario validation error: {e}")
        
        return result
    
    def validate_order_lifecycle_timing(
        self,
        order_events: List[Dict[str, Any]],
        timing_requirements: Dict[str, float]
    ) -> TestValidationResult:
        """
        Validate order lifecycle timing requirements.
        
        Args:
            order_events: List of order lifecycle events
            timing_requirements: Required timing constraints
            
        Returns:
            TestValidationResult: Timing validation result
        """
        result = TestValidationResult(message="Order lifecycle timing validation")
        
        try:
            if len(order_events) < 2:
                result.add_warning("Insufficient events for timing validation")
                return result
            
            # Group events by order
            orders = {}
            for event in order_events:
                cl_ord_id = event.get('cl_ord_id')
                if cl_ord_id:
                    if cl_ord_id not in orders:
                        orders[cl_ord_id] = []
                    orders[cl_ord_id].append(event)
            
            # Validate timing for each order
            for cl_ord_id, events in orders.items():
                events.sort(key=lambda x: x.get('timestamp', datetime.min))
                self._validate_order_timing(cl_ord_id, events, timing_requirements, result)
            
        except Exception as e:
            result.add_error(f"Timing validation error: {e}")
        
        return result
    
    def validate_fill_accuracy(
        self,
        expected_fills: List[Dict[str, Any]],
        actual_fills: List[Dict[str, Any]],
        tolerance: float = 0.01
    ) -> TestValidationResult:
        """
        Validate fill accuracy against expected fills.
        
        Args:
            expected_fills: Expected fill data
            actual_fills: Actual fill data
            tolerance: Price/quantity tolerance
            
        Returns:
            TestValidationResult: Fill accuracy validation result
        """
        result = TestValidationResult(message="Fill accuracy validation")
        
        try:
            if len(expected_fills) != len(actual_fills):
                result.add_error(f"Fill count mismatch: expected {len(expected_fills)}, got {len(actual_fills)}")
                return result
            
            total_expected_qty = sum(fill.get('quantity', 0) for fill in expected_fills)
            total_actual_qty = sum(fill.get('quantity', 0) for fill in actual_fills)
            
            if abs(total_expected_qty - total_actual_qty) > tolerance:
                result.add_error(f"Total quantity mismatch: expected {total_expected_qty}, got {total_actual_qty}")
            
            # Validate individual fills
            for i, (expected, actual) in enumerate(zip(expected_fills, actual_fills)):
                self._validate_single_fill(i, expected, actual, tolerance, result)
            
            result.add_detail("total_expected_quantity", total_expected_qty)
            result.add_detail("total_actual_quantity", total_actual_qty)
            
        except Exception as e:
            result.add_error(f"Fill accuracy validation error: {e}")
        
        return result
    
    def get_scenario_statistics(self) -> Dict[str, Any]:
        """Get statistics for all validated scenarios."""
        stats = {}
        
        for scenario_name, results in self.scenario_results.items():
            passed = sum(1 for r in results if r.is_valid())
            total = len(results)
            
            stats[scenario_name] = {
                'total_runs': total,
                'passed_runs': passed,
                'failed_runs': total - passed,
                'success_rate': passed / max(total, 1) * 100,
                'last_run': results[-1].timestamp if results else None,
                'common_errors': self._get_common_errors(results)
            }
        
        return stats
    
    def _validate_single_order_in_scenario(
        self,
        order_data: Dict[str, Any],
        order_index: int,
        result: TestValidationResult
    ) -> bool:
        """Validate single order within a scenario."""
        try:
            # Basic order validation
            if not self.validate_risk_limits(order_data):
                result.add_error(f"Order {order_index}: Risk limits validation failed")
                return False
            
            # Store order for tracking
            cl_ord_id = order_data.get('cl_ord_id', f"test_order_{order_index}")
            self.test_orders[cl_ord_id] = order_data
            
            return True
            
        except Exception as e:
            result.add_error(f"Order {order_index} validation error: {e}")
            return False
    
    def _validate_scenario_outcomes(
        self,
        orders: List[Dict[str, Any]],
        expected_outcomes: Dict[str, Any],
        result: TestValidationResult
    ):
        """Validate overall scenario outcomes."""
        try:
            # Validate total quantities
            if 'total_quantity' in expected_outcomes:
                actual_total = sum(order.get('quantity', 0) for order in orders)
                expected_total = expected_outcomes['total_quantity']
                
                if actual_total != expected_total:
                    result.add_error(f"Total quantity mismatch: expected {expected_total}, got {actual_total}")
            
            # Validate order count
            if 'order_count' in expected_outcomes:
                expected_count = expected_outcomes['order_count']
                actual_count = len(orders)
                
                if actual_count != expected_count:
                    result.add_error(f"Order count mismatch: expected {expected_count}, got {actual_count}")
            
            # Validate order types distribution
            if 'order_types' in expected_outcomes:
                expected_types = expected_outcomes['order_types']
                actual_types = {}
                
                for order in orders:
                    order_type = order.get('order_type', 'UNKNOWN')
                    actual_types[order_type] = actual_types.get(order_type, 0) + 1
                
                for order_type, expected_count in expected_types.items():
                    actual_count = actual_types.get(order_type, 0)
                    if actual_count != expected_count:
                        result.add_error(f"Order type {order_type}: expected {expected_count}, got {actual_count}")
            
            # Validate symbols distribution
            if 'symbols' in expected_outcomes:
                expected_symbols = set(expected_outcomes['symbols'])
                actual_symbols = set(order.get('symbol') for order in orders)
                
                if actual_symbols != expected_symbols:
                    result.add_error(f"Symbol mismatch: expected {expected_symbols}, got {actual_symbols}")
            
        except Exception as e:
            result.add_warning(f"Scenario outcome validation error: {e}")
    
    def _validate_order_timing(
        self,
        cl_ord_id: str,
        events: List[Dict[str, Any]],
        timing_requirements: Dict[str, float],
        result: TestValidationResult
    ):
        """Validate timing for a single order."""
        try:
            if len(events) < 2:
                return
            
            # Validate time to first ack
            if 'max_ack_time_ms' in timing_requirements:
                first_event = events[0]
                ack_event = None
                
                for event in events[1:]:
                    if event.get('event_type') in ['ACK', 'FILL', 'REJECT']:
                        ack_event = event
                        break
                
                if ack_event:
                    time_diff = (ack_event['timestamp'] - first_event['timestamp']).total_seconds() * 1000
                    max_ack_time = timing_requirements['max_ack_time_ms']
                    
                    if time_diff > max_ack_time:
                        result.add_error(f"Order {cl_ord_id}: Ack time {time_diff:.1f}ms exceeds maximum {max_ack_time}ms")
            
            # Validate time to fill
            if 'max_fill_time_ms' in timing_requirements:
                first_event = events[0]
                fill_event = None
                
                for event in events[1:]:
                    if event.get('event_type') == 'FILL':
                        fill_event = event
                        break
                
                if fill_event:
                    time_diff = (fill_event['timestamp'] - first_event['timestamp']).total_seconds() * 1000
                    max_fill_time = timing_requirements['max_fill_time_ms']
                    
                    if time_diff > max_fill_time:
                        result.add_error(f"Order {cl_ord_id}: Fill time {time_diff:.1f}ms exceeds maximum {max_fill_time}ms")
            
        except Exception as e:
            result.add_warning(f"Order timing validation error for {cl_ord_id}: {e}")
    
    def _validate_single_fill(
        self,
        fill_index: int,
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        tolerance: float,
        result: TestValidationResult
    ):
        """Validate a single fill against expected values."""
        try:
            # Validate quantity
            expected_qty = expected.get('quantity', 0)
            actual_qty = actual.get('quantity', 0)
            
            if abs(expected_qty - actual_qty) > tolerance:
                result.add_error(f"Fill {fill_index}: Quantity mismatch - expected {expected_qty}, got {actual_qty}")
            
            # Validate price
            expected_price = expected.get('price', 0)
            actual_price = actual.get('price', 0)
            
            if abs(expected_price - actual_price) > tolerance:
                result.add_error(f"Fill {fill_index}: Price mismatch - expected {expected_price}, got {actual_price}")
            
            # Validate symbol
            expected_symbol = expected.get('symbol')
            actual_symbol = actual.get('symbol')
            
            if expected_symbol and actual_symbol and expected_symbol != actual_symbol:
                result.add_error(f"Fill {fill_index}: Symbol mismatch - expected {expected_symbol}, got {actual_symbol}")
            
            # Validate side
            expected_side = expected.get('side')
            actual_side = actual.get('side')
            
            if expected_side and actual_side and expected_side != actual_side:
                result.add_error(f"Fill {fill_index}: Side mismatch - expected {expected_side}, got {actual_side}")
            
        except Exception as e:
            result.add_warning(f"Fill {fill_index} validation error: {e}")
    
    def _get_common_errors(self, results: List[TestValidationResult]) -> List[Dict[str, Any]]:
        """Get common errors from validation results."""
        error_counts = {}
        
        for result in results:
            for error in result.errors:
                if error not in error_counts:
                    error_counts[error] = 0
                error_counts[error] += 1
        
        # Sort by frequency and return top 5
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'error': error, 'count': count} for error, count in sorted_errors[:5]]


class TestSessionValidator:
    """
    Session validator for testing FIX session behavior and compliance.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.session_history: Dict[str, List[Dict[str, Any]]] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
    
    def validate_session_lifecycle(
        self,
        session_events: List[Dict[str, Any]],
        expected_sequence: List[str]
    ) -> TestValidationResult:
        """
        Validate session lifecycle events.
        
        Args:
            session_events: List of session events
            expected_sequence: Expected event sequence
            
        Returns:
            TestValidationResult: Session lifecycle validation result
        """
        result = TestValidationResult(message="Session lifecycle validation")
        
        try:
            event_types = [event.get('event_type') for event in session_events]
            result.add_detail("actual_sequence", event_types)
            result.add_detail("expected_sequence", expected_sequence)
            
            if event_types != expected_sequence:
                result.add_error(f"Session sequence mismatch. Expected: {expected_sequence}, Got: {event_types}")
            
            # Validate event timing
            self._validate_session_timing(session_events, result)
            
            # Validate event data consistency
            self._validate_session_consistency(session_events, result)
            
        except Exception as e:
            result.add_error(f"Session lifecycle validation error: {e}")
        
        return result
    
    def validate_heartbeat_behavior(
        self,
        heartbeat_events: List[Dict[str, Any]],
        expected_interval: int,
        tolerance: float = 0.1
    ) -> TestValidationResult:
        """
        Validate heartbeat timing and behavior.
        
        Args:
            heartbeat_events: List of heartbeat events
            expected_interval: Expected heartbeat interval in seconds
            tolerance: Timing tolerance as fraction of interval
            
        Returns:
            TestValidationResult: Heartbeat validation result
        """
        result = TestValidationResult(message="Heartbeat behavior validation")
        
        try:
            if len(heartbeat_events) < 2:
                result.add_warning("Insufficient heartbeat events for validation")
                return result
            
            intervals = []
            for i in range(1, len(heartbeat_events)):
                prev_time = heartbeat_events[i-1].get('timestamp')
                curr_time = heartbeat_events[i].get('timestamp')
                
                if prev_time and curr_time:
                    interval = (curr_time - prev_time).total_seconds()
                    intervals.append(interval)
            
            # Check intervals
            tolerance_seconds = expected_interval * tolerance
            min_interval = expected_interval - tolerance_seconds
            max_interval = expected_interval + tolerance_seconds
            
            violations = []
            for i, interval in enumerate(intervals):
                if interval < min_interval or interval > max_interval:
                    violations.append(f"Interval {i}: {interval:.2f}s (expected {expected_interval}±{tolerance_seconds:.2f}s)")
            
            if violations:
                result.add_error(f"Heartbeat timing violations: {violations}")
            
            # Calculate statistics
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                result.add_detail("average_interval", avg_interval)
                result.add_detail("min_interval", min(intervals))
                result.add_detail("max_interval", max(intervals))
                result.add_detail("expected_interval", expected_interval)
            
        except Exception as e:
            result.add_error(f"Heartbeat validation error: {e}")
        
        return result
    
    def validate_session_recovery(
        self,
        recovery_events: List[Dict[str, Any]],
        max_recovery_time: float
    ) -> TestValidationResult:
        """
        Validate session recovery behavior.
        
        Args:
            recovery_events: List of recovery events
            max_recovery_time: Maximum allowed recovery time in seconds
            
        Returns:
            TestValidationResult: Recovery validation result
        """
        result = TestValidationResult(message="Session recovery validation")
        
        try:
            disconnect_event = None
            reconnect_event = None
            
            for event in recovery_events:
                if event.get('event_type') == 'DISCONNECT':
                    disconnect_event = event
                elif event.get('event_type') == 'RECONNECT' and disconnect_event:
                    reconnect_event = event
                    break
            
            if not disconnect_event or not reconnect_event:
                result.add_error("Could not find disconnect/reconnect event pair")
                return result
            
            recovery_time = (reconnect_event['timestamp'] - disconnect_event['timestamp']).total_seconds()
            
            if recovery_time > max_recovery_time:
                result.add_error(f"Recovery time {recovery_time:.2f}s exceeds maximum {max_recovery_time}s")
            
            result.add_detail("recovery_time", recovery_time)
            result.add_detail("max_recovery_time", max_recovery_time)
            
        except Exception as e:
            result.add_error(f"Session recovery validation error: {e}")
        
        return result
    
    def _validate_session_timing(self, events: List[Dict[str, Any]], result: TestValidationResult):
        """Validate timing aspects of session events."""
        try:
            # Check for reasonable timing between events
            for i in range(1, len(events)):
                prev_event = events[i-1]
                curr_event = events[i]
                
                if 'timestamp' in prev_event and 'timestamp' in curr_event:
                    time_diff = (curr_event['timestamp'] - prev_event['timestamp']).total_seconds()
                    
                    # Warn about suspicious timing
                    if time_diff < 0:
                        result.add_error(f"Events out of order: {prev_event.get('event_type')} -> {curr_event.get('event_type')}")
                    elif time_diff > 300:  # 5 minutes
                        result.add_warning(f"Large gap between events: {time_diff:.1f} seconds")
            
        except Exception as e:
            result.add_warning(f"Session timing validation error: {e}")
    
    def _validate_session_consistency(self, events: List[Dict[str, Any]], result: TestValidationResult):
        """Validate consistency of session event data."""
        try:
            session_id = None
            
            for event in events:
                # Check session ID consistency
                event_session_id = event.get('session_id')
                if session_id is None:
                    session_id = event_session_id
                elif event_session_id and event_session_id != session_id:
                    result.add_error(f"Session ID inconsistency: {session_id} vs {event_session_id}")
                
                # Validate event data completeness
                required_fields = ['event_type', 'timestamp']
                for field in required_fields:
                    if field not in event or event[field] is None:
                        result.add_warning(f"Missing required field '{field}' in event")
            
        except Exception as e:
            result.add_warning(f"Session consistency validation error: {e}")


class ComplianceValidator:
    """
    Validator for regulatory compliance and best practices.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.compliance_rules = self._load_compliance_rules()
    
    def validate_regulatory_compliance(
        self,
        messages: List[fix.Message],
        regulation: str = "MiFID_II"
    ) -> TestValidationResult:
        """
        Validate regulatory compliance for message flows.
        
        Args:
            messages: List of FIX messages to validate
            regulation: Regulation to validate against
            
        Returns:
            TestValidationResult: Compliance validation result
        """
        result = TestValidationResult(message=f"Regulatory compliance validation: {regulation}")
        
        try:
            rules = self.compliance_rules.get(regulation, {})
            
            for i, message in enumerate(messages):
                msg_type = message.getHeader().getField(35)
                
                # Apply regulation-specific rules
                if msg_type in rules:
                    msg_rules = rules[msg_type]
                    self._apply_compliance_rules(message, msg_rules, f"Message {i}", result)
            
            result.add_detail("regulation", regulation)
            result.add_detail("messages_checked", len(messages))
            
        except Exception as e:
            result.add_error(f"Compliance validation error: {e}")
        
        return result
    
    def validate_best_execution(
        self,
        execution_data: List[Dict[str, Any]],
        benchmarks: Dict[str, float]
    ) -> TestValidationResult:
        """
        Validate best execution requirements.
        
        Args:
            execution_data: List of execution records
            benchmarks: Performance benchmarks
            
        Returns:
            TestValidationResult: Best execution validation result
        """
        result = TestValidationResult(message="Best execution validation")
        
        try:
            total_value = sum(exec['quantity'] * exec['price'] for exec in execution_data)
            weighted_avg_price = total_value / sum(exec['quantity'] for exec in execution_data)
            
            # Compare against benchmarks
            if 'vwap_benchmark' in benchmarks:
                vwap_diff = abs(weighted_avg_price - benchmarks['vwap_benchmark'])
                if vwap_diff > benchmarks.get('max_vwap_deviation', 0.01):
                    result.add_error(f"VWAP deviation {vwap_diff:.4f} exceeds threshold")
            
            result.add_detail("weighted_average_price", weighted_avg_price)
            result.add_detail("total_executions", len(execution_data))
            
        except Exception as e:
            result.add_error(f"Best execution validation error: {e}")
        
        return result
    
    def validate_results(self, results: Any, criteria: Dict[str, Any]) -> TestValidationResult:
        """
        Validate test results against specified criteria.
        
        Args:
            results: Test results to validate
            criteria: Validation criteria
            
        Returns:
            TestValidationResult: Validation result
        """
        result = TestValidationResult(message="Results validation")
        
        try:
            # Validate performance criteria
            if 'performance' in criteria and hasattr(results, 'performance_metrics'):
                perf_criteria = criteria['performance']
                metrics = results.performance_metrics
                
                for metric_name, threshold in perf_criteria.items():
                    if metric_name in metrics:
                        actual_value = metrics[metric_name]
                        if isinstance(threshold, dict):
                            # Range validation
                            if 'min' in threshold and actual_value < threshold['min']:
                                result.add_error(f"{metric_name}: {actual_value} below minimum {threshold['min']}")
                            if 'max' in threshold and actual_value > threshold['max']:
                                result.add_error(f"{metric_name}: {actual_value} above maximum {threshold['max']}")
                        else:
                            # Simple threshold
                            if actual_value > threshold:
                                result.add_error(f"{metric_name}: {actual_value} exceeds threshold {threshold}")
            
            # Validate accuracy criteria
            if 'accuracy' in criteria and hasattr(results, 'accuracy_metrics'):
                acc_criteria = criteria['accuracy']
                metrics = results.accuracy_metrics
                
                for metric_name, threshold in acc_criteria.items():
                    if metric_name in metrics:
                        actual_value = metrics[metric_name]
                        if actual_value < threshold:
                            result.add_error(f"{metric_name}: {actual_value} below required {threshold}")
            
        except Exception as e:
            result.add_error(f"Results validation error: {e}")
        
        return result
    
    def _apply_compliance_rules(
        self,
        message: fix.Message,
        rules: Dict[str, Any],
        context: str,
        result: TestValidationResult
    ):
        """Apply compliance rules to a message."""
        try:
            # Required fields check
            if 'required_fields' in rules:
                for field_tag in rules['required_fields']:
                    if not message.isSetField(field_tag):
                        result.add_error(f"{context}: Missing required field {field_tag}")
            
            # Field value constraints
            if 'field_constraints' in rules:
                constraints = rules['field_constraints']
                for field_tag, constraint in constraints.items():
                    if message.isSetField(field_tag):
                        field_value = message.getField(field_tag)
                        
                        if isinstance(constraint, list):
                            if field_value not in constraint:
                                result.add_error(f"{context}: Field {field_tag} value {field_value} not in allowed values {constraint}")
                        elif isinstance(constraint, dict):
                            if 'pattern' in constraint:
                                if not re.match(constraint['pattern'], field_value):
                                    result.add_error(f"{context}: Field {field_tag} doesn't match required pattern")
            
        except Exception as e:
            result.add_warning(f"Compliance rule application error for {context}: {e}")
    
    def _load_compliance_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance rules for different regulations."""
        return {
            'MiFID_II': {
                'D': {  # New Order Single
                    'required_fields': [11, 55, 54, 60, 40, 38],  # Basic order fields
                    'field_constraints': {
                        54: ['1', '2'],  # Side
                        40: ['1', '2', '3', '4'],  # OrdType
                    }
                },
                '8': {  # Execution Report
                    'required_fields': [11, 17, 150, 39, 55, 54, 151, 14, 60],
                    'field_constraints': {
                        39: ['0', '1', '2', '4', '6', '8', 'A', 'E'],  # OrdStatus
                        150: ['0', '4', 'F'],  # ExecType
                    }
                }
            },
            'SEC': {
                # SEC-specific rules would go here
            },
            'CFTC': {
                # CFTC-specific rules would go here
            }
        }