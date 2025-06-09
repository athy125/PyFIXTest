"""
Unit tests for validators.
"""

import unittest
import quickfix as fix
from pyfixtest.testing.validators import MessageValidator, OrderValidator, ValidationError
from pyfixtest.core.message_factory import MessageFactory


class TestMessageValidator(unittest.TestCase):
    """Test MessageValidator functionality."""
    
    def setUp(self):
        self.validator = MessageValidator()
        self.factory = MessageFactory()
    
    def test_validate_message_structure_valid(self):
        """Test validating valid message structure."""
        message = self.factory.create_new_order_single(
            symbol='AAPL',
            side='1',
            order_qty=100.0
        )
        
        # Should not raise exception
        result = self.validator.validate_message_structure(message)
        self.assertTrue(result)
    
    def test_validate_required_fields_new_order(self):
        """Test validating required fields for New Order Single."""
        message = self.factory.create_new_order_single(
            symbol='AAPL',
            side='1',
            order_qty=100.0
        )
        
        # Should not raise exception
        result = self.validator.validate_required_fields(message, 'D')
        self.assertTrue(result)
    
    def test_validate_field_values_with_rules(self):
        """Test validating field values against rules."""
        message = self.factory.create_new_order_single(
            symbol='AAPL',
            side='1',
            order_qty=100.0,
            order_type='2',
            price=150.0
        )
        
        rules = {
            54: ['1', '2'],  # Side must be Buy or Sell
            40: ['1', '2'],  # OrdType must be Market or Limit
            44: {'min': 0.01, 'max': 10000.0}  # Price range
        }
        
        # Should not raise exception
        result = self.validator.validate_field_values(message, rules)
        self.assertTrue(result)
    
    def test_validate_field_values_invalid(self):
        """Test validating invalid field values."""
        message = self.factory.create_new_order_single(
            symbol='AAPL',
            side='3',  # Invalid side
            order_qty=100.0
        )
        
        rules = {
            54: ['1', '2']  # Side must be Buy or Sell
        }
        
        # Should raise ValidationError
        with self.assertRaises(ValidationError):
            self.validator.validate_field_values(message, rules)
    
    def test_validate_order_message_new_order(self):
        """Test validating New Order Single message."""
        message = self.factory.create_new_order_single(
            symbol='AAPL',
            side='1',
            order_qty=100.0,
            order_type='2',
            price=150.0
        )
        
        # Should not raise exception
        result = self.validator.validate_order_message(message)
        self.assertTrue(result)
    
    def test_validate_order_message_invalid_quantity(self):
        """Test validating order with invalid quantity."""
        # Create message with negative quantity
        message = fix.Message()
        header = message.getHeader()
        header.setField(fix.MsgType('D'))
        message.setField(fix.ClOrdID('TEST_ORDER'))
        message.setField(fix.Symbol('AAPL'))
        message.setField(fix.Side('1'))
        message.setField(fix.TransactTime('20240101-12:00:00'))
        message.setField(fix.OrdType('1'))
        message.setField(fix.OrderQty(-100))  # Invalid negative quantity
        
        # Should raise ValidationError
        with self.assertRaises(ValidationError):
            self.validator.validate_order_message(message)


class TestOrderValidator(unittest.TestCase):
    """Test OrderValidator functionality."""
    
    def setUp(self):
        self.validator = OrderValidator()
        self.factory = MessageFactory()
    
    def test_validate_order_state_transition_valid(self):
        """Test valid order state transition."""
        # Should not raise exception
        result = self.validator.validate_order_state_transition(
            'ORDER_123', '0', '1'  # New -> Partially Filled
        )
        self.assertTrue(result)
    
    def test_validate_order_state_transition_invalid(self):
        """Test invalid order state transition."""
        # Should raise ValidationError
        with self.assertRaises(ValidationError):
            self.validator.validate_order_state_transition(
                'ORDER_123', '2', '0'  # Filled -> New (invalid)
            )
    
    def test_validate_execution_report_valid(self):
        """Test validating valid execution report."""
        exec_report = self.factory.create_execution_report(
            cl_ord_id='ORDER_123',
            exec_id='EXEC_456',
            exec_type='F',
            ord_status='2',
            symbol='AAPL',
            side='1',
            leaves_qty=0.0,
            cum_qty=100.0
        )
        
        # Should not raise exception
        result = self.validator.validate_execution_report(exec_report)
        self.assertTrue(result)
    
    def test_validate_fill_consistency_valid(self):
        """Test validating consistent fills."""
        fills = [
            {'qty': 50.0, 'price': 150.0, 'timestamp': 1000},
            {'qty': 30.0, 'price': 150.5, 'timestamp': 1001},
            {'qty': 20.0, 'price': 151.0, 'timestamp': 1002}
        ]
        
        # Should not raise exception
        result = self.validator.validate_fill_consistency('ORDER_123', fills)
        self.assertTrue(result)
    
    def test_validate_fill_consistency_invalid_quantity(self):
        """Test validating fills with invalid quantity."""
        fills = [
            {'qty': -50.0, 'price': 150.0, 'timestamp': 1000}  # Invalid negative qty
        ]
        
        # Should raise ValidationError
        with self.assertRaises(ValidationError):
            self.validator.validate_fill_consistency('ORDER_123', fills)
    
    def test_validate_risk_limits_valid(self):
        """Test validating order against risk limits."""
        order_data = {
            'symbol': 'AAPL',
            'quantity': 100.0,
            'price': 150.0,
            'side': '1',
            'order_type': 'LIMIT'
        }
        
        position_limits = {'AAPL': 1000.0}
        credit_limits = {'max_order_value': 20000.0}
        
        # Should not raise exception
        result = self.validator.validate_risk_limits(
            order_data, position_limits, credit_limits
        )
        self.assertTrue(result)
    
    def test_validate_risk_limits_position_exceeded(self):
        """Test validating order that exceeds position limits."""
        order_data = {
            'symbol': 'AAPL',
            'quantity': 1500.0,  # Exceeds limit
            'price': 150.0,
            'side': '1'
        }
        
        position_limits = {'AAPL': 1000.0}
        
        # Should raise ValidationError
        with self.assertRaises(ValidationError):
            self.validator.validate_risk_limits(order_data, position_limits)


if __name__ == '__main__':
    unittest.main()