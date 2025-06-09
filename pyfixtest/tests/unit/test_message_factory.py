"""
Unit tests for MessageFactory.
"""

import unittest
import quickfix as fix
from pyfixtest.core.message_factory import MessageFactory


class TestMessageFactory(unittest.TestCase):
    """Test MessageFactory functionality."""
    
    def setUp(self):
        self.factory = MessageFactory()
    
    def test_create_new_order_single(self):
        """Test creating New Order Single message."""
        order = self.factory.create_new_order_single(
            symbol='AAPL',
            side='1',
            order_qty=100.0,
            order_type='2',
            price=150.0
        )
        
        # Check message type
        msg_type = order.getHeader().getField(35)
        self.assertEqual(msg_type, 'D')
        
        # Check required fields
        self.assertEqual(order.getField(55), 'AAPL')  # Symbol
        self.assertEqual(order.getField(54), '1')     # Side
        self.assertEqual(order.getField(38), '100')   # OrderQty
        self.assertEqual(order.getField(40), '2')     # OrdType
        self.assertEqual(order.getField(44), '150')   # Price
    
    def test_create_order_cancel_request(self):
        """Test creating Order Cancel Request message."""
        cancel = self.factory.create_order_cancel_request(
            orig_cl_ord_id='ORDER_123',
            symbol='MSFT',
            side='2'
        )
        
        # Check message type
        msg_type = cancel.getHeader().getField(35)
        self.assertEqual(msg_type, 'F')
        
        # Check required fields
        self.assertEqual(cancel.getField(41), 'ORDER_123')  # OrigClOrdID
        self.assertEqual(cancel.getField(55), 'MSFT')       # Symbol
        self.assertEqual(cancel.getField(54), '2')          # Side
    
    def test_create_execution_report(self):
        """Test creating Execution Report message."""
        exec_report = self.factory.create_execution_report(
            cl_ord_id='ORDER_456',
            exec_id='EXEC_789',
            exec_type='F',
            ord_status='2',
            symbol='GOOGL',
            side='1',
            leaves_qty=0.0,
            cum_qty=50.0,
            avg_px=2800.0
        )
        
        # Check message type
        msg_type = exec_report.getHeader().getField(35)
        self.assertEqual(msg_type, '8')
        
        # Check required fields
        self.assertEqual(exec_report.getField(11), 'ORDER_456')  # ClOrdID
        self.assertEqual(exec_report.getField(17), 'EXEC_789')   # ExecID
        self.assertEqual(exec_report.getField(150), 'F')         # ExecType
        self.assertEqual(exec_report.getField(39), '2')          # OrdStatus
    
    def test_create_heartbeat(self):
        """Test creating Heartbeat message."""
        heartbeat = self.factory.create_heartbeat()
        
        # Check message type
        msg_type = heartbeat.getHeader().getField(35)
        self.assertEqual(msg_type, '0')
    
    def test_create_test_request(self):
        """Test creating Test Request message."""
        test_req = self.factory.create_test_request('TEST_REQ_123')
        
        # Check message type
        msg_type = test_req.getHeader().getField(35)
        self.assertEqual(msg_type, '1')
        
        # Check TestReqID field
        self.assertEqual(test_req.getField(112), 'TEST_REQ_123')


if __name__ == '__main__':
    unittest.main()