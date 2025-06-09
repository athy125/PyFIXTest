
#!/usr/bin/env python3
"""
Simple order test example demonstrating basic PyFIXTest usage.
"""

import unittest
from pyfixtest import BaseFIXTest, OrderTestHelper


class SimpleOrderTest(BaseFIXTest):
    """Simple example of order testing."""
    
    def test_buy_order(self):
        """Test placing a simple buy order."""
        print("Testing simple buy order...")
        
        with self.fix_session() as engine:
            # Create order helper
            order_helper = OrderTestHelper(engine, self.message_factory)
            
            # Place market buy order
            success, cl_ord_id, exec_report = order_helper.place_market_order(
                symbol='AAPL',
                side='1',  # Buy
                quantity=100.0
            )
            
            # Verify order was placed successfully
            self.assertTrue(success, "Order placement failed")
            self.assertIsNotNone(exec_report, "No execution report received")
            
            # Check order status
            self.assertMessageType(exec_report, '8')  # Execution Report
            self.assertOrderStatus(exec_report, '0')  # New order status
            
            print(f"Order {cl_ord_id} placed successfully!")
    
    def test_limit_order_with_fill(self):
        """Test limit order with simulated fill."""
        print("Testing limit order with fill...")
        
        with self.fix_session() as engine:
            order_helper = OrderTestHelper(engine, self.message_factory)
            
            # Place limit order
            success, cl_ord_id, exec_report = order_helper.place_limit_order(
                symbol='MSFT',
                side='2',  # Sell
                quantity=50.0,
                price=300.0
            )
            
            self.assertTrue(success)
            self.assertOrderStatus(exec_report, '0')  # New
            
            # Simulate partial fill
            fill_report = order_helper.simulate_fill(
                cl_ord_id=cl_ord_id,
                fill_qty=25.0,
                fill_price=300.0
            )
            
            # Verify partial fill
            self.assertOrderPartiallyFilled(
                fill_report,
                expected_cum_qty=25.0,
                expected_leaves_qty=25.0
            )
            
            # Simulate remaining fill
            final_fill = order_helper.simulate_fill(
                cl_ord_id=cl_ord_id,
                fill_qty=25.0,
                fill_price=300.0
            )
            
            # Verify complete fill
            self.assertOrderFilled(final_fill, expected_qty=50.0)
            
            print(f"Order {cl_ord_id} fully filled!")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)