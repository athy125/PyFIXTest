"""
Integration tests for complete order lifecycle.
"""

import unittest
from pyfixtest import BaseFIXTest, OrderTestHelper


class TestOrderLifecycle(BaseFIXTest):
    """Integration tests for order lifecycle scenarios."""
    
    def test_complete_market_order_lifecycle(self):
        """Test complete market order from placement to fill."""
        with self.fix_session() as engine:
            order_helper = OrderTestHelper(engine, self.message_factory)
            
            # Place market order
            success, cl_ord_id, exec_report = order_helper.place_market_order(
                symbol='AAPL',
                side='1',  # Buy
                quantity=100.0,
                account='TEST_ACCOUNT'
            )
            
            # Verify order placement
            self.assertTrue(success)
            self.assertIsNotNone(exec_report)
            self.assertOrderStatus(exec_report, '0')  # New
            
            # Verify order tracking
            order_status = order_helper.get_order_status(cl_ord_id)
            self.assertIsNotNone(order_status)
            self.assertEqual(order_status['status'], 'PENDING')
            
            # Simulate immediate fill (market order)
            fill_report = order_helper.simulate_fill(
                cl_ord_id=cl_ord_id,
                fill_qty=100.0,
                fill_price=150.0
            )
            
            # Verify fill
            self.assertOrderFilled(fill_report, expected_qty=100.0)
            self.assertExecutionPrice(fill_report, 150.0)
            
            # Verify final order status
            final_status = order_helper.get_order_status(cl_ord_id)
            self.assertEqual(final_status['status'], 'FILLED')
    
    def test_limit_order_partial_fill_workflow(self):
        """Test limit order with partial fills."""
        with self.fix_session() as engine:
            order_helper = OrderTestHelper(engine, self.message_factory)
            
            # Place limit order
            success, cl_ord_id, exec_report = order_helper.place_limit_order(
                symbol='MSFT',
                side='2',  # Sell
                quantity=200.0,
                price=300.0,
                time_in_force='0'  # Day order
            )
            
            # Verify order placement
            self.assertTrue(success)
            self.assertOrderStatus(exec_report, '0')  # New
            
            # Simulate first partial fill
            fill1 = order_helper.simulate_fill(
                cl_ord_id=cl_ord_id,
                fill_qty=75.0,
                fill_price=300.0
            )
            
            self.assertOrderPartiallyFilled(
                fill1,
                expected_cum_qty=75.0,
                expected_leaves_qty=125.0
            )
            
            # Simulate second partial fill
            fill2 = order_helper.simulate_fill(
                cl_ord_id=cl_ord_id,
                fill_qty=50.0,
                fill_price=300.5
            )
            
            self.assertOrderPartiallyFilled(
                fill2,
                expected_cum_qty=125.0,
                expected_leaves_qty=75.0
            )
            
            # Simulate final fill
            fill3 = order_helper.simulate_fill(
                cl_ord_id=cl_ord_id,
                fill_qty=75.0,
                fill_price=301.0
            )
            
            self.assertOrderFilled(fill3, expected_qty=200.0)
            
            # Verify fill history
            fills = order_helper.get_order_fills(cl_ord_id)
            self.assertEqual(len(fills), 3)
    
    def test_order_cancel_workflow(self):
        """Test order cancellation workflow."""
        with self.fix_session() as engine:
            order_helper = OrderTestHelper(engine, self.message_factory)
            
            # Place limit order
            success, cl_ord_id, exec_report = order_helper.place_limit_order(
                symbol='GOOGL',
                side='1',  # Buy
                quantity=25.0,
                price=2800.0
            )
            
            self.assertTrue(success)
            self.assertOrderStatus(exec_report, '0')  # New
            
            # Cancel the order
            cancel_success, cancel_report = order_helper.cancel_order(cl_ord_id)
            
            self.assertTrue(cancel_success)
            self.assertIsNotNone(cancel_report)
            self.assertOrderCanceled(cancel_report)
            
            # Verify order is no longer active
            order_status = order_helper.get_order_status(cl_ord_id)
            self.assertIn(order_status['status'], ['CANCELLED', 'CANCELED'])
    
    def test_order_replace_workflow(self):
        """Test order cancel/replace workflow."""
        with self.fix_session() as engine:
            order_helper = OrderTestHelper(engine, self.message_factory)
            
            # Place initial limit order
            success, cl_ord_id, exec_report = order_helper.place_limit_order(
                symbol='AAPL',
                side='1',  # Buy
                quantity=100.0,
                price=149.0
            )
            
            self.assertTrue(success)
            self.assertOrderStatus(exec_report, '0')  # New
            
            # Create replace request (modify price)
            replace_order = self.message_factory.create_order_cancel_replace_request(
                orig_cl_ord_id=cl_ord_id,
                symbol='AAPL',
                side='1',
                order_qty=100.0,
                price=149.5,  # New price
                cl_ord_id=f"REPLACE_{cl_ord_id}"
            )
            
            # Send replace request
            replace_success = engine.send_message(replace_order)
            self.assertTrue(replace_success)
            
            # Wait for replace response
            replace_response = engine.wait_for_message('8', timeout=10.0)
            self.assertIsNotNone(replace_response)
            
            # Verify the replace was accepted
            # (Implementation would depend on counterparty behavior)
    
    def test_multiple_orders_workflow(self):
        """Test managing multiple orders simultaneously."""
        with self.fix_session() as engine:
            order_helper = OrderTestHelper(engine, self.message_factory)
            
            orders = []
            symbols = ['AAPL', 'MSFT', 'GOOGL']
            
            # Place multiple orders
            for i, symbol in enumerate(symbols):
                success, cl_ord_id, exec_report = order_helper.place_limit_order(
                    symbol=symbol,
                    side='1',  # Buy
                    quantity=100.0 + i * 50,
                    price=150.0 + i * 50
                )
                
                self.assertTrue(success)
                orders.append(cl_ord_id)
            
            # Verify all orders are active
            active_orders = order_helper.get_active_orders()
            self.assertEqual(len(active_orders), 3)
            
            # Fill orders one by one
            for i, cl_ord_id in enumerate(orders):
                fill_report = order_helper.simulate_fill(
                    cl_ord_id=cl_ord_id,
                    fill_qty=100.0 + i * 50,
                    fill_price=150.0 + i * 50
                )
                
                self.assertOrderFilled(fill_report)
            
            # Verify all orders are filled
            for cl_ord_id in orders:
                order_status = order_helper.get_order_status(cl_ord_id)
                self.assertEqual(order_status['status'], 'FILLED')


if __name__ == '__main__':
    unittest.main()