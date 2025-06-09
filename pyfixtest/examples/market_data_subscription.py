#!/usr/bin/env python3
"""
Market data subscription example.
"""

import unittest
import time
from pyfixtest import BaseFIXTest, MarketDataTestHelper


class MarketDataTest(BaseFIXTest):
    """Example of market data testing."""
    
    def test_market_data_subscription(self):
        """Test subscribing to market data."""
        print("Testing market data subscription...")
        
        with self.fix_session() as engine:
            md_helper = MarketDataTestHelper(engine, self.message_factory)
            
            # Subscribe to market data for multiple symbols
            symbols = ['AAPL', 'MSFT', 'GOOGL']
            success, md_req_id = md_helper.subscribe_market_data(
                symbols=symbols,
                entry_types=['0', '1'],  # Bid, Offer
                market_depth=5
            )
            
            self.assertTrue(success, "Market data subscription failed")
            print(f"Subscribed to market data for {symbols}")
            
            # Wait for initial market data snapshot
            snapshot = engine.wait_for_message('W', timeout=10.0)
            if snapshot:
                self.assertMarketDataValid(snapshot)
                print("Received initial market data snapshot")
            
            # Simulate some price updates
            prices = [149.50, 149.55, 149.60, 149.45, 149.70]
            
            for i, price in enumerate(prices):
                md_helper.simulate_price_update(
                    symbol='AAPL',
                    bid_price=price,
                    bid_size=1000 + i * 100,
                    offer_price=price + 0.05,
                    offer_size=500 + i * 50
                )
                
                print(f"Simulated price update {i+1}: {price} @ {1000 + i * 100}")
                time.sleep(0.1)  # Small delay between updates
            
            # Check price history
            price_history = md_helper.get_price_history('AAPL')
            self.assertEqual(len(price_history), len(prices))
            
            # Get latest price
            latest_price = md_helper.get_latest_price('AAPL')
            self.assertIsNotNone(latest_price)
            self.assertEqual(latest_price['bid_price'], prices[-1])
            
            print("Market data testing completed successfully!")


if __name__ == '__main__':
    unittest.main(verbosity=2)