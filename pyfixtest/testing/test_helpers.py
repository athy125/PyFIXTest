"""
Test helpers providing specialized functionality for different FIX testing scenarios.
"""

import time
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import quickfix as fix

from ..core.fix_engine import FIXEngine
from ..core.message_factory import MessageFactory
from ..utils.logging_config import get_logger


class OrderTestHelper:
    """
    Helper class for testing order lifecycle and management.
    
    Provides utilities for:
    - Order placement and tracking
    - Fill simulation
    - Order state validation
    - Complex order scenarios
    """
    
    def __init__(self, engine: FIXEngine, message_factory: MessageFactory):
        self.engine = engine
        self.message_factory = message_factory
        self.logger = get_logger(__name__)
        self.active_orders: Dict[str, Dict] = {}
        self.order_history: List[Dict] = []
    
    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        account: Optional[str] = None,
        timeout: float = 10.0
    ) -> Tuple[bool, Optional[str], Optional[Any]]:
        """
        Place market order and wait for acknowledgment.
        
        Args:
            symbol: Trading symbol
            side: Order side ('1'=Buy, '2'=Sell)
            quantity: Order quantity
            account: Trading account
            timeout: Timeout for acknowledgment
            
        Returns:
            Tuple of (success, client_order_id, execution_report)
        """
        cl_ord_id = f"MKT_{int(time.time() * 1000)}"
        
        order = self.message_factory.create_new_order_single(
            symbol=symbol,
            side=side,
            order_qty=quantity,
            order_type="1",  # Market order
            cl_ord_id=cl_ord_id,
            account=account
        )
        
        # Track order
        self.active_orders[cl_ord_id] = {
            'cl_ord_id': cl_ord_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': 'MARKET',
            'status': 'PENDING',
            'timestamp': time.time()
        }
        
        # Send order
        success = self.engine.send_message(order)
        if not success:
            return False, cl_ord_id, None
        
        # Wait for execution report
        exec_report = self.engine.wait_for_message('8', timeout)
        if exec_report:
            self._process_execution_report(exec_report)
        
        return success, cl_ord_id, exec_report
    
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        time_in_force: str = "0",
        account: Optional[str] = None,
        timeout: float = 10.0
    ) -> Tuple[bool, Optional[str], Optional[Any]]:
        """
        Place limit order and wait for acknowledgment.
        
        Args:
            symbol: Trading symbol
            side: Order side ('1'=Buy, '2'=Sell)
            quantity: Order quantity
            price: Limit price
            time_in_force: Time in force
            account: Trading account
            timeout: Timeout for acknowledgment
            
        Returns:
            Tuple of (success, client_order_id, execution_report)
        """
        cl_ord_id = f"LMT_{int(time.time() * 1000)}"
        
        order = self.message_factory.create_new_order_single(
            symbol=symbol,
            side=side,
            order_qty=quantity,
            order_type="2",  # Limit order
            price=price,
            time_in_force=time_in_force,
            cl_ord_id=cl_ord_id,
            account=account
        )
        
        # Track order
        self.active_orders[cl_ord_id] = {
            'cl_ord_id': cl_ord_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'order_type': 'LIMIT',
            'status': 'PENDING',
            'timestamp': time.time()
        }
        
        # Send order
        success = self.engine.send_message(order)
        if not success:
            return False, cl_ord_id, None
        
        # Wait for execution report
        exec_report = self.engine.wait_for_message('8', timeout)
        if exec_report:
            self._process_execution_report(exec_report)
        
        return success, cl_ord_id, exec_report
    
    def cancel_order(
        self,
        cl_ord_id: str,
        timeout: float = 10.0
    ) -> Tuple[bool, Optional[Any]]:
        """
        Cancel existing order.
        
        Args:
            cl_ord_id: Client order ID to cancel
            timeout: Timeout for cancel confirmation
            
        Returns:
            Tuple of (success, execution_report)
        """
        if cl_ord_id not in self.active_orders:
            self.logger.error(f"Order {cl_ord_id} not found in active orders")
            return False, None
        
        order_info = self.active_orders[cl_ord_id]
        
        cancel_request = self.message_factory.create_order_cancel_request(
            orig_cl_ord_id=cl_ord_id,
            symbol=order_info['symbol'],
            side=order_info['side']
        )
        
        # Send cancel request
        success = self.engine.send_message(cancel_request)
        if not success:
            return False, None
        
        # Wait for cancel confirmation
        exec_report = self.engine.wait_for_message('8', timeout)
        if exec_report:
            self._process_execution_report(exec_report)
        
        return success, exec_report
    
    def simulate_fill(
        self,
        cl_ord_id: str,
        fill_qty: float,
        fill_price: float,
        partial: bool = False
    ) -> Any:
        """
        Simulate order fill (for testing counterparty behavior).
        
        Args:
            cl_ord_id: Client order ID
            fill_qty: Fill quantity
            fill_price: Fill price
            partial: Whether this is a partial fill
            
        Returns:
            Generated execution report
        """
        if cl_ord_id not in self.active_orders:
            raise ValueError(f"Order {cl_ord_id} not found")
        
        order_info = self.active_orders[cl_ord_id]
        
        # Calculate execution details
        remaining_qty = order_info['quantity'] - order_info.get('filled_qty', 0)
        actual_fill_qty = min(fill_qty, remaining_qty)
        
        cum_qty = order_info.get('filled_qty', 0) + actual_fill_qty
        leaves_qty = order_info['quantity'] - cum_qty
        
        # Determine order status
        if leaves_qty == 0:
            ord_status = '2'  # Filled
            exec_type = 'F'   # Trade
        elif cum_qty > 0:
            ord_status = '1'  # Partially filled
            exec_type = 'F'   # Trade
        else:
            ord_status = '0'  # New
            exec_type = '0'   # New
        
        # Create execution report
        exec_report = self.message_factory.create_execution_report(
            cl_ord_id=cl_ord_id,
            exec_id=f"EXEC_{int(time.time() * 1000)}",
            exec_type=exec_type,
            ord_status=ord_status,
            symbol=order_info['symbol'],
            side=order_info['side'],
            leaves_qty=leaves_qty,
            cum_qty=cum_qty,
            avg_px=fill_price,
            last_qty=actual_fill_qty,
            last_px=fill_price
        )
        
        # Update order tracking
        order_info['filled_qty'] = cum_qty
        order_info['avg_price'] = fill_price  # Simplified
        order_info['status'] = 'FILLED' if leaves_qty == 0 else 'PARTIALLY_FILLED'
        
        return exec_report
    
    def get_order_status(self, cl_ord_id: str) -> Optional[Dict]:
        """Get current status of order."""
        return self.active_orders.get(cl_ord_id)
    
    def get_active_orders(self) -> Dict[str, Dict]:
        """Get all active orders."""
        return self.active_orders.copy()
    
    def _process_execution_report(self, exec_report):
        """Process received execution report and update order tracking."""
        try:
            cl_ord_id = exec_report.getField(11)  # ClOrdID
            exec_type = exec_report.getField(150)  # ExecType
            ord_status = exec_report.getField(39)  # OrdStatus
            
            if cl_ord_id in self.active_orders:
                order_info = self.active_orders[cl_ord_id]
                order_info['last_exec_type'] = exec_type
                order_info['last_ord_status'] = ord_status
                
                # Update quantities if available
                try:
                    leaves_qty = float(exec_report.getField(151))
                    cum_qty = float(exec_report.getField(14))
                    order_info['leaves_qty'] = leaves_qty
                    order_info['cum_qty'] = cum_qty
                except:
                    pass
                
                # Move to history if order is complete
                if ord_status in ['2', '4', '8']:  # Filled, Canceled, Rejected
                    self.order_history.append(order_info.copy())
                    if ord_status != '1':  # Keep partially filled orders active
                        del self.active_orders[cl_ord_id]
        
        except Exception as e:
            self.logger.error(f"Error processing execution report: {e}")


class MarketDataTestHelper:
    """
    Helper class for testing market data functionality.
    
    Provides utilities for:
    - Market data subscription
    - Price feed simulation
    - Market data validation
    """
    
    def __init__(self, engine: FIXEngine, message_factory: MessageFactory):
        self.engine = engine
        self.message_factory = message_factory
        self.logger = get_logger(__name__)
        self.subscriptions: Dict[str, Dict] = {}
        self.market_data: Dict[str, List] = {}
    
    def subscribe_market_data(
        self,
        symbols: List[str],
        entry_types: List[str] = ['0', '1'],  # Bid, Offer
        market_depth: int = 1,
        timeout: float = 10.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Subscribe to market data for symbols.
        
        Args:
            symbols: List of symbols to subscribe
            entry_types: List of MD entry types
            market_depth: Market depth level
            timeout: Timeout for subscription response
            
        Returns:
            Tuple of (success, md_req_id)
        """
        md_req_id = f"MD_{int(time.time() * 1000)}"
        
        md_request = self.message_factory.create_market_data_request(
            md_req_id=md_req_id,
            subscription_request_type='1',  # Snapshot + Updates
            market_depth=market_depth,
            symbols=symbols,
            md_entry_types=entry_types
        )
        
        # Track subscription
        self.subscriptions[md_req_id] = {
            'md_req_id': md_req_id,
            'symbols': symbols,
            'entry_types': entry_types,
            'market_depth': market_depth,
            'timestamp': time.time()
        }
        
        # Send subscription request
        success = self.engine.send_message(md_request)
        
        if success:
            # Wait for response
            response = self.engine.wait_for_message('W', timeout)  # Market Data Snapshot
            if response:
                self._process_market_data(response)
        
        return success, md_req_id
    
    def unsubscribe_market_data(self, md_req_id: str) -> bool:
        """
        Unsubscribe from market data.
        
        Args:
            md_req_id: Market data request ID to unsubscribe
            
        Returns:
            bool: Success status
        """
        if md_req_id not in self.subscriptions:
            return False
        
        subscription = self.subscriptions[md_req_id]
        
        md_request = self.message_factory.create_market_data_request(
            md_req_id=md_req_id,
            subscription_request_type='2',  # Disable previous snapshot
            market_depth=subscription['market_depth'],
            symbols=subscription['symbols'],
            md_entry_types=subscription['entry_types']
        )
        
        success = self.engine.send_message(md_request)
        
        if success:
            del self.subscriptions[md_req_id]
        
        return success
    
    def simulate_price_update(
        self,
        symbol: str,
        bid_price: float,
        bid_size: float,
        offer_price: float,
        offer_size: float
    ):
        """
        Simulate market data price update.
        
        Args:
            symbol: Trading symbol
            bid_price: Bid price
            bid_size: Bid size
            offer_price: Offer price
            offer_size: Offer size
        """
        # Store simulated market data
        if symbol not in self.market_data:
            self.market_data[symbol] = []
        
        self.market_data[symbol].append({
            'timestamp': time.time(),
            'bid_price': bid_price,
            'bid_size': bid_size,
            'offer_price': offer_price,
            'offer_size': offer_size
        })
        
        self.logger.info(f"Simulated price update for {symbol}: "
                        f"Bid {bid_price}@{bid_size}, Offer {offer_price}@{offer_size}")
    
    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """Get latest price for symbol."""
        if symbol in self.market_data and self.market_data[symbol]:
            return self.market_data[symbol][-1]
        return None
    
    def get_price_history(self, symbol: str) -> List[Dict]:
        """Get price history for symbol."""
        return self.market_data.get(symbol, [])
    
    def _process_market_data(self, md_message):
        """Process received market data message."""
        try:
            # This is a simplified processor
            # Real implementation would parse all MD entries
            symbol = md_message.getField(55)  # Symbol
            
            self.logger.info(f"Received market data for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")


class SessionTestHelper:
    """
    Helper class for testing FIX session management.
    
    Provides utilities for:
    - Session lifecycle testing
    - Connection management
    - Heartbeat handling
    - Sequence number management
    """
    
    def __init__(self, engine: FIXEngine, message_factory: MessageFactory):
        self.engine = engine
        self.message_factory = message_factory
        self.logger = get_logger(__name__)
    
    def test_heartbeat_mechanism(self, interval: float = 30.0) -> bool:
        """
        Test heartbeat mechanism by sending test request.
        
        Args:
            interval: Heartbeat interval
            
        Returns:
            bool: True if heartbeat mechanism works correctly
        """
        test_req_id = f"TEST_{int(time.time() * 1000)}"
        
        # Send test request
        test_request = self.message_factory.create_test_request(test_req_id)
        success = self.engine.send_message(test_request)
        
        if not success:
            return False
        
        # Wait for heartbeat response
        heartbeat = self.engine.wait_for_message('0', timeout=10.0)
        
        if heartbeat:
            try:
                received_test_req_id = heartbeat.getField(112)  # TestReqID
                return received_test_req_id == test_req_id
            except:
                return False
        
        return False
    
    def simulate_sequence_gap(self) -> bool:
        """
        Simulate sequence number gap to test resend logic.
        
        Returns:
            bool: Success status
        """
        # This would require more complex session manipulation
        # Placeholder for sequence gap testing
        self.logger.info("Simulating sequence gap (placeholder)")
        return True
    
    def test_session_recovery(self) -> bool:
        """
        Test session recovery after disconnect.
        
        Returns:
            bool: True if session recovers successfully
        """
        # This would involve disconnecting and reconnecting
        # Placeholder for session recovery testing
        self.logger.info("Testing session recovery (placeholder)")
        return True