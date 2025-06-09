"""
Message factory for creating common FIX messages used in trading systems.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import quickfix as fix

from ..utils.time_utils import get_utc_timestamp, format_fix_time


class MessageFactory:
    """
    Factory class for creating common FIX messages.
    
    Provides convenient methods for creating:
    - Order messages (New Order Single, Order Cancel, etc.)
    - Market data messages
    - Administrative messages
    """
    
    def __init__(self, fix_version: str = "FIX.4.4"):
        self.fix_version = fix_version
        self.order_id_counter = 1000
        self.exec_id_counter = 2000
    
    def create_new_order_single(
        self,
        symbol: str,
        side: str,  # '1' = Buy, '2' = Sell
        order_qty: float,
        order_type: str = "2",  # '1' = Market, '2' = Limit
        price: Optional[float] = None,
        time_in_force: str = "0",  # '0' = Day
        cl_ord_id: Optional[str] = None,
        account: Optional[str] = None,
        **kwargs
    ) -> fix.Message:
        """
        Create New Order Single (MsgType=D) message.
        
        Args:
            symbol: Trading symbol
            side: Order side ('1'=Buy, '2'=Sell)
            order_qty: Order quantity
            order_type: Order type ('1'=Market, '2'=Limit)
            price: Limit price (required for limit orders)
            time_in_force: Time in force
            cl_ord_id: Client order ID
            account: Trading account
            **kwargs: Additional fields
            
        Returns:
            FIX NewOrderSingle message
        """
        message = fix.Message()
        header = message.getHeader()
        header.setField(fix.MsgType(fix.MsgType_NewOrderSingle))
        
        # Required fields
        message.setField(fix.ClOrdID(cl_ord_id or self._generate_order_id()))
        message.setField(fix.Symbol(symbol))
        message.setField(fix.Side(side))
        message.setField(fix.TransactTime(get_utc_timestamp()))
        message.setField(fix.OrdType(order_type))
        message.setField(fix.OrderQty(order_qty))
        message.setField(fix.TimeInForce(time_in_force))
        
        # Optional fields
        if price is not None:
            message.setField(fix.Price(price))
        
        if account:
            message.setField(fix.Account(account))
        
        # Add any additional fields
        for field_tag, field_value in kwargs.items():
            if isinstance(field_tag, int):
                message.setField(field_tag, str(field_value))
        
        return message
    
    def create_order_cancel_request(
        self,
        orig_cl_ord_id: str,
        symbol: str,
        side: str,
        cl_ord_id: Optional[str] = None,
        **kwargs
    ) -> fix.Message:
        """
        Create Order Cancel Request (MsgType=F) message.
        
        Args:
            orig_cl_ord_id: Original client order ID to cancel
            symbol: Trading symbol
            side: Order side
            cl_ord_id: New client order ID for cancel request
            **kwargs: Additional fields
            
        Returns:
            FIX OrderCancelRequest message
        """
        message = fix.Message()
        header = message.getHeader()
        header.setField(fix.MsgType(fix.MsgType_OrderCancelRequest))
        
        message.setField(fix.ClOrdID(cl_ord_id or self._generate_order_id()))
        message.setField(fix.OrigClOrdID(orig_cl_ord_id))
        message.setField(fix.Symbol(symbol))
        message.setField(fix.Side(side))
        message.setField(fix.TransactTime(get_utc_timestamp()))
        
        # Add any additional fields
        for field_tag, field_value in kwargs.items():
            if isinstance(field_tag, int):
                message.setField(field_tag, str(field_value))
        
        return message
    
    def create_execution_report(
        self,
        cl_ord_id: str,
        exec_id: str,
        exec_type: str,
        ord_status: str,
        symbol: str,
        side: str,
        leaves_qty: float,
        cum_qty: float = 0.0,
        avg_px: float = 0.0,
        last_qty: float = 0.0,
        last_px: float = 0.0,
        **kwargs
    ) -> fix.Message:
        """
        Create Execution Report (MsgType=8) message.
        
        Args:
            cl_ord_id: Client order ID
            exec_id: Execution ID
            exec_type: Execution type
            ord_status: Order status
            symbol: Trading symbol
            side: Order side
            leaves_qty: Remaining quantity
            cum_qty: Cumulative quantity
            avg_px: Average price
            last_qty: Last fill quantity
            last_px: Last fill price
            **kwargs: Additional fields
            
        Returns:
            FIX ExecutionReport message
        """
        message = fix.Message()
        header = message.getHeader()
        header.setField(fix.MsgType(fix.MsgType_ExecutionReport))
        
        message.setField(fix.ClOrdID(cl_ord_id))
        message.setField(fix.ExecID(exec_id))
        message.setField(fix.ExecType(exec_type))
        message.setField(fix.OrdStatus(ord_status))
        message.setField(fix.Symbol(symbol))
        message.setField(fix.Side(side))
        message.setField(fix.LeavesQty(leaves_qty))
        message.setField(fix.CumQty(cum_qty))
        message.setField(fix.AvgPx(avg_px))
        message.setField(fix.TransactTime(get_utc_timestamp()))
        
        if last_qty > 0:
            message.setField(fix.LastQty(last_qty))
        if last_px > 0:
            message.setField(fix.LastPx(last_px))
        
        # Add any additional fields
        for field_tag, field_value in kwargs.items():
            if isinstance(field_tag, int):
                message.setField(field_tag, str(field_value))
        
        return message
    
    def create_market_data_request(
        self,
        md_req_id: str,
        subscription_request_type: str,
        market_depth: int,
        symbols: list,
        md_entry_types: list,
        **kwargs
    ) -> fix.Message:
        """
        Create Market Data Request (MsgType=V) message.
        
        Args:
            md_req_id: Market data request ID
            subscription_request_type: Subscription type
            market_depth: Market depth
            symbols: List of symbols to subscribe
            md_entry_types: List of MD entry types
            **kwargs: Additional fields
            
        Returns:
            FIX MarketDataRequest message
        """
        message = fix.Message()
        header = message.getHeader()
        header.setField(fix.MsgType(fix.MsgType_MarketDataRequest))
        
        message.setField(fix.MDReqID(md_req_id))
        message.setField(fix.SubscriptionRequestType(subscription_request_type))
        message.setField(fix.MarketDepth(market_depth))
        
        # Add symbols group
        symbols_group = fix.Group(fix.NoRelatedSym(), fix.Symbol())
        for symbol in symbols:
            symbols_group.setField(fix.Symbol(symbol))
            message.addGroup(symbols_group)
        
        # Add MD entry types group
        entry_types_group = fix.Group(fix.NoMDEntryTypes(), fix.MDEntryType())
        for entry_type in md_entry_types:
            entry_types_group.setField(fix.MDEntryType(entry_type))
            message.addGroup(entry_types_group)
        
        # Add any additional fields
        for field_tag, field_value in kwargs.items():
            if isinstance(field_tag, int):
                message.setField(field_tag, str(field_value))
        
        return message
    
    def create_heartbeat(self, test_req_id: Optional[str] = None) -> fix.Message:
        """
        Create Heartbeat (MsgType=0) message.
        
        Args:
            test_req_id: Test request ID if responding to test request
            
        Returns:
            FIX Heartbeat message
        """
        message = fix.Message()
        header = message.getHeader()
        header.setField(fix.MsgType(fix.MsgType_Heartbeat))
        
        if test_req_id:
            message.setField(fix.TestReqID(test_req_id))
        
        return message
    
    def create_test_request(self, test_req_id: str) -> fix.Message:
        """
        Create Test Request (MsgType=1) message.
        
        Args:
            test_req_id: Test request ID
            
        Returns:
            FIX TestRequest message
        """
        message = fix.Message()
        header = message.getHeader()
        header.setField(fix.MsgType(fix.MsgType_TestRequest))
        message.setField(fix.TestReqID(test_req_id))
        
        return message
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        order_id = f"ORDER_{self.order_id_counter}"
        self.order_id_counter += 1
        return order_id
    
    def _generate_exec_id(self) -> str:
        """Generate unique execution ID."""
        exec_id = f"EXEC_{self.exec_id_counter}"
        self.exec_id_counter += 1
        return exec_id