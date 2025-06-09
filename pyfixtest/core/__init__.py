"""
FIX Trading System Core Components

This module provides the core components and engines for the FIX trading system,
including the main FIX engine, session management, protocol handling, and
order management capabilities.

Core Components:
    - FIX Engine: Main engine for FIX protocol communication
    - Session Manager: FIX session lifecycle management
    - Protocol Handler: Message processing and routing
    - Order Manager: Order lifecycle and state management
    - Market Data Manager: Market data subscription and handling
    - Risk Manager: Real-time risk management and controls
    - Message Builder: FIX message construction utilities
    - Connection Manager: Network connection management
"""

__version__ = "1.0.0"
__author__ = "FIX Trading System Team"

# Core engine components
from .fix_engine import (
    FIXEngine,
    FIXEngineState,
    FIXApplication,
    EngineConfiguration,
    EngineStatistics
)

# Session management
from .session_manager import (
    SessionManager,
    SessionState,
    SessionInfo,
    SessionFactory,
    SessionEventType
)

# Protocol handling
from .protocol_handler import (
    ProtocolHandler,
    MessageContext,
    MessageDirection,
    MessagePriority,
    ProcessingResult,
    MessageRouter
)

# Order management
from .order_manager import (
    OrderManager,
    Order,
    OrderState,
    OrderType,
    OrderSide,
    ExecutionReport,
    OrderBook,
    Position
)

# Market data management
from .market_data_manager import (
    MarketDataManager,
    MarketDataSubscription,
    MarketDataSnapshot,
    MarketDataUpdate,
    Quote,
    Trade,
    Level2Data
)

# Risk management
from .risk_manager import (
    RiskManager,
    RiskCheck,
    RiskRule,
    RiskViolation,
    PositionLimit,
    CreditLimit,
    RiskMetrics
)

# Message utilities
from .message_builder import (
    MessageBuilder,
    OrderMessageBuilder,
    MarketDataMessageBuilder,
    AdminMessageBuilder,
    MessageTemplate
)

# Connection management
from .connection_manager import (
    ConnectionManager,
    Connection,
    ConnectionState,
    ConnectionPool,
    NetworkManager,
    HeartbeatManager
)

# Export all public classes and functions
__all__ = [
    # FIX Engine
    'FIXEngine',
    'FIXEngineState',
    'FIXApplication',
    'EngineConfiguration',
    'EngineStatistics',
    
    # Session Management
    'SessionManager',
    'SessionState',
    'SessionInfo',
    'SessionFactory',
    'SessionEventType',
    
    # Protocol Handling
    'ProtocolHandler',
    'MessageContext',
    'MessageDirection',
    'MessagePriority',
    'ProcessingResult',
    'MessageRouter',
    
    # Order Management
    'OrderManager',
    'Order',
    'OrderState',
    'OrderType',
    'OrderSide',
    'ExecutionReport',
    'OrderBook',
    'Position',
    
    # Market Data Management
    'MarketDataManager',
    'MarketDataSubscription',
    'MarketDataSnapshot',
    'MarketDataUpdate',
    'Quote',
    'Trade',
    'Level2Data',
    
    # Risk Management
    'RiskManager',
    'RiskCheck',
    'RiskRule',
    'RiskViolation',
    'PositionLimit',
    'CreditLimit',
    'RiskMetrics',
    
    # Message Building
    'MessageBuilder',
    'OrderMessageBuilder',
    'MarketDataMessageBuilder',
    'AdminMessageBuilder',
    'MessageTemplate',
    
    # Connection Management
    'ConnectionManager',
    'Connection',
    'ConnectionState',
    'ConnectionPool',
    'NetworkManager',
    'HeartbeatManager',
    
    # Factory functions
    'create_fix_engine',
    'create_session_manager',
    'create_order_manager',
    'create_market_data_manager',
    'create_risk_manager',
    'create_trading_system',
    
    # Utility functions
    'initialize_core_system',
    'shutdown_core_system',
    'get_system_status',
    'get_system_metrics',
    'validate_system_health',
]

# System-wide constants
FIX_VERSIONS = {
    'FIX.4.0': '4.0',
    'FIX.4.1': '4.1',
    'FIX.4.2': '4.2',
    'FIX.4.3': '4.3',
    'FIX.4.4': '4.4',
    'FIX.5.0': '5.0',
    'FIXT.1.1': '1.1'
}

DEFAULT_FIX_VERSION = 'FIX.4.4'

# Message type constants
MESSAGE_TYPES = {
    # Admin Messages
    'HEARTBEAT': '0',
    'TEST_REQUEST': '1',
    'RESEND_REQUEST': '2',
    'REJECT': '3',
    'SEQUENCE_RESET': '4',
    'LOGOUT': '5',
    'LOGON': 'A',
    
    # Application Messages
    'NEW_ORDER_SINGLE': 'D',
    'ORDER_CANCEL_REQUEST': 'F',
    'ORDER_CANCEL_REPLACE_REQUEST': 'G',
    'EXECUTION_REPORT': '8',
    'ORDER_CANCEL_REJECT': '9',
    'BUSINESS_MESSAGE_REJECT': 'j',
    
    # Market Data Messages
    'MARKET_DATA_REQUEST': 'V',
    'MARKET_DATA_SNAPSHOT': 'W',
    'MARKET_DATA_INCREMENTAL_REFRESH': 'X',
    'MARKET_DATA_REQUEST_REJECT': 'Y',
    'QUOTE_REQUEST': 'R',
    'QUOTE': 'S',
    'QUOTE_CANCEL': 'Z',
    'QUOTE_STATUS_REQUEST': 'a',
    'QUOTE_ACKNOWLEDGMENT': 'b'
}

# Field tag constants (commonly used)
FIELD_TAGS = {
    'BEGIN_STRING': 8,
    'BODY_LENGTH': 9,
    'MSG_TYPE': 35,
    'SENDER_COMP_ID': 49,
    'TARGET_COMP_ID': 56,
    'MSG_SEQ_NUM': 34,
    'SENDING_TIME': 52,
    'CHECKSUM': 10,
    
    # Order fields
    'CL_ORD_ID': 11,
    'ORIG_CL_ORD_ID': 41,
    'ORDER_ID': 37,
    'EXEC_ID': 17,
    'SYMBOL': 55,
    'SIDE': 54,
    'ORDER_QTY': 38,
    'ORD_TYPE': 40,
    'PRICE': 44,
    'TIME_IN_FORCE': 59,
    'TRANSACT_TIME': 60,
    'ORD_STATUS': 39,
    'EXEC_TYPE': 150,
    'LEAVES_QTY': 151,
    'CUM_QTY': 14,
    'AVG_PX': 6,
    'LAST_QTY': 32,
    'LAST_PX': 31,
    
    # Market data fields
    'MD_REQ_ID': 262,
    'SUBSCRIPTION_REQUEST_TYPE': 263,
    'MARKET_DEPTH': 264,
    'MD_UPDATE_TYPE': 265,
    'NO_MD_ENTRIES': 268,
    'MD_ENTRY_TYPE': 269,
    'MD_ENTRY_PX': 270,
    'MD_ENTRY_SIZE': 271,
    'MD_ENTRY_TIME': 273,
    
    # Common fields
    'TEXT': 58,
    'HEARTBT_INT': 108,
    'TEST_REQ_ID': 112,
    'BEGIN_SEQ_NO': 7,
    'END_SEQ_NO': 16,
    'NEW_SEQ_NO': 36,
    'GAP_FILL_FLAG': 123,
    'POSS_DUP_FLAG': 43,
    'POSS_RESEND': 97
}

# Order side constants
ORDER_SIDES = {
    'BUY': '1',
    'SELL': '2',
    'BUY_MINUS': '3',
    'SELL_PLUS': '4',
    'SELL_SHORT': '5',
    'SELL_SHORT_EXEMPT': '6',
    'UNDISCLOSED': '7',
    'CROSS': '8'
}

# Order type constants
ORDER_TYPES = {
    'MARKET': '1',
    'LIMIT': '2',
    'STOP': '3',
    'STOP_LIMIT': '4',
    'MARKET_ON_CLOSE': '5',
    'WITH_OR_WITHOUT': '6',
    'LIMIT_OR_BETTER': '7',
    'LIMIT_WITH_OR_WITHOUT': '8',
    'ON_BASIS': '9',
    'ON_CLOSE': 'A',
    'LIMIT_ON_CLOSE': 'B',
    'FOREX_MARKET': 'C',
    'PREVIOUSLY_QUOTED': 'D',
    'PREVIOUSLY_INDICATED': 'E',
    'FOREX_LIMIT': 'F',
    'FOREX_SWAP': 'G',
    'FOREX_PREVIOUSLY_QUOTED': 'H',
    'FUNARI': 'I',
    'MARKET_IF_TOUCHED': 'J',
    'MARKET_WITH_LEFTOVER_AS_LIMIT': 'K',
    'PREVIOUS_FUND_VALUATION_POINT': 'L',
    'NEXT_FUND_VALUATION_POINT': 'M',
    'PEGGED': 'P'
}

# Order status constants
ORDER_STATUSES = {
    'NEW': '0',
    'PARTIALLY_FILLED': '1',
    'FILLED': '2',
    'DONE_FOR_DAY': '3',
    'CANCELED': '4',
    'REPLACED': '5',
    'PENDING_CANCEL': '6',
    'STOPPED': '7',
    'REJECTED': '8',
    'SUSPENDED': '9',
    'PENDING_NEW': 'A',
    'CALCULATED': 'B',
    'EXPIRED': 'C',
    'ACCEPTED_FOR_BIDDING': 'D',
    'PENDING_REPLACE': 'E'
}

# Execution type constants
EXEC_TYPES = {
    'NEW': '0',
    'PARTIAL_FILL': '1',
    'FILL': '2',
    'DONE_FOR_DAY': '3',
    'CANCELED': '4',
    'REPLACE': '5',
    'PENDING_CANCEL': '6',
    'STOPPED': '7',
    'REJECTED': '8',
    'SUSPENDED': '9',
    'PENDING_NEW': 'A',
    'CALCULATED': 'B',
    'EXPIRED': 'C',
    'RESTATED': 'D',
    'PENDING_REPLACE': 'E',
    'TRADE': 'F',
    'TRADE_CORRECT': 'G',
    'TRADE_CANCEL': 'H',
    'ORDER_STATUS': 'I'
}

# Time in force constants
TIME_IN_FORCE = {
    'DAY': '0',
    'GOOD_TILL_CANCEL': '1',
    'AT_THE_OPENING': '2',
    'IMMEDIATE_OR_CANCEL': '3',
    'FILL_OR_KILL': '4',
    'GOOD_TILL_CROSSING': '5',
    'GOOD_TILL_DATE': '6',
    'AT_THE_CLOSE': '7'
}

# Market data entry types
MD_ENTRY_TYPES = {
    'BID': '0',
    'OFFER': '1',
    'TRADE': '2',
    'INDEX_VALUE': '3',
    'OPENING_PRICE': '4',
    'CLOSING_PRICE': '5',
    'SETTLEMENT_PRICE': '6',
    'TRADING_SESSION_HIGH_PRICE': '7',
    'TRADING_SESSION_LOW_PRICE': '8',
    'TRADING_SESSION_VWAP_PRICE': '9',
    'IMBALANCE': 'A',
    'TRADE_VOLUME': 'B',
    'OPEN_INTEREST': 'C'
}

# System status tracking
_system_components = {}
_system_initialized = False
_system_metrics = {
    'start_time': None,
    'total_messages_processed': 0,
    'total_orders_processed': 0,
    'total_sessions': 0,
    'active_connections': 0,
    'errors_count': 0
}

def create_fix_engine(config=None, **kwargs):
    """
    Create and configure a FIX engine instance.
    
    Args:
        config: FIX configuration object
        **kwargs: Additional configuration parameters
        
    Returns:
        FIXEngine: Configured FIX engine instance
    """
    from ..config.fix_config import FIXConfig
    
    if config is None:
        config = FIXConfig()
        if kwargs:
            config.load_from_dict(kwargs)
    
    engine = FIXEngine(config)
    
    # Register with system
    _register_component('fix_engine', engine)
    
    return engine

def create_session_manager(config=None):
    """
    Create a session manager instance.
    
    Args:
        config: Configuration object
        
    Returns:
        SessionManager: Session manager instance
    """
    from ..config.fix_config import FIXConfig
    
    if config is None:
        config = FIXConfig()
    
    session_manager = SessionManager(config)
    
    # Register with system
    _register_component('session_manager', session_manager)
    
    return session_manager

def create_order_manager(session_manager=None, risk_manager=None):
    """
    Create an order manager instance.
    
    Args:
        session_manager: Session manager instance
        risk_manager: Risk manager instance
        
    Returns:
        OrderManager: Order manager instance
    """
    if session_manager is None:
        session_manager = create_session_manager()
    
    order_manager = OrderManager(session_manager, risk_manager)
    
    # Register with system
    _register_component('order_manager', order_manager)
    
    return order_manager

def create_market_data_manager(session_manager=None):
    """
    Create a market data manager instance.
    
    Args:
        session_manager: Session manager instance
        
    Returns:
        MarketDataManager: Market data manager instance
    """
    if session_manager is None:
        session_manager = create_session_manager()
    
    md_manager = MarketDataManager(session_manager)
    
    # Register with system
    _register_component('market_data_manager', md_manager)
    
    return md_manager

def create_risk_manager(config=None):
    """
    Create a risk manager instance.
    
    Args:
        config: Risk configuration
        
    Returns:
        RiskManager: Risk manager instance
    """
    risk_manager = RiskManager(config)
    
    # Register with system
    _register_component('risk_manager', risk_manager)
    
    return risk_manager

def create_trading_system(config=None, **kwargs):
    """
    Create a complete trading system with all components.
    
    Args:
        config: System configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        dict: Dictionary of system components
    """
    from ..config.fix_config import FIXConfig
    
    if config is None:
        config = FIXConfig()
        if kwargs:
            config.load_from_dict(kwargs)
    
    # Create core components
    session_manager = create_session_manager(config)
    risk_manager = create_risk_manager(config)
    order_manager = create_order_manager(session_manager, risk_manager)
    market_data_manager = create_market_data_manager(session_manager)
    fix_engine = create_fix_engine(config)
    
    # Create protocol handler
    protocol_handler = ProtocolHandler(session_manager)
    _register_component('protocol_handler', protocol_handler)
    
    # Create connection manager
    connection_manager = ConnectionManager(config)
    _register_component('connection_manager', connection_manager)
    
    # Wire components together
    fix_engine.set_session_manager(session_manager)
    fix_engine.set_protocol_handler(protocol_handler)
    fix_engine.set_order_manager(order_manager)
    fix_engine.set_market_data_manager(market_data_manager)
    fix_engine.set_risk_manager(risk_manager)
    
    system_components = {
        'fix_engine': fix_engine,
        'session_manager': session_manager,
        'protocol_handler': protocol_handler,
        'order_manager': order_manager,
        'market_data_manager': market_data_manager,
        'risk_manager': risk_manager,
        'connection_manager': connection_manager,
        'config': config
    }
    
    return system_components

def initialize_core_system(config=None, **kwargs):
    """
    Initialize the core FIX trading system.
    
    Args:
        config: System configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        bool: True if initialization successful
    """
    global _system_initialized, _system_metrics
    
    if _system_initialized:
        return True
    
    try:
        from datetime import datetime, timezone
        
        # Create trading system
        system = create_trading_system(config, **kwargs)
        
        # Initialize components
        for name, component in system.items():
            if hasattr(component, 'initialize'):
                component.initialize()
        
        # Start core services
        if 'session_manager' in system:
            system['session_manager'].start_monitoring()
        
        if 'protocol_handler' in system:
            system['protocol_handler'].start_processing()
        
        # Update system metrics
        _system_metrics['start_time'] = datetime.now(timezone.utc)
        _system_metrics['total_sessions'] = len(_system_components.get('sessions', []))
        
        _system_initialized = True
        
        from ..utils.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Core FIX trading system initialized successfully")
        
        return True
        
    except Exception as e:
        from ..utils.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Failed to initialize core system: {e}")
        return False

def shutdown_core_system():
    """Shutdown the core FIX trading system."""
    global _system_initialized, _system_components
    
    if not _system_initialized:
        return
    
    try:
        from ..utils.logging_config import get_logger
        logger = get_logger(__name__)
        
        # Stop components in reverse order
        component_order = [
            'protocol_handler',
            'session_manager',
            'order_manager',
            'market_data_manager',
            'risk_manager',
            'connection_manager',
            'fix_engine'
        ]
        
        for component_name in component_order:
            if component_name in _system_components:
                component = _system_components[component_name]
                try:
                    if hasattr(component, 'stop'):
                        component.stop()
                    elif hasattr(component, 'shutdown'):
                        component.shutdown()
                    elif hasattr(component, 'cleanup'):
                        component.cleanup()
                except Exception as e:
                    logger.warning(f"Error stopping {component_name}: {e}")
        
        # Clear components
        _system_components.clear()
        _system_initialized = False
        
        logger.info("Core FIX trading system shutdown complete")
        
    except Exception as e:
        from ..utils.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Error during system shutdown: {e}")

def get_system_status():
    """
    Get current system status.
    
    Returns:
        dict: System status information
    """
    status = {
        'initialized': _system_initialized,
        'components': list(_system_components.keys()),
        'component_count': len(_system_components),
        'metrics': _system_metrics.copy()
    }
    
    # Add component-specific status
    for name, component in _system_components.items():
        if hasattr(component, 'get_status'):
            status[f'{name}_status'] = component.get_status()
        elif hasattr(component, 'is_running'):
            status[f'{name}_running'] = component.is_running()
    
    return status

def get_system_metrics():
    """
    Get system performance metrics.
    
    Returns:
        dict: System metrics
    """
    metrics = _system_metrics.copy()
    
    # Collect metrics from components
    for name, component in _system_components.items():
        if hasattr(component, 'get_metrics'):
            component_metrics = component.get_metrics()
            metrics[f'{name}_metrics'] = component_metrics
        elif hasattr(component, 'get_statistics'):
            component_stats = component.get_statistics()
            metrics[f'{name}_statistics'] = component_stats
    
    return metrics

def validate_system_health():
    """
    Validate system health and return health report.
    
    Returns:
        dict: Health validation report
    """
    health_report = {
        'overall_health': 'HEALTHY',
        'issues': [],
        'warnings': [],
        'component_health': {}
    }
    
    if not _system_initialized:
        health_report['overall_health'] = 'NOT_INITIALIZED'
        health_report['issues'].append('System not initialized')
        return health_report
    
    # Check each component
    for name, component in _system_components.items():
        component_health = 'HEALTHY'
        component_issues = []
        
        try:
            # Check if component has health check method
            if hasattr(component, 'health_check'):
                health_result = component.health_check()
                if not health_result.get('healthy', True):
                    component_health = 'UNHEALTHY'
                    component_issues.extend(health_result.get('issues', []))
            
            # Check if component is running
            elif hasattr(component, 'is_running'):
                if not component.is_running():
                    component_health = 'STOPPED'
                    component_issues.append('Component not running')
            
            # Check for error conditions
            if hasattr(component, 'get_error_count'):
                error_count = component.get_error_count()
                if error_count > 100:  # Threshold for too many errors
                    component_health = 'DEGRADED'
                    component_issues.append(f'High error count: {error_count}')
            
        except Exception as e:
            component_health = 'ERROR'
            component_issues.append(f'Health check failed: {e}')
        
        health_report['component_health'][name] = {
            'status': component_health,
            'issues': component_issues
        }
        
        # Update overall health
        if component_health in ['UNHEALTHY', 'ERROR']:
            health_report['overall_health'] = 'UNHEALTHY'
            health_report['issues'].extend(component_issues)
        elif component_health in ['DEGRADED', 'STOPPED'] and health_report['overall_health'] == 'HEALTHY':
            health_report['overall_health'] = 'DEGRADED'
            health_report['warnings'].extend(component_issues)
    
    return health_report

def get_component(component_name: str):
    """
    Get a specific system component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        Component instance or None
    """
    return _system_components.get(component_name)

def list_components():
    """
    List all registered system components.
    
    Returns:
        list: List of component names
    """
    return list(_system_components.keys())

def _register_component(name: str, component):
    """Register a component with the system."""
    _system_components[name] = component

def _unregister_component(name: str):
    """Unregister a component from the system."""
    if name in _system_components:
        del _system_components[name]

def update_system_metrics(metric_name: str, value):
    """Update system metrics."""
    _system_metrics[metric_name] = value

def increment_system_counter(counter_name: str, increment: int = 1):
    """Increment system counter."""
    if counter_name not in _system_metrics:
        _system_metrics[counter_name] = 0
    _system_metrics[counter_name] += increment

# Context manager for system lifecycle
class FIXTradingSystem:
    """Context manager for FIX trading system lifecycle."""
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize trading system context manager.
        
        Args:
            config: System configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config
        self.kwargs = kwargs
        self.system = None
    
    def __enter__(self):
        """Enter context - initialize system."""
        if initialize_core_system(self.config, **self.kwargs):
            self.system = {name: get_component(name) for name in list_components()}
            return self.system
        else:
            raise RuntimeError("Failed to initialize FIX trading system")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - shutdown system."""
        shutdown_core_system()

# Convenience functions for common operations
def send_order(symbol: str, side: str, quantity: int, price: float = None, order_type: str = 'MARKET', **kwargs):
    """
    Send a new order (convenience function).
    
    Args:
        symbol: Trading symbol
        side: Order side (BUY/SELL)
        quantity: Order quantity
        price: Order price (for limit orders)
        order_type: Order type
        **kwargs: Additional order parameters
        
    Returns:
        str: Client order ID
    """
    order_manager = get_component('order_manager')
    if not order_manager:
        raise RuntimeError("Order manager not available")
    
    order_data = {
        'symbol': symbol,
        'side': ORDER_SIDES.get(side.upper(), side),
        'quantity': quantity,
        'order_type': ORDER_TYPES.get(order_type.upper(), order_type),
        **kwargs
    }
    
    if price is not None:
        order_data['price'] = price
    
    return order_manager.send_new_order(**order_data)

def cancel_order(cl_ord_id: str, **kwargs):
    """
    Cancel an order (convenience function).
    
    Args:
        cl_ord_id: Client order ID to cancel
        **kwargs: Additional cancel parameters
        
    Returns:
        bool: True if cancel request sent successfully
    """
    order_manager = get_component('order_manager')
    if not order_manager:
        raise RuntimeError("Order manager not available")
    
    return order_manager.send_cancel_request(cl_ord_id, **kwargs)

def subscribe_market_data(symbols, md_entry_types=None, **kwargs):
    """
    Subscribe to market data (convenience function).
    
    Args:
        symbols: List of symbols or single symbol
        md_entry_types: Market data entry types
        **kwargs: Additional subscription parameters
        
    Returns:
        str: Market data request ID
    """
    md_manager = get_component('market_data_manager')
    if not md_manager:
        raise RuntimeError("Market data manager not available")
    
    if isinstance(symbols, str):
        symbols = [symbols]
    
    if md_entry_types is None:
        md_entry_types = [MD_ENTRY_TYPES['BID'], MD_ENTRY_TYPES['OFFER'], MD_ENTRY_TYPES['TRADE']]
    
    return md_manager.subscribe(symbols, md_entry_types, **kwargs)

def get_order_status(cl_ord_id: str):
    """
    Get order status (convenience function).
    
    Args:
        cl_ord_id: Client order ID
        
    Returns:
        dict: Order status information
    """
    order_manager = get_component('order_manager')
    if not order_manager:
        raise RuntimeError("Order manager not available")
    
    return order_manager.get_order_status(cl_ord_id)

def get_position(symbol: str):
    """
    Get position for symbol (convenience function).
    
    Args:
        symbol: Trading symbol
        
    Returns:
        dict: Position information
    """
    order_manager = get_component('order_manager')
    if not order_manager:
        raise RuntimeError("Order manager not available")
    
    return order_manager.get_position(symbol)

# Version information
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'core_components': len(__all__),
    'default_fix_version': DEFAULT_FIX_VERSION,
    'supported_fix_versions': list(FIX_VERSIONS.keys())
}

def get_version():
    """Get version string."""
    return __version__

def get_version_info():
    """Get detailed version information."""
    return VERSION_INFO.copy()

# Module initialization
def _initialize_module():
    """Initialize core module."""
    from ..utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.debug("FIX trading system core module loaded")

# Initialize module
_initialize_module()

# Usage documentation
USAGE_EXAMPLES = """
FIX Trading System Core Usage Examples:

1. Initialize complete trading system:
   >>> from fix_trading_system.core import FIXTradingSystem
   >>> with FIXTradingSystem() as system:
   ...     # Use system components
   ...     pass

2. Create individual components:
   >>> from fix_trading_system.core import create_fix_engine, create_order_manager
   >>> engine = create_fix_engine()
   >>> order_mgr = create_order_manager()

3. Send orders using convenience functions:
   >>> from fix_trading_system.core import send_order, cancel_order
   >>> cl_ord_id = send_order('AAPL', 'BUY', 100, 150.00, 'LIMIT')
   >>> cancel_order(cl_ord_id)

4. Subscribe to market data:
   >>> from fix_trading_system.core import subscribe_market_data
   >>> md_req_id = subscribe_market_data(['AAPL', 'MSFT'])

5. Check system health:
   >>> from fix_trading_system.core import get_system_status, validate_system_health
   >>> status = get_system_status()
   >>> health = validate_system_health()

6. Manual system lifecycle:
   >>> from fix_trading_system.core import initialize_core_system, shutdown_core_system
   >>> initialize_core_system()
   >>> # ... use system ...
   >>> shutdown_core_system()

7. Access system components:
   >>> from fix_trading_system.core import get_component, list_components
   >>> order_manager = get_component('order_manager')
   >>> components = list_components()

8. Monitor system metrics:
   >>> from fix_trading_system.core import get_system_metrics
   >>> metrics = get_system_metrics()
   >>> print(f"Messages processed: {metrics['total_messages_processed']}")

Core Components:
- FIXEngine: Main FIX protocol engine
- SessionManager: FIX session lifecycle management
- ProtocolHandler: Message processing and routing
- OrderManager: Order lifecycle and execution
- MarketDataManager: Market data subscription and handling
- RiskManager: Real-time risk controls
- ConnectionManager: Network connection management
"""

# Add to module documentation
__doc__ += USAGE_EXAMPLES

# Exception classes for core system
class CoreSystemError(Exception):
    """Base exception for core system errors."""
    pass

class SystemNotInitializedError(CoreSystemError):
    """Exception raised when system is not initialized."""
    pass

class ComponentNotFoundError(CoreSystemError):
    """Exception raised when component is not found."""
    pass

class SystemConfigurationError(CoreSystemError):
    """Exception raised for system configuration errors."""
    pass

class SystemShutdownError(CoreSystemError):
    """Exception raised during system shutdown."""
    pass

# System event callbacks
_system_event_callbacks = {
    'startup': [],
    'shutdown': [],
    'component_added': [],
    'component_removed': [],
    'error': []
}

def register_system_callback(event_type: str, callback):
    """
    Register callback for system events.
    
    Args:
        event_type: Type of event (startup, shutdown, component_added, etc.)
        callback: Callback function
    """
    if event_type in _system_event_callbacks:
        _system_event_callbacks[event_type].append(callback)

def unregister_system_callback(event_type: str, callback):
    """
    Unregister system event callback.
    
    Args:
        event_type: Type of event
        callback: Callback function to remove
    """
    if event_type in _system_event_callbacks:
        try:
            _system_event_callbacks[event_type].remove(callback)
        except ValueError:
            pass

def _fire_system_event(event_type: str, *args, **kwargs):
    """Fire system event callbacks."""
    if event_type in _system_event_callbacks:
        for callback in _system_event_callbacks[event_type]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                from ..utils.logging_config import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Error in system event callback: {e}")

# System performance monitoring
class SystemMonitor:
    """System performance and health monitor."""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitor_thread = None
        self.alerts = []
        self.thresholds = {
            'max_memory_mb': 2048,
            'max_cpu_percent': 80,
            'max_error_rate': 0.01,
            'max_latency_ms': 100
        }
    
    def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring_active:
            return
        
        import threading
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def set_threshold(self, metric: str, value):
        """Set monitoring threshold."""
        self.thresholds[metric] = value
    
    def get_alerts(self):
        """Get current alerts."""
        return self.alerts.copy()
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        import time
        
        while self.monitoring_active:
            try:
                self._check_system_health()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                from ..utils.logging_config import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Error in system monitoring: {e}")
                time.sleep(5)
    
    def _check_system_health(self):
        """Check system health against thresholds."""
        try:
            import psutil
            from datetime import datetime, timezone
            
            # Check memory usage
            memory_mb = psutil.virtual_memory().used / (1024 * 1024)
            if memory_mb > self.thresholds.get('max_memory_mb', 2048):
                self._add_alert('HIGH_MEMORY', f'Memory usage: {memory_mb:.1f}MB')
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.thresholds.get('max_cpu_percent', 80):
                self._add_alert('HIGH_CPU', f'CPU usage: {cpu_percent:.1f}%')
            
            # Check system metrics
            metrics = get_system_metrics()
            
            # Check error rate
            total_messages = metrics.get('total_messages_processed', 1)
            total_errors = metrics.get('errors_count', 0)
            error_rate = total_errors / max(total_messages, 1)
            
            if error_rate > self.thresholds.get('max_error_rate', 0.01):
                self._add_alert('HIGH_ERROR_RATE', f'Error rate: {error_rate:.3f}')
            
        except ImportError:
            # psutil not available, skip system monitoring
            pass
        except Exception as e:
            from ..utils.logging_config import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Error checking system health: {e}")
    
    def _add_alert(self, alert_type: str, message: str):
        """Add system alert."""
        from datetime import datetime, timezone
        
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now(timezone.utc),
            'severity': 'WARNING'
        }
        
        # Avoid duplicate alerts
        for existing_alert in self.alerts:
            if existing_alert['type'] == alert_type:
                existing_alert.update(alert)
                return
        
        self.alerts.append(alert)
        
        # Fire alert callback
        _fire_system_event('alert', alert)

# Global system monitor
_system_monitor = SystemMonitor()

def start_system_monitoring():
    """Start system performance monitoring."""
    _system_monitor.start_monitoring()

def stop_system_monitoring():
    """Stop system performance monitoring."""
    _system_monitor.stop_monitoring()

def get_system_alerts():
    """Get current system alerts."""
    return _system_monitor.get_alerts()

def clear_system_alerts():
    """Clear system alerts."""
    _system_monitor.clear_alerts()

def set_monitoring_threshold(metric: str, value):
    """Set system monitoring threshold."""
    _system_monitor.set_threshold(metric, value)

# System configuration validation
def validate_system_configuration(config):
    """
    Validate system configuration before initialization.
    
    Args:
        config: System configuration
        
    Returns:
        tuple: (is_valid, errors)
    """
    errors = []
    
    try:
        if hasattr(config, 'validate'):
            config_errors = config.validate()
            errors.extend(config_errors)
        
        # Check required components
        required_settings = [
            'session.sender_comp_id',
            'session.target_comp_id',
            'session.heartbeat_interval',
            'network.host',
            'network.port'
        ]
        
        for setting in required_settings:
            if not _get_nested_config_value(config, setting):
                errors.append(f"Missing required setting: {setting}")
        
        # Validate network settings
        if hasattr(config, 'network'):
            if not (1 <= config.network.port <= 65535):
                errors.append("Network port must be between 1 and 65535")
            
            if config.network.connect_timeout <= 0:
                errors.append("Connect timeout must be positive")
        
        # Validate session settings
        if hasattr(config, 'session'):
            if config.session.heartbeat_interval <= 0:
                errors.append("Heartbeat interval must be positive")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"Configuration validation error: {e}")
        return False, errors

def _get_nested_config_value(config, setting_path):
    """Get nested configuration value by dot-separated path."""
    try:
        parts = setting_path.split('.')
        value = config
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return None
        return value
    except:
        return None

# System backup and restore
def backup_system_state(backup_path: str = None):
    """
    Backup current system state.
    
    Args:
        backup_path: Path for backup file
        
    Returns:
        str: Backup file path
    """
    import json
    import os
    from datetime import datetime
    
    if backup_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"system_backup_{timestamp}.json"
    
    try:
        backup_data = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': _system_metrics.copy(),
            'components': list(_system_components.keys()),
            'component_states': {}
        }
        
        # Backup component states
        for name, component in _system_components.items():
            if hasattr(component, 'get_state'):
                backup_data['component_states'][name] = component.get_state()
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        return backup_path
        
    except Exception as e:
        from ..utils.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Failed to backup system state: {e}")
        return None

def restore_system_state(backup_path: str):
    """
    Restore system state from backup.
    
    Args:
        backup_path: Path to backup file
        
    Returns:
        bool: True if restore successful
    """
    import json
    
    try:
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)
        
        # Restore component states
        for name, state in backup_data.get('component_states', {}).items():
            if name in _system_components:
                component = _system_components[name]
                if hasattr(component, 'restore_state'):
                    component.restore_state(state)
        
        from ..utils.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info(f"System state restored from {backup_path}")
        
        return True
        
    except Exception as e:
        from ..utils.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Failed to restore system state: {e}")
        return False

# System debugging utilities
def dump_system_state():
    """
    Dump complete system state for debugging.
    
    Returns:
        dict: Complete system state
    """
    state = {
        'initialized': _system_initialized,
        'components': {},
        'metrics': _system_metrics.copy(),
        'alerts': _system_monitor.get_alerts(),
        'callbacks': {k: len(v) for k, v in _system_event_callbacks.items()}
    }
    
    # Dump component states
    for name, component in _system_components.items():
        component_state = {
            'type': type(component).__name__,
            'memory_address': hex(id(component))
        }
        
        if hasattr(component, 'get_debug_info'):
            component_state['debug_info'] = component.get_debug_info()
        elif hasattr(component, 'get_state'):
            component_state['state'] = component.get_state()
        elif hasattr(component, 'get_status'):
            component_state['status'] = component.get_status()
        
        state['components'][name] = component_state
    
    return state

def get_system_dependencies():
    """
    Get system component dependencies.
    
    Returns:
        dict: Component dependency graph
    """
    dependencies = {}
    
    for name, component in _system_components.items():
        deps = []
        
        # Analyze component dependencies
        if hasattr(component, 'get_dependencies'):
            deps = component.get_dependencies()
        else:
            # Infer dependencies from component attributes
            for attr_name in dir(component):
                if not attr_name.startswith('_'):
                    attr_value = getattr(component, attr_name)
                    if attr_value in _system_components.values():
                        for dep_name, dep_component in _system_components.items():
                            if dep_component is attr_value:
                                deps.append(dep_name)
                                break
        
        dependencies[name] = deps
    
    return dependencies

# Enhanced error handling and recovery
class SystemRecoveryManager:
    """Manages system recovery procedures."""
    
    def __init__(self):
        self.recovery_procedures = {}
        self.recovery_history = []
    
    def register_recovery_procedure(self, error_type: str, procedure):
        """Register recovery procedure for error type."""
        self.recovery_procedures[error_type] = procedure
    
    def attempt_recovery(self, error_type: str, error_context: dict = None):
        """Attempt recovery for specific error type."""
        if error_type not in self.recovery_procedures:
            return False
        
        try:
            procedure = self.recovery_procedures[error_type]
            result = procedure(error_context)
            
            # Record recovery attempt
            from datetime import datetime, timezone
            recovery_record = {
                'timestamp': datetime.now(timezone.utc),
                'error_type': error_type,
                'context': error_context,
                'success': bool(result)
            }
            self.recovery_history.append(recovery_record)
            
            return result
            
        except Exception as e:
            from ..utils.logging_config import get_logger
            logger = get_logger(__name__)
            logger.error(f"Recovery procedure failed for {error_type}: {e}")
            return False
    
    def get_recovery_history(self):
        """Get recovery attempt history."""
        return self.recovery_history.copy()

# Global recovery manager
_recovery_manager = SystemRecoveryManager()

def register_recovery_procedure(error_type: str, procedure):
    """Register system recovery procedure."""
    _recovery_manager.register_recovery_procedure(error_type, procedure)

def attempt_system_recovery(error_type: str, error_context: dict = None):
    """Attempt system recovery."""
    return _recovery_manager.attempt_recovery(error_type, error_context)

def get_recovery_history():
    """Get system recovery history."""
    return _recovery_manager.get_recovery_history()

# Final module exports and documentation
__doc__ += f"""

Available Constants:
- FIX_VERSIONS: {list(FIX_VERSIONS.keys())}
- MESSAGE_TYPES: {len(MESSAGE_TYPES)} message types
- FIELD_TAGS: {len(FIELD_TAGS)} common field tags
- ORDER_SIDES: {list(ORDER_SIDES.keys())}
- ORDER_TYPES: {len(ORDER_TYPES)} order types
- ORDER_STATUSES: {len(ORDER_STATUSES)} order statuses
- TIME_IN_FORCE: {list(TIME_IN_FORCE.keys())}

System Management:
- System initialization and shutdown
- Component lifecycle management
- Health monitoring and alerting
- Performance metrics collection
- Error recovery procedures
- Configuration validation
- State backup and restore

For detailed documentation, see individual component modules.
"""