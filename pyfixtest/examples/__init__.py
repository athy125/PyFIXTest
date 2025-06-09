"""
FIX Trading System Examples

This module provides comprehensive examples and tutorials for using the FIX trading system,
including basic usage, advanced scenarios, integration patterns, and best practices.

Examples Categories:
    - Basic FIX operations (login, orders, market data)
    - Advanced trading strategies and workflows
    - Integration with external systems
    - Performance optimization techniques
    - Error handling and recovery patterns
    - Testing and validation examples
    - Configuration and deployment guides
"""

__version__ = "1.0.0"
__author__ = "FIX Trading System Team"

# Basic examples
from .basic_examples import (
    SimpleOrderExample,
    BasicSessionExample,
    MarketDataExample,
    HeartbeatExample,
    LoginLogoutExample
)

# Trading examples
from .trading_examples import (
    OrderManagementExample,
    OrderBookExample,
    MultiLegOrderExample,
    AlgorithmicTradingExample,
    BasketTradingExample,
    TWAPStrategyExample,
    VWAPStrategyExample,
    IcebergOrderExample
)

# Integration examples
from .integration_examples import (
    DatabaseIntegrationExample,
    RestAPIIntegrationExample,
    WebSocketIntegrationExample,
    ExternalSystemExample,
    RiskSystemIntegrationExample,
    PortfolioIntegrationExample,
    ComplianceIntegrationExample
)

# Advanced examples
from .advanced_examples import (
    HighFrequencyTradingExample,
    MultiVenueExample,
    CrossAssetTradingExample,
    FailoverExample,
    LoadBalancingExample,
    SessionClusteringExample,
    LatencyOptimizationExample
)

# Performance examples
from .performance_examples import (
    ThroughputOptimizationExample,
    MemoryOptimizationExample,
    ConnectionPoolingExample,
    MessageCompressionExample,
    BatchProcessingExample,
    AsynchronousProcessingExample
)

# Testing examples
from .testing_examples import (
    UnitTestExample,
    IntegrationTestExample,
    PerformanceTestExample,
    LoadTestExample,
    MockingExample,
    TestAutomationExample
)

# Configuration examples
from .config_examples import (
    BasicConfigExample,
    ProductionConfigExample,
    SecurityConfigExample,
    ClusterConfigExample,
    MonitoringConfigExample,
    LoggingConfigExample
)

# Best practices
from .best_practices import (
    ErrorHandlingPatterns,
    SecurityBestPractices,
    PerformanceBestPractices,
    MonitoringBestPractices,
    DeploymentBestPractices,
    MaintenanceBestPractices
)

# Export all example classes
__all__ = [
    # Basic examples
    'SimpleOrderExample',
    'BasicSessionExample',
    'MarketDataExample',
    'HeartbeatExample',
    'LoginLogoutExample',
    
    # Trading examples
    'OrderManagementExample',
    'OrderBookExample',
    'MultiLegOrderExample',
    'AlgorithmicTradingExample',
    'BasketTradingExample',
    'TWAPStrategyExample',
    'VWAPStrategyExample',
    'IcebergOrderExample',
    
    # Integration examples
    'DatabaseIntegrationExample',
    'RestAPIIntegrationExample',
    'WebSocketIntegrationExample',
    'ExternalSystemExample',
    'RiskSystemIntegrationExample',
    'PortfolioIntegrationExample',
    'ComplianceIntegrationExample',
    
    # Advanced examples
    'HighFrequencyTradingExample',
    'MultiVenueExample',
    'CrossAssetTradingExample',
    'FailoverExample',
    'LoadBalancingExample',
    'SessionClusteringExample',
    'LatencyOptimizationExample',
    
    # Performance examples
    'ThroughputOptimizationExample',
    'MemoryOptimizationExample',
    'ConnectionPoolingExample',
    'MessageCompressionExample',
    'BatchProcessingExample',
    'AsynchronousProcessingExample',
    
    # Testing examples
    'UnitTestExample',
    'IntegrationTestExample',
    'PerformanceTestExample',
    'LoadTestExample',
    'MockingExample',
    'TestAutomationExample',
    
    # Configuration examples
    'BasicConfigExample',
    'ProductionConfigExample',
    'SecurityConfigExample',
    'ClusterConfigExample',
    'MonitoringConfigExample',
    'LoggingConfigExample',
    
    # Best practices
    'ErrorHandlingPatterns',
    'SecurityBestPractices',
    'PerformanceBestPractices',
    'MonitoringBestPractices',
    'DeploymentBestPractices',
    'MaintenanceBestPractices',
    
    # Helper functions
    'run_example',
    'get_example_list',
    'create_example_session',
    'setup_example_environment',
    'cleanup_example_environment',
    'validate_example_config',
]

# Example categories for easy navigation
EXAMPLE_CATEGORIES = {
    'basic': [
        'SimpleOrderExample',
        'BasicSessionExample',
        'MarketDataExample',
        'HeartbeatExample',
        'LoginLogoutExample'
    ],
    'trading': [
        'OrderManagementExample',
        'OrderBookExample',
        'MultiLegOrderExample',
        'AlgorithmicTradingExample',
        'BasketTradingExample',
        'TWAPStrategyExample',
        'VWAPStrategyExample',
        'IcebergOrderExample'
    ],
    'integration': [
        'DatabaseIntegrationExample',
        'RestAPIIntegrationExample',
        'WebSocketIntegrationExample',
        'ExternalSystemExample',
        'RiskSystemIntegrationExample',
        'PortfolioIntegrationExample',
        'ComplianceIntegrationExample'
    ],
    'advanced': [
        'HighFrequencyTradingExample',
        'MultiVenueExample',
        'CrossAssetTradingExample',
        'FailoverExample',
        'LoadBalancingExample',
        'SessionClusteringExample',
        'LatencyOptimizationExample'
    ],
    'performance': [
        'ThroughputOptimizationExample',
        'MemoryOptimizationExample',
        'ConnectionPoolingExample',
        'MessageCompressionExample',
        'BatchProcessingExample',
        'AsynchronousProcessingExample'
    ],
    'testing': [
        'UnitTestExample',
        'IntegrationTestExample',
        'PerformanceTestExample',
        'LoadTestExample',
        'MockingExample',
        'TestAutomationExample'
    ],
    'configuration': [
        'BasicConfigExample',
        'ProductionConfigExample',
        'SecurityConfigExample',
        'ClusterConfigExample',
        'MonitoringConfigExample',
        'LoggingConfigExample'
    ],
    'best_practices': [
        'ErrorHandlingPatterns',
        'SecurityBestPractices',
        'PerformanceBestPractices',
        'MonitoringBestPractices',
        'DeploymentBestPractices',
        'MaintenanceBestPractices'
    ]
}

# Example difficulty levels
DIFFICULTY_LEVELS = {
    'beginner': [
        'SimpleOrderExample',
        'BasicSessionExample',
        'MarketDataExample',
        'HeartbeatExample',
        'LoginLogoutExample',
        'BasicConfigExample'
    ],
    'intermediate': [
        'OrderManagementExample',
        'OrderBookExample',
        'DatabaseIntegrationExample',
        'RestAPIIntegrationExample',
        'UnitTestExample',
        'IntegrationTestExample',
        'ProductionConfigExample'
    ],
    'advanced': [
        'MultiLegOrderExample',
        'AlgorithmicTradingExample',
        'HighFrequencyTradingExample',
        'MultiVenueExample',
        'FailoverExample',
        'LoadBalancingExample',
        'ThroughputOptimizationExample',
        'PerformanceTestExample'
    ],
    'expert': [
        'CrossAssetTradingExample',
        'SessionClusteringExample',
        'LatencyOptimizationExample',
        'LoadTestExample',
        'SecurityConfigExample',
        'ClusterConfigExample'
    ]
}

# Common example configurations
DEFAULT_EXAMPLE_CONFIG = {
    'sender_comp_id': 'EXAMPLE_CLIENT',
    'target_comp_id': 'EXAMPLE_SERVER',
    'host': 'localhost',
    'port': 9876,
    'heartbeat_interval': 30,
    'timeout': 10,
    'log_level': 'INFO',
    'store_path': './examples/store',
    'log_path': './examples/logs'
}

DEMO_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']
DEMO_CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY']
DEMO_EXCHANGES = ['NASDAQ', 'NYSE', 'LSE', 'TSE']

def get_version():
    """Get version string."""
    return __version__

def run_example(example_name: str, config: dict = None, **kwargs):
    """
    Run a specific example by name.
    
    Args:
        example_name: Name of the example to run
        config: Optional configuration dict
        **kwargs: Additional parameters for the example
        
    Returns:
        Example execution result
    """
    try:
        # Import the example class dynamically
        example_class = globals().get(example_name)
        if not example_class:
            raise ValueError(f"Example '{example_name}' not found")
        
        # Use default config if none provided
        if config is None:
            config = DEFAULT_EXAMPLE_CONFIG.copy()
        
        # Create and run the example
        example = example_class(config, **kwargs)
        return example.run()
        
    except Exception as e:
        print(f"Error running example '{example_name}': {e}")
        return None

def get_example_list(category: str = None, difficulty: str = None) -> list:
    """
    Get list of available examples, optionally filtered by category or difficulty.
    
    Args:
        category: Example category to filter by
        difficulty: Difficulty level to filter by
        
    Returns:
        List of example names
    """
    if category and category in EXAMPLE_CATEGORIES:
        examples = EXAMPLE_CATEGORIES[category]
    elif difficulty and difficulty in DIFFICULTY_LEVELS:
        examples = DIFFICULTY_LEVELS[difficulty]
    else:
        examples = __all__[:len(__all__)//2]  # All example classes
    
    return examples

def get_example_info(example_name: str) -> dict:
    """
    Get information about a specific example.
    
    Args:
        example_name: Name of the example
        
    Returns:
        Dictionary with example information
    """
    try:
        example_class = globals().get(example_name)
        if not example_class:
            return {'error': f"Example '{example_name}' not found"}
        
        # Find category and difficulty
        category = None
        for cat, examples in EXAMPLE_CATEGORIES.items():
            if example_name in examples:
                category = cat
                break
        
        difficulty = None
        for diff, examples in DIFFICULTY_LEVELS.items():
            if example_name in examples:
                difficulty = diff
                break
        
        return {
            'name': example_name,
            'category': category,
            'difficulty': difficulty,
            'description': getattr(example_class, '__doc__', 'No description available'),
            'requirements': getattr(example_class, 'REQUIREMENTS', []),
            'estimated_time': getattr(example_class, 'ESTIMATED_TIME', 'Unknown')
        }
        
    except Exception as e:
        return {'error': f"Error getting info for '{example_name}': {e}"}

def create_example_session(config: dict = None):
    """
    Create a FIX session for examples.
    
    Args:
        config: Session configuration
        
    Returns:
        Configured FIX session
    """
    from ..core.fix_engine import FIXEngine
    from ..config.fix_config import FIXConfig
    
    if config is None:
        config = DEFAULT_EXAMPLE_CONFIG.copy()
    
    fix_config = FIXConfig()
    fix_config.load_from_dict(config)
    
    engine = FIXEngine(fix_config)
    return engine

def setup_example_environment(config: dict = None) -> dict:
    """
    Setup environment for running examples.
    
    Args:
        config: Environment configuration
        
    Returns:
        Environment context
    """
    import os
    import logging
    from datetime import datetime
    
    if config is None:
        config = DEFAULT_EXAMPLE_CONFIG.copy()
    
    # Create directories
    store_path = config.get('store_path', './examples/store')
    log_path = config.get('log_path', './examples/logs')
    
    os.makedirs(store_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Setup logging
    log_level = config.get('log_level', 'INFO')
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        filename=os.path.join(log_path, f'examples_{datetime.now().strftime("%Y%m%d")}.log')
    )
    
    return {
        'store_path': store_path,
        'log_path': log_path,
        'config': config,
        'start_time': datetime.now()
    }

def cleanup_example_environment(env_context: dict):
    """
    Cleanup example environment.
    
    Args:
        env_context: Environment context from setup
    """
    try:
        # Log cleanup
        if 'start_time' in env_context:
            duration = datetime.now() - env_context['start_time']
            logging.info(f"Example session completed. Duration: {duration}")
        
        # Any cleanup logic here
        
    except Exception as e:
        logging.warning(f"Error during environment cleanup: {e}")

def validate_example_config(config: dict) -> tuple:
    """
    Validate example configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Required fields
    required_fields = ['sender_comp_id', 'target_comp_id', 'host', 'port']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate port
    if 'port' in config:
        try:
            port = int(config['port'])
            if port < 1 or port > 65535:
                errors.append("Port must be between 1 and 65535")
        except (ValueError, TypeError):
            errors.append("Port must be a valid integer")
    
    # Validate heartbeat interval
    if 'heartbeat_interval' in config:
        try:
            interval = int(config['heartbeat_interval'])
            if interval < 10 or interval > 300:
                errors.append("Heartbeat interval should be between 10 and 300 seconds")
        except (ValueError, TypeError):
            errors.append("Heartbeat interval must be a valid integer")
    
    return len(errors) == 0, errors

def print_example_menu():
    """Print interactive menu of available examples."""
    print("\n" + "="*60)
    print("FIX Trading System Examples")
    print("="*60)
    
    for category, examples in EXAMPLE_CATEGORIES.items():
        print(f"\n{category.upper()} Examples:")
        print("-" * 40)
        for i, example in enumerate(examples, 1):
            info = get_example_info(example)
            difficulty = info.get('difficulty', 'Unknown')
            print(f"  {i:2d}. {example:<30} [{difficulty}]")
    
    print("\n" + "="*60)
    print("Usage: run_example('ExampleName') or get_example_info('ExampleName')")
    print("="*60)

def run_interactive_examples():
    """Run examples in interactive mode."""
    while True:
        print_example_menu()
        
        try:
            choice = input("\nEnter example name (or 'quit' to exit): ").strip()
            
            if choice.lower() in ['quit', 'exit', 'q']:
                break
            
            if choice in globals():
                print(f"\nRunning {choice}...")
                result = run_example(choice)
                print(f"Example completed. Result: {result}")
            else:
                print(f"Example '{choice}' not found. Please try again.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

# Example usage documentation
USAGE_EXAMPLES = """
Quick Start Examples:

1. Run a simple order example:
   >>> from fix_trading_system.examples import run_example
   >>> run_example('SimpleOrderExample')

2. Get list of beginner examples:
   >>> from fix_trading_system.examples import get_example_list
   >>> examples = get_example_list(difficulty='beginner')

3. Setup example environment:
   >>> from fix_trading_system.examples import setup_example_environment
   >>> env = setup_example_environment()

4. Interactive mode:
   >>> from fix_trading_system.examples import run_interactive_examples
   >>> run_interactive_examples()

5. Get example information:
   >>> from fix_trading_system.examples import get_example_info
   >>> info = get_example_info('AlgorithmicTradingExample')

Categories Available:
- basic: Simple FIX operations and connections
- trading: Order management and trading strategies
- integration: External system integrations
- advanced: Complex scenarios and optimizations
- performance: Performance tuning and optimization
- testing: Testing frameworks and methodologies
- configuration: System configuration examples
- best_practices: Recommended patterns and practices

Difficulty Levels:
- beginner: Basic concepts and simple examples
- intermediate: More complex scenarios
- advanced: Sophisticated implementations
- expert: Cutting-edge techniques and optimizations
"""

# Module-level documentation
__doc__ += USAGE_EXAMPLES

# Configuration validation on import
def _validate_module_config():
    """Validate module configuration on import."""
    try:
        # Check if default config is valid
        is_valid, errors = validate_example_config(DEFAULT_EXAMPLE_CONFIG)
        if not is_valid:
            print(f"Warning: Default example configuration has issues: {errors}")
    except Exception:
        pass  # Ignore validation errors during import

# Run validation
_validate_module_config()

# Example exception classes
class ExampleError(Exception):
    """Base exception for example-related errors."""
    pass

class ExampleConfigError(ExampleError):
    """Exception raised for configuration errors."""
    pass

class ExampleExecutionError(ExampleError):
    """Exception raised during example execution."""
    pass

class ExampleTimeoutError(ExampleError):
    """Exception raised when example times out."""
    pass

# Helper decorators for examples
def example_wrapper(timeout=60, cleanup=True):
    """Decorator for example methods with timeout and cleanup."""
    def decorator(func):
        import functools
        import signal
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise ExampleTimeoutError(f"Example timed out after {timeout} seconds")
            
            # Set timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Cleanup
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
                if cleanup and len(args) > 0:
                    instance = args[0]
                    if hasattr(instance, 'cleanup'):
                        instance.cleanup()
        
        return wrapper
    return decorator

def requires_connection(func):
    """Decorator that ensures FIX connection is available."""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 0:
            instance = args[0]
            if hasattr(instance, 'is_connected') and not instance.is_connected():
                raise ExampleExecutionError("FIX connection required but not available")
        return func(*args, **kwargs)
    return wrapper

# Performance monitoring for examples
class ExamplePerformanceMonitor:
    """Monitor performance of example execution."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timing(self, operation: str):
        """Start timing an operation."""
        import time
        self.start_times[operation] = time.time()
    
    def end_timing(self, operation: str):
        """End timing an operation."""
        import time
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            del self.start_times[operation]
            return duration
        return None
    
    def get_statistics(self) -> dict:
        """Get performance statistics."""
        stats = {}
        for operation, times in self.metrics.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        return stats

# Global performance monitor instance
performance_monitor = ExamplePerformanceMonitor()