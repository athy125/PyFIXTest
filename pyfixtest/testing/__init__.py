"""
FIX Trading System Testing Framework

This module provides comprehensive testing utilities for FIX protocol trading systems,
including unit tests, integration tests, performance tests, and test data generation.

Components:
    - Test fixtures and factories for FIX messages
    - Mock implementations for external dependencies
    - Performance testing and benchmarking tools
    - Test data generators and scenarios
    - Integration test frameworks
    - Validation and assertion helpers
"""

__version__ = "1.0.0"
__author__ = "FIX Trading System Team"

# Core testing imports
from .test_fixtures import (
    FIXMessageFixture,
    SessionFixture,
    OrderFixture,
    MarketDataFixture,
    TestDataFactory
)

from .mock_implementations import (
    MockFIXEngine,
    MockSessionManager,
    MockProtocolHandler,
    MockOrderManager,
    MockMarketDataProvider,
    MockRiskManager
)

from .test_scenarios import (
    OrderTestScenarios,
    SessionTestScenarios,
    MarketDataTestScenarios,
    ErrorTestScenarios,
    PerformanceTestScenarios
)

from .integration_tests import (
    FIXIntegrationTestCase,
    OrderLifecycleTest,
    SessionManagementTest,
    ProtocolComplianceTest,
    EndToEndTest
)

from .performance_tests import (
    PerformanceBenchmark,
    LoadTester,
    LatencyTester,
    ThroughputTester,
    MemoryProfiler
)

from .test_utilities import (
    MessageAssertions,
    SessionAssertions,
    OrderAssertions,
    TimingAssertions,
    TestTimer,
    TestReport,
    TestDataGenerator
)

from .validators import (
    TestMessageValidator,
    TestOrderValidator,
    TestSessionValidator,
    ComplianceValidator
)

# Test configuration and settings
from .test_config import (
    TestConfig,
    TestEnvironment,
    MockConfig,
    TestSessionSettings
)

# Export all public classes and functions
__all__ = [
    # Core fixtures
    'FIXMessageFixture',
    'SessionFixture', 
    'OrderFixture',
    'MarketDataFixture',
    'TestDataFactory',
    
    # Mock implementations
    'MockFIXEngine',
    'MockSessionManager',
    'MockProtocolHandler',
    'MockOrderManager',
    'MockMarketDataProvider',
    'MockRiskManager',
    
    # Test scenarios
    'OrderTestScenarios',
    'SessionTestScenarios',
    'MarketDataTestScenarios',
    'ErrorTestScenarios',
    'PerformanceTestScenarios',
    
    # Integration tests
    'FIXIntegrationTestCase',
    'OrderLifecycleTest',
    'SessionManagementTest',
    'ProtocolComplianceTest',
    'EndToEndTest',
    
    # Performance testing
    'PerformanceBenchmark',
    'LoadTester',
    'LatencyTester',
    'ThroughputTester',
    'MemoryProfiler',
    
    # Utilities
    'MessageAssertions',
    'SessionAssertions',
    'OrderAssertions',
    'TimingAssertions',
    'TestTimer',
    'TestReport',
    'TestDataGenerator',
    
    # Validators
    'TestMessageValidator',
    'TestOrderValidator',
    'TestSessionValidator',
    'ComplianceValidator',
    
    # Configuration
    'TestConfig',
    'TestEnvironment',
    'MockConfig',
    'TestSessionSettings',
    
    # Helper functions
    'create_test_message',
    'create_test_session',
    'create_test_order',
    'generate_test_data',
    'setup_test_environment',
    'cleanup_test_environment',
    'run_performance_test',
    'validate_test_results',
]

# Version information
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

def get_version():
    """Get version string."""
    return __version__

def get_version_info():
    """Get detailed version information."""
    return VERSION_INFO.copy()

# Quick setup functions for common testing scenarios
def create_test_message(msg_type: str, **kwargs):
    """
    Quick helper to create test FIX messages.
    
    Args:
        msg_type: FIX message type
        **kwargs: Message field values
        
    Returns:
        FIX message for testing
    """
    from .test_fixtures import FIXMessageFixture
    fixture = FIXMessageFixture()
    return fixture.create_message(msg_type, **kwargs)

def create_test_session(**kwargs):
    """
    Quick helper to create test sessions.
    
    Args:
        **kwargs: Session configuration
        
    Returns:
        Test session object
    """
    from .test_fixtures import SessionFixture
    fixture = SessionFixture()
    return fixture.create_session(**kwargs)

def create_test_order(symbol: str = "AAPL", side: str = "1", quantity: int = 100, **kwargs):
    """
    Quick helper to create test orders.
    
    Args:
        symbol: Trading symbol
        side: Order side (1=Buy, 2=Sell)
        quantity: Order quantity
        **kwargs: Additional order fields
        
    Returns:
        Test order object
    """
    from .test_fixtures import OrderFixture
    fixture = OrderFixture()
    return fixture.create_order(symbol=symbol, side=side, quantity=quantity, **kwargs)

def generate_test_data(scenario: str, count: int = 10):
    """
    Generate test data for common scenarios.
    
    Args:
        scenario: Test scenario name
        count: Number of items to generate
        
    Returns:
        Generated test data
    """
    from .test_utilities import TestDataGenerator
    generator = TestDataGenerator()
    return generator.generate_scenario_data(scenario, count)

def setup_test_environment(config_name: str = "default"):
    """
    Setup test environment with configuration.
    
    Args:
        config_name: Configuration to use
        
    Returns:
        Test environment context
    """
    from .test_config import TestEnvironment
    env = TestEnvironment(config_name)
    env.setup()
    return env

def cleanup_test_environment(env):
    """
    Cleanup test environment.
    
    Args:
        env: Test environment to cleanup
    """
    if env and hasattr(env, 'cleanup'):
        env.cleanup()

def run_performance_test(test_name: str, **kwargs):
    """
    Run performance test by name.
    
    Args:
        test_name: Performance test to run
        **kwargs: Test parameters
        
    Returns:
        Performance test results
    """
    from .performance_tests import PerformanceBenchmark
    benchmark = PerformanceBenchmark()
    return benchmark.run_test(test_name, **kwargs)

def validate_test_results(results, criteria):
    """
    Validate test results against criteria.
    
    Args:
        results: Test results to validate
        criteria: Validation criteria
        
    Returns:
        Validation result
    """
    from .validators import ComplianceValidator
    validator = ComplianceValidator()
    return validator.validate_results(results, criteria)

# Test discovery and execution helpers
def discover_tests(path: str = None):
    """
    Discover all tests in the testing framework.
    
    Args:
        path: Optional path to search for tests
        
    Returns:
        List of discovered tests
    """
    import unittest
    import os
    
    if path is None:
        path = os.path.dirname(__file__)
    
    loader = unittest.TestLoader()
    return loader.discover(path, pattern='test_*.py')

def run_all_tests():
    """
    Run all tests in the framework.
    
    Returns:
        Test results
    """
    import unittest
    
    suite = discover_tests()
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)

# Common test decorators
def requires_connection(func):
    """Decorator for tests that require active FIX connection."""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check for active connection or skip test
        return func(*args, **kwargs)
    return wrapper

def performance_test(timeout=30):
    """Decorator for performance tests with timeout."""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Add performance monitoring
            return func(*args, **kwargs)
        return wrapper
    return decorator

def integration_test(dependencies=None):
    """Decorator for integration tests with dependency checking."""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check dependencies before running
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Test data constants
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]
TEST_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"]
TEST_EXCHANGES = ["NASDAQ", "NYSE", "LSE", "TSE", "XETRA"]

# Common test configurations
DEFAULT_TEST_CONFIG = {
    'timeout': 30,
    'max_retries': 3,
    'log_level': 'INFO',
    'cleanup_on_exit': True,
    'parallel_execution': False,
    'performance_monitoring': True
}

INTEGRATION_TEST_CONFIG = {
    'timeout': 60,
    'max_retries': 5,
    'log_level': 'DEBUG',
    'cleanup_on_exit': True,
    'parallel_execution': False,
    'performance_monitoring': True,
    'external_services': True
}

PERFORMANCE_TEST_CONFIG = {
    'timeout': 300,
    'max_retries': 1,
    'log_level': 'WARNING',
    'cleanup_on_exit': True,
    'parallel_execution': True,
    'performance_monitoring': True,
    'detailed_metrics': True
}

# Test result codes
class TestResultCode:
    """Test result codes for standardized reporting."""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    INCOMPLETE = "INCOMPLETE"

# Exception classes for testing
class TestFrameworkError(Exception):
    """Base exception for test framework errors."""
    pass

class TestSetupError(TestFrameworkError):
    """Exception raised during test setup."""
    pass

class TestExecutionError(TestFrameworkError):
    """Exception raised during test execution."""
    pass

class TestValidationError(TestFrameworkError):
    """Exception raised during test validation."""
    pass

class TestTimeoutError(TestFrameworkError):
    """Exception raised when test times out."""
    pass

# Logging configuration for tests
def configure_test_logging(level="INFO", format_string=None):
    """
    Configure logging for test execution.
    
    Args:
        level: Logging level
        format_string: Custom format string
    """
    import logging
    
    if format_string is None:
        format_string = '[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Initialize test logging with default configuration
configure_test_logging()

# Module-level documentation
__doc__ += """

Quick Start Examples:

1. Create a simple test message:
    >>> msg = create_test_message('D', symbol='AAPL', side='1', quantity=100)

2. Setup test environment:
    >>> env = setup_test_environment('integration')
    >>> # ... run tests ...
    >>> cleanup_test_environment(env)

3. Run performance test:
    >>> results = run_performance_test('order_latency', iterations=1000)

4. Generate test data:
    >>> orders = generate_test_data('random_orders', count=50)

5. Run all tests:
    >>> results = run_all_tests()

For more detailed examples and documentation, see the individual module files.
"""