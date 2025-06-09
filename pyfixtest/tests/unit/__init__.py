"""
Unit Tests for FIX Trading System

This module provides comprehensive unit tests for the FIX trading system components,
ensuring individual component functionality, edge cases, and error conditions are
properly tested in isolation.

Test Categories:
    - Core component tests (engines, managers, handlers)
    - Configuration and validation tests
    - Message processing and protocol tests
    - Order management and lifecycle tests
    - Market data handling tests
    - Risk management and compliance tests
    - Utilities and helper function tests
    - Error handling and recovery tests
"""

__version__ = "1.0.0"
__author__ = "FIX Trading System Test Team"

import unittest
import sys
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Core component test modules
from .test_fix_engine import (
    TestFIXEngine,
    TestFIXApplication,
    TestEngineConfiguration,
    TestEngineStatistics
)

from .test_session_manager import (
    TestSessionManager,
    TestSessionInfo,
    TestSessionFactory,
    TestSessionEventHandling
)

from .test_protocol_handler import (
    TestProtocolHandler,
    TestMessageContext,
    TestMessageRouter,
    TestMessageProcessing
)

from .test_order_manager import (
    TestOrderManager,
    TestOrder,
    TestOrderBook,
    TestPositionManager,
    TestOrderLifecycle
)

from .test_market_data_manager import (
    TestMarketDataManager,
    TestMarketDataSubscription,
    TestMarketDataProcessing,
    TestQuoteHandling
)

from .test_risk_manager import (
    TestRiskManager,
    TestRiskRules,
    TestPositionLimits,
    TestCreditLimits,
    TestRiskMetrics
)

from .test_message_builder import (
    TestMessageBuilder,
    TestOrderMessageBuilder,
    TestMarketDataMessageBuilder,
    TestAdminMessageBuilder
)

from .test_connection_manager import (
    TestConnectionManager,
    TestConnection,
    TestConnectionPool,
    TestHeartbeatManager
)

# Configuration test modules
from .test_fix_config import (
    TestFIXConfig,
    TestSessionConfig,
    TestNetworkConfig,
    TestSecurityConfig,
    TestPerformanceConfig
)

from .test_config_validation import (
    TestConfigValidator,
    TestEnvironmentConfig,
    TestConfigMerging
)

# Utility test modules
from .test_time_utils import (
    TestTimeUtilities,
    TestFIXTimeFormatting,
    TestBusinessDayCalculations
)

from .test_message_utils import (
    TestMessageUtilities,
    TestMessageParsing,
    TestMessageValidation,
    TestFieldExtraction
)

from .test_logging_config import (
    TestLoggingConfiguration,
    TestLogFormatting,
    TestLogRotation
)

# Validator test modules
from .test_validators import (
    TestMessageValidator,
    TestOrderValidator,
    TestWorkflowValidator,
    TestValidationRules
)

# Export all test classes
__all__ = [
    # Core component tests
    'TestFIXEngine',
    'TestFIXApplication',
    'TestEngineConfiguration',
    'TestEngineStatistics',
    'TestSessionManager',
    'TestSessionInfo',
    'TestSessionFactory',
    'TestSessionEventHandling',
    'TestProtocolHandler',
    'TestMessageContext',
    'TestMessageRouter',
    'TestMessageProcessing',
    'TestOrderManager',
    'TestOrder',
    'TestOrderBook',
    'TestPositionManager',
    'TestOrderLifecycle',
    'TestMarketDataManager',
    'TestMarketDataSubscription',
    'TestMarketDataProcessing',
    'TestQuoteHandling',
    'TestRiskManager',
    'TestRiskRules',
    'TestPositionLimits',
    'TestCreditLimits',
    'TestRiskMetrics',
    'TestMessageBuilder',
    'TestOrderMessageBuilder',
    'TestMarketDataMessageBuilder',
    'TestAdminMessageBuilder',
    'TestConnectionManager',
    'TestConnection',
    'TestConnectionPool',
    'TestHeartbeatManager',
    
    # Configuration tests
    'TestFIXConfig',
    'TestSessionConfig',
    'TestNetworkConfig',
    'TestSecurityConfig',
    'TestPerformanceConfig',
    'TestConfigValidator',
    'TestEnvironmentConfig',
    'TestConfigMerging',
    
    # Utility tests
    'TestTimeUtilities',
    'TestFIXTimeFormatting',
    'TestBusinessDayCalculations',
    'TestMessageUtilities',
    'TestMessageParsing',
    'TestMessageValidation',
    'TestFieldExtraction',
    'TestLoggingConfiguration',
    'TestLogFormatting',
    'TestLogRotation',
    
    # Validator tests
    'TestMessageValidator',
    'TestOrderValidator',
    'TestWorkflowValidator',
    'TestValidationRules',
    
    # Test suite functions
    'create_test_suite',
    'run_all_tests',
    'run_test_category',
    'run_component_tests',
    'get_test_coverage',
    'generate_test_report',
]

# Test configuration
TEST_CONFIG = {
    'verbosity': 2,
    'failfast': False,
    'buffer': True,
    'catch_ctrl_c': True,
    'tb_locals': False,
    'warnings': 'default'
}

# Test categories for organization
TEST_CATEGORIES = {
    'core': [
        'TestFIXEngine',
        'TestFIXApplication',
        'TestEngineConfiguration',
        'TestEngineStatistics',
        'TestSessionManager',
        'TestSessionInfo',
        'TestSessionFactory',
        'TestProtocolHandler',
        'TestMessageContext',
        'TestMessageRouter',
        'TestConnectionManager',
        'TestConnection',
        'TestConnectionPool'
    ],
    'trading': [
        'TestOrderManager',
        'TestOrder',
        'TestOrderBook',
        'TestPositionManager',
        'TestOrderLifecycle',
        'TestMarketDataManager',
        'TestMarketDataSubscription',
        'TestMarketDataProcessing',
        'TestQuoteHandling'
    ],
    'risk': [
        'TestRiskManager',
        'TestRiskRules',
        'TestPositionLimits',
        'TestCreditLimits',
        'TestRiskMetrics'
    ],
    'messaging': [
        'TestMessageBuilder',
        'TestOrderMessageBuilder',
        'TestMarketDataMessageBuilder',
        'TestAdminMessageBuilder',
        'TestMessageProcessing',
        'TestMessageUtilities',
        'TestMessageParsing',
        'TestMessageValidation'
    ],
    'configuration': [
        'TestFIXConfig',
        'TestSessionConfig',
        'TestNetworkConfig',
        'TestSecurityConfig',
        'TestPerformanceConfig',
        'TestConfigValidator',
        'TestEnvironmentConfig',
        'TestConfigMerging'
    ],
    'utilities': [
        'TestTimeUtilities',
        'TestFIXTimeFormatting',
        'TestBusinessDayCalculations',
        'TestFieldExtraction',
        'TestLoggingConfiguration',
        'TestLogFormatting',
        'TestLogRotation'
    ],
    'validation': [
        'TestMessageValidator',
        'TestOrderValidator',
        'TestWorkflowValidator',
        'TestValidationRules'
    ]
}

# Test priorities (for running critical tests first)
CRITICAL_TESTS = [
    'TestFIXEngine',
    'TestSessionManager',
    'TestOrderManager',
    'TestRiskManager',
    'TestFIXConfig'
]

HIGH_PRIORITY_TESTS = [
    'TestProtocolHandler',
    'TestMessageBuilder',
    'TestMarketDataManager',
    'TestConnectionManager',
    'TestMessageValidator'
]

def create_test_suite(test_names: List[str] = None) -> unittest.TestSuite:
    """
    Create test suite from specified test names.
    
    Args:
        test_names: List of test class names to include
        
    Returns:
        unittest.TestSuite: Configured test suite
    """
    suite = unittest.TestSuite()
    
    if test_names is None:
        # Include all tests
        test_names = __all__[:-6]  # Exclude utility functions
    
    # Get test classes from globals
    for test_name in test_names:
        if test_name in globals():
            test_class = globals()[test_name]
            if isinstance(test_class, type) and issubclass(test_class, unittest.TestCase):
                # Add all test methods from the class
                suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test_class))
    
    return suite

def run_all_tests(verbosity: int = 2, failfast: bool = False) -> unittest.TestResult:
    """
    Run all unit tests.
    
    Args:
        verbosity: Test output verbosity level
        failfast: Stop on first failure
        
    Returns:
        unittest.TestResult: Test execution results
    """
    suite = create_test_suite()
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        failfast=failfast,
        buffer=TEST_CONFIG['buffer'],
        warnings=TEST_CONFIG['warnings']
    )
    
    print(f"Running {suite.countTestCases()} unit tests...")
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result

def run_test_category(category: str, verbosity: int = 2) -> unittest.TestResult:
    """
    Run tests from a specific category.
    
    Args:
        category: Test category name
        verbosity: Test output verbosity level
        
    Returns:
        unittest.TestResult: Test execution results
    """
    if category not in TEST_CATEGORIES:
        raise ValueError(f"Unknown test category: {category}. Available: {list(TEST_CATEGORIES.keys())}")
    
    test_names = TEST_CATEGORIES[category]
    suite = create_test_suite(test_names)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    print(f"Running {category} tests ({suite.countTestCases()} tests)...")
    return runner.run(suite)

def run_component_tests(component_name: str, verbosity: int = 2) -> unittest.TestResult:
    """
    Run tests for a specific component.
    
    Args:
        component_name: Component name (e.g., 'fix_engine', 'order_manager')
        verbosity: Test output verbosity level
        
    Returns:
        unittest.TestResult: Test execution results
    """
    # Find test classes related to the component
    test_names = []
    component_patterns = [
        f"Test{component_name.title().replace('_', '')}",
        f"Test{component_name.replace('_', '').upper()}"
    ]
    
    for test_name in __all__:
        for pattern in component_patterns:
            if pattern in test_name:
                test_names.append(test_name)
                break
    
    if not test_names:
        raise ValueError(f"No tests found for component: {component_name}")
    
    suite = create_test_suite(test_names)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    print(f"Running {component_name} tests ({suite.countTestCases()} tests)...")
    return runner.run(suite)

def run_critical_tests(verbosity: int = 2) -> unittest.TestResult:
    """
    Run only critical tests (fastest, most important).
    
    Args:
        verbosity: Test output verbosity level
        
    Returns:
        unittest.TestResult: Test execution results
    """
    suite = create_test_suite(CRITICAL_TESTS)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    print(f"Running critical tests ({suite.countTestCases()} tests)...")
    return runner.run(suite)

def run_smoke_tests() -> bool:
    """
    Run smoke tests (basic functionality check).
    
    Returns:
        bool: True if all smoke tests pass
    """
    # Create minimal test suite with one test from each critical component
    smoke_tests = []
    
    for category, test_names in TEST_CATEGORIES.items():
        if test_names:
            # Take first test from each category
            smoke_tests.append(test_names[0])
    
    suite = create_test_suite(smoke_tests)
    runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
    
    result = runner.run(suite)
    return result.wasSuccessful()

def get_test_coverage() -> Dict[str, Any]:
    """
    Get test coverage information.
    
    Returns:
        dict: Test coverage statistics
    """
    try:
        import coverage
        
        # This would integrate with coverage.py if available
        cov = coverage.Coverage()
        cov.start()
        
        # Run all tests
        result = run_all_tests(verbosity=0)
        
        cov.stop()
        cov.save()
        
        # Get coverage report
        coverage_data = cov.get_data()
        
        return {
            'lines_covered': len(coverage_data.lines),
            'total_tests': result.testsRun,
            'test_success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1),
            'coverage_available': True
        }
        
    except ImportError:
        # Coverage.py not available
        result = run_all_tests(verbosity=0)
        
        return {
            'total_tests': result.testsRun,
            'test_success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1),
            'coverage_available': False,
            'message': 'Install coverage.py for detailed coverage metrics'
        }

def generate_test_report(output_file: str = None) -> str:
    """
    Generate comprehensive test report.
    
    Args:
        output_file: Optional output file path
        
    Returns:
        str: Test report content
    """
    from datetime import datetime
    import platform
    
    # Run all tests and collect results
    result = run_all_tests(verbosity=0)
    coverage_info = get_test_coverage()
    
    # Generate report
    report_lines = [
        "FIX Trading System - Unit Test Report",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Platform: {platform.platform()}",
        f"Python: {platform.python_version()}",
        "",
        "Test Summary:",
        f"  Total Tests: {result.testsRun}",
        f"  Passed: {result.testsRun - len(result.failures) - len(result.errors)}",
        f"  Failed: {len(result.failures)}",
        f"  Errors: {len(result.errors)}",
        f"  Success Rate: {coverage_info['test_success_rate']:.1%}",
        ""
    ]
    
    # Add category breakdown
    report_lines.append("Test Categories:")
    for category, test_names in TEST_CATEGORIES.items():
        category_result = run_test_category(category, verbosity=0)
        success_rate = (category_result.testsRun - len(category_result.failures) - len(category_result.errors)) / max(category_result.testsRun, 1)
        report_lines.append(f"  {category.title()}: {category_result.testsRun} tests, {success_rate:.1%} success")
    
    report_lines.append("")
    
    # Add failure details
    if result.failures:
        report_lines.append("Failures:")
        for i, (test, traceback) in enumerate(result.failures, 1):
            report_lines.append(f"  {i}. {test}")
            report_lines.append(f"     {traceback.split('AssertionError:')[-1].strip()}")
        report_lines.append("")
    
    # Add error details
    if result.errors:
        report_lines.append("Errors:")
        for i, (test, traceback) in enumerate(result.errors, 1):
            report_lines.append(f"  {i}. {test}")
            report_lines.append(f"     {traceback.split('Exception:')[-1].strip()}")
        report_lines.append("")
    
    # Add coverage information
    if coverage_info.get('coverage_available'):
        report_lines.append("Coverage Information:")
        report_lines.append(f"  Lines Covered: {coverage_info.get('lines_covered', 'N/A')}")
    else:
        report_lines.append("Coverage: Not available (install coverage.py)")
    
    report_lines.append("")
    report_lines.append("Recommendations:")
    
    if result.failures or result.errors:
        report_lines.append("  - Fix failing tests before deployment")
    
    if coverage_info['test_success_rate'] < 0.95:
        report_lines.append("  - Improve test reliability (target: 95%+ success rate)")
    
    if not coverage_info.get('coverage_available'):
        report_lines.append("  - Install coverage.py for code coverage analysis")
    
    report_content = "\n".join(report_lines)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_content)
        print(f"Test report saved to: {output_file}")
    
    return report_content

def discover_and_validate_tests() -> Dict[str, Any]:
    """
    Discover and validate all test modules.
    
    Returns:
        dict: Test discovery and validation results
    """
    results = {
        'discovered_tests': [],
        'missing_tests': [],
        'invalid_tests': [],
        'test_count': 0,
        'issues': []
    }
    
    # Expected test modules based on core components
    expected_modules = [
        'test_fix_engine',
        'test_session_manager',
        'test_protocol_handler',
        'test_order_manager',
        'test_market_data_manager',
        'test_risk_manager',
        'test_message_builder',
        'test_connection_manager',
        'test_fix_config',
        'test_validators',
        'test_time_utils',
        'test_message_utils'
    ]
    
    # Check for test modules
    current_dir = Path(__file__).parent
    
    for module_name in expected_modules:
        module_file = current_dir / f"{module_name}.py"
        
        if module_file.exists():
            try:
                # Try to import the module
                __import__(f"tests.unit.{module_name}")
                results['discovered_tests'].append(module_name)
            except ImportError as e:
                results['invalid_tests'].append((module_name, str(e)))
        else:
            results['missing_tests'].append(module_name)
    
    # Count total test methods
    for test_class_name in __all__:
        if test_class_name.startswith('Test') and test_class_name in globals():
            test_class = globals()[test_class_name]
            if isinstance(test_class, type) and issubclass(test_class, unittest.TestCase):
                # Count test methods
                test_methods = [method for method in dir(test_class) if method.startswith('test_')]
                results['test_count'] += len(test_methods)
    
    # Identify issues
    if results['missing_tests']:
        results['issues'].append(f"Missing test modules: {results['missing_tests']}")
    
    if results['invalid_tests']:
        results['issues'].append(f"Invalid test modules: {[name for name, _ in results['invalid_tests']]}")
    
    if results['test_count'] < 50:  # Minimum expected test count
        results['issues'].append(f"Low test count: {results['test_count']} (expected: 50+)")
    
    return results

# Test utilities for common test scenarios
class FIXTestCase(unittest.TestCase):
    """Base test case class with FIX-specific utilities."""
    
    def setUp(self):
        """Set up test case."""
        from fix_trading_system.testing import create_test_config
        self.test_config = create_test_config("unit")
        self.test_config.setup()
    
    def tearDown(self):
        """Clean up test case."""
        if hasattr(self, 'test_config'):
            self.test_config.cleanup()
    
    def assertFIXMessage(self, message, expected_msg_type):
        """Assert FIX message has expected type."""
        self.assertIsNotNone(message)
        msg_type = message.getHeader().getField(35)
        self.assertEqual(msg_type, expected_msg_type)
    
    def assertOrderValid(self, order_data):
        """Assert order data is valid."""
        required_fields = ['symbol', 'side', 'quantity']
        for field in required_fields:
            self.assertIn(field, order_data)
            self.assertIsNotNone(order_data[field])
    
    def create_test_order(self, **kwargs):
        """Create test order with default values."""
        default_order = {
            'symbol': 'TEST.AAPL',
            'side': '1',  # Buy
            'quantity': 100,
            'order_type': '2',  # Limit
            'price': 150.00
        }
        default_order.update(kwargs)
        return default_order

# Test runner configuration
def configure_test_runner():
    """Configure test runner with optimal settings."""
    # Set up test environment
    os.environ['FIX_ENVIRONMENT'] = 'test'
    os.environ['FIX_LOG_LEVEL'] = 'ERROR'  # Reduce log noise during tests
    
    # Configure warnings
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

# Command-line interface for running tests
def main():
    """Main entry point for running unit tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FIX Trading System Unit Tests')
    parser.add_argument('--category', '-c', choices=list(TEST_CATEGORIES.keys()),
                       help='Run tests from specific category')
    parser.add_argument('--component', help='Run tests for specific component')
    parser.add_argument('--critical', action='store_true',
                       help='Run only critical tests')
    parser.add_argument('--smoke', action='store_true',
                       help='Run smoke tests only')
    parser.add_argument('--report', help='Generate test report to file')
    parser.add_argument('--coverage', action='store_true',
                       help='Show coverage information')
    parser.add_argument('--discover', action='store_true',
                       help='Discover and validate test modules')
    parser.add_argument('--verbose', '-v', action='count', default=2,
                       help='Increase verbosity')
    parser.add_argument('--failfast', '-f', action='store_true',
                       help='Stop on first failure')
    
    args = parser.parse_args()
    
    # Configure test environment
    configure_test_runner()
    
    try:
        if args.discover:
            # Discover and validate tests
            results = discover_and_validate_tests()
            print("Test Discovery Results:")
            print(f"  Discovered: {len(results['discovered_tests'])} modules")
            print(f"  Missing: {len(results['missing_tests'])} modules")
            print(f"  Invalid: {len(results['invalid_tests'])} modules")
            print(f"  Total Tests: {results['test_count']}")
            
            if results['issues']:
                print("Issues:")
                for issue in results['issues']:
                    print(f"  - {issue}")
            
            return 0 if not results['issues'] else 1
        
        elif args.smoke:
            # Run smoke tests
            success = run_smoke_tests()
            print(f"Smoke tests: {'PASSED' if success else 'FAILED'}")
            return 0 if success else 1
        
        elif args.critical:
            # Run critical tests
            result = run_critical_tests(args.verbose)
            return 0 if result.wasSuccessful() else 1
        
        elif args.category:
            # Run category tests
            result = run_test_category(args.category, args.verbose)
            return 0 if result.wasSuccessful() else 1
        
        elif args.component:
            # Run component tests
            result = run_component_tests(args.component, args.verbose)
            return 0 if result.wasSuccessful() else 1
        
        else:
            # Run all tests
            result = run_all_tests(args.verbose, args.failfast)
            
            if args.coverage:
                coverage_info = get_test_coverage()
                print(f"\nCoverage Information:")
                print(f"  Test Success Rate: {coverage_info['test_success_rate']:.1%}")
                if coverage_info.get('coverage_available'):
                    print(f"  Lines Covered: {coverage_info.get('lines_covered', 'N/A')}")
                else:
                    print(f"  {coverage_info.get('message', 'Coverage not available')}")
            
            if args.report:
                generate_test_report(args.report)
            
            return 0 if result.wasSuccessful() else 1
    
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == '__main__':
    exit(main())

# Module initialization
configure_test_runner()

# Export version info
VERSION_INFO = {
    'version': __version__,
    'test_categories': len(TEST_CATEGORIES),
    'total_test_classes': len([name for name in __all__ if name.startswith('Test')]),
    'critical_tests': len(CRITICAL_TESTS),
    'high_priority_tests': len(HIGH_PRIORITY_TESTS)
}

def get_version():
    """Get version string."""
    return __version__

def get_version_info():
    """Get detailed version information."""
    return VERSION_INFO.copy()