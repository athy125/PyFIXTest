"""
FIX Trading System Integration Tests

This module provides comprehensive integration tests for the FIX trading system,
focusing on end-to-end workflows, system interactions, and real-world scenarios.

Integration test categories:
    - Session lifecycle and workflow testing
    - Order management and execution workflows
    - Market data subscription and processing
    - Protocol compliance and message flows
    - Error handling and recovery scenarios
    - Performance and load testing
    - Multi-component system integration
    - External system integration patterns
"""

__version__ = "1.0.0"
__author__ = "FIX Trading System Integration Test Team"

import unittest
import sys
import os
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Core integration test modules
from .test_session_workflow import (
    TestSessionWorkflow,
    TestSessionEstablishment,
    TestHeartbeatManagement,
    TestMessageFlowWorkflows,
    TestErrorRecoveryWorkflows,
    TestSessionTermination,
    TestConcurrentSessionWorkflows,
    TestSessionPerformanceWorkflows,
    TestSessionComplianceWorkflows
)

from .test_order_lifecycle import (
    TestOrderLifecycle,
    TestOrderPlacement,
    TestOrderExecution,
    TestOrderCancellation,
    TestOrderModification,
    TestComplexOrderScenarios
)

# Additional integration test modules (to be implemented)
from .test_market_data_integration import (
    TestMarketDataIntegration,
    TestMarketDataSubscription,
    TestMarketDataProcessing,
    TestMarketDataRecovery
)

from .test_protocol_compliance import (
    TestProtocolCompliance,
    TestMessageValidation,
    TestSequenceManagement,
    TestTimingCompliance
)

from .test_system_integration import (
    TestSystemIntegration,
    TestComponentInteraction,
    TestEndToEndWorkflows,
    TestSystemResilience
)

from .test_performance_integration import (
    TestPerformanceIntegration,
    TestThroughputScenarios,
    TestLatencyMeasurement,
    TestLoadTesting
)

from .test_external_integration import (
    TestExternalIntegration,
    TestDatabaseIntegration,
    TestAPIIntegration,
    TestMessageBrokerIntegration
)

# Export all test classes
__all__ = [
    # Session workflow tests
    'TestSessionWorkflow',
    'TestSessionEstablishment',
    'TestHeartbeatManagement',
    'TestMessageFlowWorkflows',
    'TestErrorRecoveryWorkflows',
    'TestSessionTermination',
    'TestConcurrentSessionWorkflows',
    'TestSessionPerformanceWorkflows',
    'TestSessionComplianceWorkflows',
    
    # Order lifecycle tests
    'TestOrderLifecycle',
    'TestOrderPlacement',
    'TestOrderExecution',
    'TestOrderCancellation',
    'TestOrderModification',
    'TestComplexOrderScenarios',
    
    # Market data integration tests
    'TestMarketDataIntegration',
    'TestMarketDataSubscription',
    'TestMarketDataProcessing',
    'TestMarketDataRecovery',
    
    # Protocol compliance tests
    'TestProtocolCompliance',
    'TestMessageValidation',
    'TestSequenceManagement',
    'TestTimingCompliance',
    
    # System integration tests
    'TestSystemIntegration',
    'TestComponentInteraction',
    'TestEndToEndWorkflows',
    'TestSystemResilience',
    
    # Performance integration tests
    'TestPerformanceIntegration',
    'TestThroughputScenarios',
    'TestLatencyMeasurement',
    'TestLoadTesting',
    
    # External integration tests
    'TestExternalIntegration',
    'TestDatabaseIntegration',
    'TestAPIIntegration',
    'TestMessageBrokerIntegration',
    
    # Utility functions
    'run_integration_tests',
    'run_test_suite',
    'run_workflow_tests',
    'run_performance_tests',
    'create_integration_test_suite',
    'setup_integration_environment',
    'cleanup_integration_environment',
    'get_integration_test_config',
    'validate_test_environment',
]

# Test categories for organized execution
INTEGRATION_TEST_CATEGORIES = {
    'workflow': [
        'TestSessionWorkflow',
        'TestSessionEstablishment',
        'TestHeartbeatManagement',
        'TestMessageFlowWorkflows',
        'TestErrorRecoveryWorkflows',
        'TestSessionTermination'
    ],
    'trading': [
        'TestOrderLifecycle',
        'TestOrderPlacement',
        'TestOrderExecution',
        'TestOrderCancellation',
        'TestOrderModification',
        'TestComplexOrderScenarios'
    ],
    'market_data': [
        'TestMarketDataIntegration',
        'TestMarketDataSubscription',
        'TestMarketDataProcessing',
        'TestMarketDataRecovery'
    ],
    'protocol': [
        'TestProtocolCompliance',
        'TestMessageValidation',
        'TestSequenceManagement',
        'TestTimingCompliance'
    ],
    'system': [
        'TestSystemIntegration',
        'TestComponentInteraction',
        'TestEndToEndWorkflows',
        'TestSystemResilience'
    ],
    'performance': [
        'TestPerformanceIntegration',
        'TestThroughputScenarios',
        'TestLatencyMeasurement',
        'TestLoadTesting',
        'TestConcurrentSessionWorkflows',
        'TestSessionPerformanceWorkflows'
    ],
    'external': [
        'TestExternalIntegration',
        'TestDatabaseIntegration',
        'TestAPIIntegration',
        'TestMessageBrokerIntegration'
    ]
}

# Test complexity levels
TEST_COMPLEXITY = {
    'basic': [
        'TestSessionEstablishment',
        'TestOrderPlacement',
        'TestMarketDataSubscription',
        'TestMessageValidation'
    ],
    'intermediate': [
        'TestSessionWorkflow',
        'TestOrderLifecycle',
        'TestHeartbeatManagement',
        'TestProtocolCompliance',
        'TestComponentInteraction'
    ],
    'advanced': [
        'TestErrorRecoveryWorkflows',
        'TestComplexOrderScenarios',
        'TestSystemIntegration',
        'TestPerformanceIntegration',
        'TestConcurrentSessionWorkflows'
    ],
    'expert': [
        'TestEndToEndWorkflows',
        'TestSystemResilience',
        'TestLoadTesting',
        'TestExternalIntegration'
    ]
}

# Default test configuration
DEFAULT_INTEGRATION_CONFIG = {
    'test_timeout': 300,  # 5 minutes per test
    'setup_timeout': 60,  # 1 minute for setup
    'teardown_timeout': 30,  # 30 seconds for teardown
    'parallel_execution': False,  # Sequential by default for integration tests
    'retry_failed_tests': True,
    'max_retries': 2,
    'log_level': 'INFO',
    'capture_logs': True,
    'generate_reports': True,
    'environment': 'integration'
}

# Test environment requirements
ENVIRONMENT_REQUIREMENTS = {
    'network_connectivity': True,
    'file_system_access': True,
    'temp_directory': True,
    'process_isolation': True,
    'threading_support': True,
    'mock_services': True,
    'test_data_access': True
}

def get_version():
    """Get integration tests version."""
    return __version__

def run_integration_tests(
    categories: Optional[List[str]] = None,
    complexity: Optional[str] = None,
    parallel: bool = False,
    verbosity: int = 2
) -> unittest.TestResult:
    """
    Run integration tests with specified filters.
    
    Args:
        categories: List of test categories to run
        complexity: Test complexity level to run
        parallel: Whether to run tests in parallel
        verbosity: Test output verbosity level
        
    Returns:
        unittest.TestResult: Test execution results
    """
    # Create test suite based on filters
    suite = create_integration_test_suite(categories, complexity)
    
    # Configure test runner
    runner_config = DEFAULT_INTEGRATION_CONFIG.copy()
    runner_config['parallel_execution'] = parallel
    
    # Setup test environment
    env_context = setup_integration_environment(runner_config)
    
    try:
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=verbosity,
            buffer=runner_config['capture_logs'],
            failfast=False,
            warnings='default'
        )
        
        print(f"Running {suite.countTestCases()} integration tests...")
        if categories:
            print(f"Categories: {', '.join(categories)}")
        if complexity:
            print(f"Complexity: {complexity}")
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # Generate summary
        _print_test_summary(result, end_time - start_time)
        
        return result
        
    finally:
        # Cleanup environment
        cleanup_integration_environment(env_context)

def run_test_suite(suite_name: str, **kwargs) -> unittest.TestResult:
    """
    Run a specific test suite.
    
    Args:
        suite_name: Name of the test suite to run
        **kwargs: Additional arguments for test execution
        
    Returns:
        unittest.TestResult: Test execution results
    """
    if suite_name == 'all':
        return run_integration_tests(**kwargs)
    elif suite_name in INTEGRATION_TEST_CATEGORIES:
        return run_integration_tests(categories=[suite_name], **kwargs)
    elif suite_name in TEST_COMPLEXITY:
        return run_integration_tests(complexity=suite_name, **kwargs)
    else:
        raise ValueError(f"Unknown test suite: {suite_name}")

def run_workflow_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run workflow-focused integration tests."""
    return run_integration_tests(
        categories=['workflow', 'trading'],
        verbosity=verbosity
    )

def run_performance_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run performance-focused integration tests."""
    return run_integration_tests(
        categories=['performance'],
        verbosity=verbosity
    )

def create_integration_test_suite(
    categories: Optional[List[str]] = None,
    complexity: Optional[str] = None
) -> unittest.TestSuite:
    """
    Create integration test suite with specified filters.
    
    Args:
        categories: Test categories to include
        complexity: Test complexity level to include
        
    Returns:
        unittest.TestSuite: Configured test suite
    """
    suite = unittest.TestSuite()
    
    # Determine which test classes to include
    test_classes_to_run = set()
    
    if categories:
        for category in categories:
            if category in INTEGRATION_TEST_CATEGORIES:
                test_classes_to_run.update(INTEGRATION_TEST_CATEGORIES[category])
    
    if complexity:
        if complexity in TEST_COMPLEXITY:
            if test_classes_to_run:
                # Intersect with complexity filter
                test_classes_to_run &= set(TEST_COMPLEXITY[complexity])
            else:
                test_classes_to_run.update(TEST_COMPLEXITY[complexity])
    
    # If no filters specified, include all tests
    if not test_classes_to_run:
        for category_tests in INTEGRATION_TEST_CATEGORIES.values():
            test_classes_to_run.update(category_tests)
    
    # Add test classes to suite
    for test_class_name in test_classes_to_run:
        if test_class_name in globals():
            test_class = globals()[test_class_name]
            if isinstance(test_class, type) and issubclass(test_class, unittest.TestCase):
                suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test_class))
    
    return suite

def setup_integration_environment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup integration test environment.
    
    Args:
        config: Test configuration
        
    Returns:
        Dict: Environment context for cleanup
    """
    import tempfile
    import logging
    
    # Create temporary directory for test artifacts
    temp_dir = tempfile.mkdtemp(prefix='fix_integration_tests_')
    
    # Setup logging
    log_level = config.get('log_level', 'INFO')
    if config.get('capture_logs', True):
        log_file = os.path.join(temp_dir, 'integration_tests.log')
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    # Validate environment requirements
    validation_result = validate_test_environment()
    if not validation_result['valid']:
        raise RuntimeError(f"Environment validation failed: {validation_result['issues']}")
    
    # Setup test configuration
    test_config = get_integration_test_config(config.get('environment', 'integration'))
    
    environment_context = {
        'temp_dir': temp_dir,
        'log_file': log_file if config.get('capture_logs') else None,
        'test_config': test_config,
        'start_time': datetime.now(timezone.utc),
        'config': config
    }
    
    print(f"Integration test environment setup complete: {temp_dir}")
    return environment_context

def cleanup_integration_environment(env_context: Dict[str, Any]):
    """
    Cleanup integration test environment.
    
    Args:
        env_context: Environment context from setup
    """
    import shutil
    
    try:
        # Calculate test session duration
        if 'start_time' in env_context:
            duration = datetime.now(timezone.utc) - env_context['start_time']
            print(f"Integration test session duration: {duration}")
        
        # Cleanup temporary directory
        if 'temp_dir' in env_context and os.path.exists(env_context['temp_dir']):
            shutil.rmtree(env_context['temp_dir'])
            print(f"Cleaned up test directory: {env_context['temp_dir']}")
        
        # Additional cleanup tasks
        _cleanup_test_resources()
        
    except Exception as e:
        print(f"Warning: Error during environment cleanup: {e}")

def get_integration_test_config(environment: str = 'integration') -> Dict[str, Any]:
    """
    Get integration test configuration for specified environment.
    
    Args:
        environment: Test environment name
        
    Returns:
        Dict: Test configuration
    """
    from pyfixtest.testing.test_config import create_integration_test_config
    
    # Create environment-specific test configuration
    test_config = create_integration_test_config()
    
    # Environment-specific overrides
    env_overrides = {
        'integration': {
            'mock_config.simulate_network_latency': True,
            'mock_config.network_latency_ms': 20,
            'mock_config.simulate_errors': True,
            'mock_config.error_rate': 0.001,
            'session_settings.heartbeat_interval': 30,
            'session_settings.strict_validation': True
        },
        'performance': {
            'mock_config.simulate_network_latency': False,
            'mock_config.simulate_errors': False,
            'session_settings.heartbeat_interval': 10,
            'fix_config.performance.low_latency_mode': True,
            'fix_config.performance.message_queue_size': 50000
        },
        'load': {
            'mock_config.simulate_network_latency': True,
            'mock_config.network_latency_ms': 50,
            'mock_config.simulate_errors': True,
            'mock_config.error_rate': 0.01,
            'fix_config.performance.high_throughput_mode': True,
            'fix_config.performance.worker_threads': 16
        }
    }
    
    # Apply environment-specific overrides
    overrides = env_overrides.get(environment, {})
    for key, value in overrides.items():
        _apply_config_override(test_config, key, value)
    
    return {
        'test_config': test_config,
        'environment': environment,
        'overrides_applied': list(overrides.keys())
    }

def validate_test_environment() -> Dict[str, Any]:
    """
    Validate integration test environment requirements.
    
    Returns:
        Dict: Validation results
    """
    validation_result = {
        'valid': True,
        'issues': [],
        'requirements_met': {},
        'system_info': {}
    }
    
    # Check environment requirements
    for requirement, needed in ENVIRONMENT_REQUIREMENTS.items():
        if not needed:
            validation_result['requirements_met'][requirement] = True
            continue
        
        met = _check_requirement(requirement)
        validation_result['requirements_met'][requirement] = met
        
        if not met:
            validation_result['valid'] = False
            validation_result['issues'].append(f"Requirement not met: {requirement}")
    
    # Gather system information
    validation_result['system_info'] = _gather_system_info()
    
    return validation_result

def discover_integration_tests() -> Dict[str, Any]:
    """
    Discover all available integration tests.
    
    Returns:
        Dict: Test discovery results
    """
    discovery_result = {
        'total_test_classes': 0,
        'total_test_methods': 0,
        'categories': {},
        'complexity_levels': {},
        'test_files': [],
        'missing_implementations': []
    }
    
    # Discover test files
    integration_dir = Path(__file__).parent
    test_files = list(integration_dir.glob('test_*.py'))
    discovery_result['test_files'] = [f.name for f in test_files]
    
    # Count test classes and methods
    for category, test_classes in INTEGRATION_TEST_CATEGORIES.items():
        category_info = {
            'test_classes': len(test_classes),
            'implemented': 0,
            'missing': []
        }
        
        for test_class_name in test_classes:
            if test_class_name in globals():
                test_class = globals()[test_class_name]
                if isinstance(test_class, type) and issubclass(test_class, unittest.TestCase):
                    category_info['implemented'] += 1
                    discovery_result['total_test_classes'] += 1
                    
                    # Count test methods
                    test_methods = [method for method in dir(test_class) 
                                  if method.startswith('test_')]
                    discovery_result['total_test_methods'] += len(test_methods)
                else:
                    category_info['missing'].append(test_class_name)
            else:
                category_info['missing'].append(test_class_name)
        
        discovery_result['categories'][category] = category_info
        discovery_result['missing_implementations'].extend(category_info['missing'])
    
    # Count by complexity
    for complexity, test_classes in TEST_COMPLEXITY.items():
        implemented = sum(1 for tc in test_classes if tc in globals())
        discovery_result['complexity_levels'][complexity] = {
            'total': len(test_classes),
            'implemented': implemented,
            'coverage': implemented / len(test_classes) * 100
        }
    
    return discovery_result

def generate_integration_test_report(output_file: str = None) -> str:
    """
    Generate comprehensive integration test report.
    
    Args:
        output_file: Optional output file path
        
    Returns:
        str: Report content
    """
    discovery = discover_integration_tests()
    
    # Run all tests to get execution results
    print("Running all integration tests for report generation...")
    test_result = run_integration_tests(verbosity=1)
    
    # Generate report content
    report_lines = [
        "FIX Trading System - Integration Test Report",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Test Discovery Summary:",
        f"  Total Test Classes: {discovery['total_test_classes']}",
        f"  Total Test Methods: {discovery['total_test_methods']}",
        f"  Test Files: {len(discovery['test_files'])}",
        f"  Missing Implementations: {len(discovery['missing_implementations'])}",
        "",
        "Test Execution Summary:",
        f"  Tests Run: {test_result.testsRun}",
        f"  Passed: {test_result.testsRun - len(test_result.failures) - len(test_result.errors)}",
        f"  Failed: {len(test_result.failures)}",
        f"  Errors: {len(test_result.errors)}",
        f"  Success Rate: {(test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / max(test_result.testsRun, 1) * 100:.1f}%",
        ""
    ]
    
    # Category breakdown
    report_lines.append("Category Breakdown:")
    for category, info in discovery['categories'].items():
        coverage = info['implemented'] / info['test_classes'] * 100 if info['test_classes'] > 0 else 0
        report_lines.append(f"  {category.title()}:")
        report_lines.append(f"    Classes: {info['implemented']}/{info['test_classes']} ({coverage:.1f}%)")
        if info['missing']:
            report_lines.append(f"    Missing: {', '.join(info['missing'])}")
    
    report_lines.append("")
    
    # Complexity breakdown
    report_lines.append("Complexity Level Coverage:")
    for complexity, info in discovery['complexity_levels'].items():
        report_lines.append(f"  {complexity.title()}: {info['implemented']}/{info['total']} ({info['coverage']:.1f}%)")
    
    report_lines.append("")
    
    # Add failure/error details
    if test_result.failures:
        report_lines.append("Test Failures:")
        for i, (test, traceback) in enumerate(test_result.failures, 1):
            report_lines.append(f"  {i}. {test}")
            # Add brief error description
            lines = traceback.split('\n')
            for line in lines:
                if 'AssertionError' in line or 'AssertError' in line:
                    report_lines.append(f"     {line.strip()}")
                    break
        report_lines.append("")
    
    if test_result.errors:
        report_lines.append("Test Errors:")
        for i, (test, traceback) in enumerate(test_result.errors, 1):
            report_lines.append(f"  {i}. {test}")
            # Add brief error description
            lines = traceback.split('\n')
            for line in lines:
                if 'Error:' in line or 'Exception:' in line:
                    report_lines.append(f"     {line.strip()}")
                    break
        report_lines.append("")
    
    # Recommendations
    report_lines.append("Recommendations:")
    if discovery['missing_implementations']:
        report_lines.append(f"  - Implement {len(discovery['missing_implementations'])} missing test classes")
    
    success_rate = (test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / max(test_result.testsRun, 1)
    if success_rate < 0.95:
        report_lines.append("  - Improve test reliability (target: 95%+ success rate)")
    
    if test_result.testsRun < 50:
        report_lines.append("  - Add more comprehensive test coverage")
    
    report_content = "\n".join(report_lines)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_content)
        print(f"Integration test report saved to: {output_file}")
    
    return report_content

# Helper functions
def _print_test_summary(result: unittest.TestResult, duration: float):
    """Print test execution summary."""
    print(f"\n{'='*60}")
    print(f"Integration Test Summary")
    print(f"{'='*60}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100:.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, _ in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, _ in result.errors:
            print(f"  - {test}")

def _apply_config_override(test_config, key: str, value: Any):
    """Apply configuration override using dot notation."""
    try:
        parts = key.split('.')
        obj = test_config
        
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return  # Skip if path doesn't exist
        
        if hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], value)
    except Exception:
        pass  # Skip invalid overrides

def _check_requirement(requirement: str) -> bool:
    """Check if a specific requirement is met."""
    checks = {
        'network_connectivity': lambda: True,  # Simplified for integration tests
        'file_system_access': lambda: os.access('.', os.W_OK),
        'temp_directory': lambda: os.access(tempfile.gettempdir(), os.W_OK),
        'process_isolation': lambda: True,  # Assume available
        'threading_support': lambda: threading.active_count() >= 0,
        'mock_services': lambda: True,  # Provided by test framework
        'test_data_access': lambda: True  # Provided by test framework
    }
    
    check_func = checks.get(requirement, lambda: False)
    try:
        return check_func()
    except Exception:
        return False

def _gather_system_info() -> Dict[str, Any]:
    """Gather system information for test environment."""
    import platform
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'architecture': platform.architecture(),
        'available_memory': _get_available_memory(),
        'cpu_count': os.cpu_count(),
        'current_directory': os.getcwd(),
        'temp_directory': tempfile.gettempdir()
    }

def _get_available_memory() -> str:
    """Get available memory information."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return f"{memory.available / 1024 / 1024 / 1024:.1f} GB available of {memory.total / 1024 / 1024 / 1024:.1f} GB total"
    except ImportError:
        return "Memory information not available (psutil not installed)"

def _cleanup_test_resources():
    """Cleanup any remaining test resources."""
    # Cleanup any lingering test processes, connections, etc.
    # This is a placeholder for environment-specific cleanup
    pass

# Module initialization
def _initialize_integration_tests():
    """Initialize integration test module."""
    # Configure default logging for integration tests
    import logging
    
    # Set up basic logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress noisy loggers during tests
    logging.getLogger('quickfix').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

# Initialize the module
_initialize_integration_tests()

# Command-line interface
def main():
    """Main entry point for running integration tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FIX Trading System Integration Tests')
    parser.add_argument('--category', '-c', choices=list(INTEGRATION_TEST_CATEGORIES.keys()),
                       action='append', help='Test categories to run')
    parser.add_argument('--complexity', choices=list(TEST_COMPLEXITY.keys()),
                       help='Test complexity level to run')
    parser.add_argument('--suite', choices=['all', 'workflow', 'performance'] + list(INTEGRATION_TEST_CATEGORIES.keys()),
                       help='Test suite to run')
    parser.add_argument('--parallel', action='store_true',
                       help='Run tests in parallel (experimental)')
    parser.add_argument('--report', help='Generate test report to file')
    parser.add_argument('--discover', action='store_true',
                       help='Discover available tests')
    parser.add_argument('--verbose', '-v', action='count', default=2,
                       help='Increase verbosity')
    
    args = parser.parse_args()
    
    try:
        if args.discover:
            # Discover and display available tests
            discovery = discover_integration_tests()
            print("Integration Test Discovery Results:")
            print(f"  Total Test Classes: {discovery['total_test_classes']}")
            print(f"  Total Test Methods: {discovery['total_test_methods']}")
            print(f"  Categories: {len(discovery['categories'])}")
            
            for category, info in discovery['categories'].items():
                print(f"    {category}: {info['implemented']}/{info['test_classes']} implemented")
            
            if discovery['missing_implementations']:
                print(f"  Missing: {', '.join(discovery['missing_implementations'])}")
            
            return 0
        
        elif args.report:
            # Generate test report
            report = generate_integration_test_report(args.report)
            return 0
        
        elif args.suite:
            # Run specific test suite
            result = run_test_suite(args.suite, verbosity=args.verbose, parallel=args.parallel)
            return 0 if result.wasSuccessful() else 1
        
        else:
            # Run tests with specified filters
            result = run_integration_tests(
                categories=args.category,
                complexity=args.complexity,
                parallel=args.parallel,
                verbosity=args.verbose
            )
            return 0 if result.wasSuccessful() else 1
    
    except KeyboardInterrupt:
        print("\nIntegration test execution interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running integration tests: {e}")
        return 1

if __name__ == '__main__':
    exit(main())

# Version information
VERSION_INFO = {
    'version': __version__,
    'test_categories': len(INTEGRATION_TEST_CATEGORIES),
    'total_test_classes': sum(len(tests) for tests in INTEGRATION_TEST_CATEGORIES.values()),
    'complexity_levels': len(TEST_COMPLEXITY),
    'features': [
        'Session workflow testing',
        'Order lifecycle validation',
        'Market data integration',
        'Protocol compliance checking',
        'Performance benchmarking',
        'External system integration',
        'Error recovery testing',
        'Concurrent session handling'
    ]
}

def get_version_info():
    """Get detailed version information."""
    return VERSION_INFO.copy()

# Integration test templates and examples
INTEGRATION_TEST_TEMPLATES = {
    'basic_workflow': {
        'description': 'Basic session and order workflow test',
        'categories': ['workflow', 'trading'],
        'complexity': 'basic',
        'estimated_duration': '2-5 minutes',
        'example_classes': ['TestSessionEstablishment', 'TestOrderPlacement']
    },
    'performance_suite': {
        'description': 'Performance and load testing suite',
        'categories': ['performance'],
        'complexity': 'advanced',
        'estimated_duration': '10-30 minutes',
        'example_classes': ['TestThroughputScenarios', 'TestLoadTesting']
    },
    'compliance_validation': {
        'description': 'Protocol compliance and regulatory testing',
        'categories': ['protocol'],
        'complexity': 'intermediate',
        'estimated_duration': '5-15 minutes',
        'example_classes': ['TestProtocolCompliance', 'TestTimingCompliance']
    },
    'end_to_end': {
        'description': 'Complete end-to-end system testing',
        'categories': ['system', 'workflow', 'trading'],
        'complexity': 'expert',
        'estimated_duration': '30-60 minutes',
        'example_classes': ['TestEndToEndWorkflows', 'TestSystemIntegration']
    }
}

# Test data and configuration templates
TEST_DATA_TEMPLATES = {
    'sample_orders': [
        {
            'symbol': 'AAPL',
            'side': '1',  # Buy
            'quantity': 100,
            'order_type': '2',  # Limit
            'price': 150.00,
            'time_in_force': '0'  # Day
        },
        {
            'symbol': 'MSFT',
            'side': '2',  # Sell
            'quantity': 200,
            'order_type': '1',  # Market
            'time_in_force': '3'  # IOC
        },
        {
            'symbol': 'GOOGL',
            'side': '1',  # Buy
            'quantity': 50,
            'order_type': '3',  # Stop
            'stop_price': 2800.00,
            'time_in_force': '1'  # GTC
        }
    ],
    'sample_symbols': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ'
    ],
    'sample_accounts': [
        'TEST_ACCOUNT_001',
        'TEST_ACCOUNT_002', 
        'TEST_ACCOUNT_003'
    ],
    'market_data_symbols': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'
    ]
}

# Integration test utilities
class IntegrationTestUtils:
    """Utility class for integration test helpers."""
    
    @staticmethod
    def create_test_session_config(session_type: str = 'standard') -> Dict[str, Any]:
        """Create test session configuration."""
        base_config = {
            'sender_comp_id': 'INT_TEST_SENDER',
            'target_comp_id': 'INT_TEST_TARGET',
            'begin_string': 'FIX.4.4',
            'heartbeat_interval': 30,
            'host': 'localhost',
            'port': 19876
        }
        
        if session_type == 'performance':
            base_config.update({
                'heartbeat_interval': 10,
                'message_queue_size': 50000,
                'worker_threads': 8,
                'low_latency_mode': True
            })
        elif session_type == 'compliance':
            base_config.update({
                'strict_validation': True,
                'audit_enabled': True,
                'sequence_validation': True
            })
        
        return base_config
    
    @staticmethod
    def generate_test_orders(count: int = 10, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Generate test order data."""
        import random
        
        if symbols is None:
            symbols = TEST_DATA_TEMPLATES['sample_symbols'][:5]
        
        orders = []
        for i in range(count):
            order = {
                'cl_ord_id': f'INT_TEST_ORDER_{i+1:03d}',
                'symbol': random.choice(symbols),
                'side': random.choice(['1', '2']),  # Buy/Sell
                'quantity': random.randint(100, 1000),
                'order_type': random.choice(['1', '2']),  # Market/Limit
                'time_in_force': random.choice(['0', '1', '3'])  # Day/GTC/IOC
            }
            
            if order['order_type'] == '2':  # Limit order needs price
                base_price = random.uniform(50, 500)
                order['price'] = round(base_price, 2)
            
            orders.append(order)
        
        return orders
    
    @staticmethod
    def create_market_data_scenario(symbols: List[str], duration_minutes: int = 5) -> Dict[str, Any]:
        """Create market data test scenario."""
        import random
        
        scenario = {
            'symbols': symbols,
            'duration_minutes': duration_minutes,
            'update_frequency_ms': 100,
            'price_volatility': 0.02,  # 2%
            'updates': []
        }
        
        # Generate price updates
        base_prices = {symbol: random.uniform(50, 500) for symbol in symbols}
        
        updates_per_minute = 60000 // scenario['update_frequency_ms']
        total_updates = duration_minutes * updates_per_minute
        
        for i in range(total_updates):
            for symbol in symbols:
                current_price = base_prices[symbol]
                
                # Random price movement
                change = random.gauss(0, scenario['price_volatility'])
                new_price = current_price * (1 + change)
                base_prices[symbol] = max(new_price, 1.0)  # Minimum price of $1
                
                # Create bid/ask spread
                spread = new_price * 0.001  # 0.1% spread
                bid_price = new_price - spread / 2
                ask_price = new_price + spread / 2
                
                update = {
                    'timestamp': i * scenario['update_frequency_ms'],
                    'symbol': symbol,
                    'bid_price': round(bid_price, 2),
                    'bid_size': random.randint(100, 5000),
                    'ask_price': round(ask_price, 2),
                    'ask_size': random.randint(100, 5000)
                }
                scenario['updates'].append(update)
        
        return scenario
    
    @staticmethod
    def validate_test_results(results: Dict[str, Any], expectations: Dict[str, Any]) -> Dict[str, Any]:
        """Validate test results against expectations."""
        validation = {
            'passed': True,
            'failures': [],
            'warnings': [],
            'summary': {}
        }
        
        # Check basic metrics
        for metric, expected_value in expectations.items():
            if metric in results:
                actual_value = results[metric]
                
                if isinstance(expected_value, dict):
                    # Range or complex validation
                    if 'min' in expected_value and actual_value < expected_value['min']:
                        validation['failures'].append(f"{metric}: {actual_value} below minimum {expected_value['min']}")
                        validation['passed'] = False
                    elif 'max' in expected_value and actual_value > expected_value['max']:
                        validation['failures'].append(f"{metric}: {actual_value} above maximum {expected_value['max']}")
                        validation['passed'] = False
                    elif 'target' in expected_value:
                        tolerance = expected_value.get('tolerance', 0.1)
                        if abs(actual_value - expected_value['target']) > tolerance:
                            validation['warnings'].append(f"{metric}: {actual_value} differs from target {expected_value['target']}")
                else:
                    # Direct comparison
                    if actual_value != expected_value:
                        validation['failures'].append(f"{metric}: expected {expected_value}, got {actual_value}")
                        validation['passed'] = False
            else:
                validation['failures'].append(f"Missing result metric: {metric}")
                validation['passed'] = False
        
        # Generate summary
        validation['summary'] = {
            'total_checks': len(expectations),
            'passed_checks': len(expectations) - len(validation['failures']),
            'failed_checks': len(validation['failures']),
            'warnings': len(validation['warnings'])
        }
        
        return validation

# Export utility class
__all__.append('IntegrationTestUtils')

# Documentation and usage examples
USAGE_EXAMPLES = """
Integration Tests Usage Examples:

1. Run all integration tests:
   >>> from pyfixtest.tests.integration import run_integration_tests
   >>> result = run_integration_tests()

2. Run specific category tests:
   >>> result = run_integration_tests(categories=['workflow', 'trading'])

3. Run tests by complexity:
   >>> result = run_integration_tests(complexity='basic')

4. Run workflow tests:
   >>> from pyfixtest.tests.integration import run_workflow_tests
   >>> result = run_workflow_tests()

5. Run performance tests:
   >>> from pyfixtest.tests.integration import run_performance_tests
   >>> result = run_performance_tests()

6. Generate test report:
   >>> from pyfixtest.tests.integration import generate_integration_test_report
   >>> report = generate_integration_test_report('integration_report.txt')

7. Discover available tests:
   >>> from pyfixtest.tests.integration import discover_integration_tests
   >>> discovery = discover_integration_tests()

8. Create test environment:
   >>> from pyfixtest.tests.integration import setup_integration_environment
   >>> env = setup_integration_environment({'environment': 'performance'})
   >>> # ... run tests ...
   >>> cleanup_integration_environment(env)

Command Line Usage:

1. Run all tests:
   python -m pyfixtest.tests.integration

2. Run specific category:
   python -m pyfixtest.tests.integration --category workflow

3. Run by complexity:
   python -m pyfixtest.tests.integration --complexity basic

4. Generate report:
   python -m pyfixtest.tests.integration --report integration_report.txt

5. Discover tests:
   python -m pyfixtest.tests.integration --discover

Available Categories: {categories}
Available Complexity Levels: {complexity_levels}
Available Test Templates: {templates}
""".format(
    categories=', '.join(INTEGRATION_TEST_CATEGORIES.keys()),
    complexity_levels=', '.join(TEST_COMPLEXITY.keys()),
    templates=', '.join(INTEGRATION_TEST_TEMPLATES.keys())
)

# Add usage documentation to module docstring
__doc__ += USAGE_EXAMPLES

# Exception classes for integration testing
class IntegrationTestError(Exception):
    """Base exception for integration test errors."""
    pass

class EnvironmentSetupError(IntegrationTestError):
    """Exception raised during test environment setup."""
    pass

class TestConfigurationError(IntegrationTestError):
    """Exception raised for test configuration errors."""
    pass

class TestExecutionError(IntegrationTestError):
    """Exception raised during test execution."""
    pass

class TestValidationError(IntegrationTestError):
    """Exception raised during test result validation."""
    pass

# Add exception classes to exports
__all__.extend([
    'IntegrationTestError',
    'EnvironmentSetupError', 
    'TestConfigurationError',
    'TestExecutionError',
    'TestValidationError'
])

# Integration test metrics and monitoring
class IntegrationTestMetrics:
    """Tracks metrics for integration test execution."""
    
    def __init__(self):
        self.metrics = {
            'test_execution_times': {},
            'setup_times': {},
            'teardown_times': {},
            'resource_usage': {},
            'error_rates': {},
            'success_rates': {}
        }
        self.start_time = None
        self.end_time = None
    
    def start_test_session(self):
        """Start tracking test session metrics."""
        self.start_time = time.time()
    
    def end_test_session(self):
        """End tracking test session metrics."""
        self.end_time = time.time()
    
    def record_test_time(self, test_name: str, execution_time: float):
        """Record test execution time."""
        if test_name not in self.metrics['test_execution_times']:
            self.metrics['test_execution_times'][test_name] = []
        self.metrics['test_execution_times'][test_name].append(execution_time)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {
            'session_duration': self.end_time - self.start_time if self.end_time and self.start_time else 0,
            'total_tests': sum(len(times) for times in self.metrics['test_execution_times'].values()),
            'average_test_time': 0,
            'slowest_tests': [],
            'fastest_tests': []
        }
        
        # Calculate average test time
        all_times = []
        for times in self.metrics['test_execution_times'].values():
            all_times.extend(times)
        
        if all_times:
            summary['average_test_time'] = sum(all_times) / len(all_times)
            
            # Find slowest and fastest tests
            test_averages = []
            for test_name, times in self.metrics['test_execution_times'].items():
                avg_time = sum(times) / len(times)
                test_averages.append((test_name, avg_time))
            
            test_averages.sort(key=lambda x: x[1])
            summary['fastest_tests'] = test_averages[:3]
            summary['slowest_tests'] = test_averages[-3:]
        
        return summary

# Global metrics instance
integration_metrics = IntegrationTestMetrics()

# Add metrics to exports
__all__.append('integration_metrics')

# Final module validation
def _validate_module_integrity():
    """Validate module integrity and completeness."""
    validation_issues = []
    
    # Check that all exported classes are actually defined
    for export_name in __all__:
        if export_name.startswith('Test') and export_name not in globals():
            validation_issues.append(f"Exported test class not implemented: {export_name}")
    
    # Check test category consistency
    all_category_tests = set()
    for tests in INTEGRATION_TEST_CATEGORIES.values():
        all_category_tests.update(tests)
    
    exported_tests = {name for name in __all__ if name.startswith('Test')}
    missing_from_categories = exported_tests - all_category_tests
    if missing_from_categories:
        validation_issues.append(f"Tests not categorized: {missing_from_categories}")
    
    # Log validation results
    if validation_issues:
        import logging
        logger = logging.getLogger(__name__)
        for issue in validation_issues:
            logger.warning(f"Module integrity issue: {issue}")
    
    return len(validation_issues) == 0

# Validate module on import
_validate_module_integrity()