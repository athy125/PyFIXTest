"""
Test configuration and test environment management for FIX trading system.

This module provides comprehensive test configuration management, including
mock configurations, test environment setup, and test-specific settings
for different types of testing scenarios.
"""

import os
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
import uuid

from ..utils.logging_config import get_logger
from .fix_config import FIXConfig, SessionConfig, NetworkConfig, SecurityConfig, PerformanceConfig


@dataclass
class MockConfig:
    """Configuration for mock services and test doubles."""
    
    # Mock service settings
    enable_mock_exchange: bool = True
    enable_mock_market_data: bool = True
    enable_mock_risk_system: bool = True
    enable_mock_database: bool = True
    enable_mock_external_apis: bool = True
    
    # Mock behavior settings
    simulate_network_latency: bool = True
    network_latency_ms: int = 10
    simulate_message_loss: bool = False
    message_loss_rate: float = 0.001  # 0.1%
    simulate_disconnections: bool = False
    disconnection_rate: float = 0.0001  # 0.01%
    
    # Mock exchange settings
    exchange_host: str = "localhost"
    exchange_port: int = 19876
    exchange_auto_fill: bool = True
    exchange_fill_delay_ms: int = 50
    exchange_reject_rate: float = 0.01  # 1%
    exchange_partial_fill_rate: float = 0.1  # 10%
    
    # Mock market data settings
    market_data_symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"
    ])
    market_data_update_frequency_ms: int = 100
    market_data_price_volatility: float = 0.02  # 2%
    market_data_spread_bps: int = 5  # 5 basis points
    
    # Mock database settings
    database_type: str = "sqlite"  # sqlite, memory, postgres_mock
    database_file: str = ":memory:"
    database_auto_create_schema: bool = True
    database_populate_test_data: bool = True
    
    # Error simulation
    simulate_errors: bool = True
    error_rate: float = 0.001  # 0.1%
    error_types: List[str] = field(default_factory=lambda: [
        "TIMEOUT", "INVALID_MESSAGE", "SEQUENCE_ERROR", "BUSINESS_REJECT"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mock_services': {
                'enable_mock_exchange': self.enable_mock_exchange,
                'enable_mock_market_data': self.enable_mock_market_data,
                'enable_mock_risk_system': self.enable_mock_risk_system,
                'enable_mock_database': self.enable_mock_database,
                'enable_mock_external_apis': self.enable_mock_external_apis
            },
            'mock_behavior': {
                'simulate_network_latency': self.simulate_network_latency,
                'network_latency_ms': self.network_latency_ms,
                'simulate_message_loss': self.simulate_message_loss,
                'message_loss_rate': self.message_loss_rate,
                'simulate_disconnections': self.simulate_disconnections,
                'disconnection_rate': self.disconnection_rate
            },
            'mock_exchange': {
                'host': self.exchange_host,
                'port': self.exchange_port,
                'auto_fill': self.exchange_auto_fill,
                'fill_delay_ms': self.exchange_fill_delay_ms,
                'reject_rate': self.exchange_reject_rate,
                'partial_fill_rate': self.exchange_partial_fill_rate
            },
            'mock_market_data': {
                'symbols': self.market_data_symbols,
                'update_frequency_ms': self.market_data_update_frequency_ms,
                'price_volatility': self.market_data_price_volatility,
                'spread_bps': self.market_data_spread_bps
            },
            'mock_database': {
                'type': self.database_type,
                'file': self.database_file,
                'auto_create_schema': self.database_auto_create_schema,
                'populate_test_data': self.database_populate_test_data
            },
            'error_simulation': {
                'simulate_errors': self.simulate_errors,
                'error_rate': self.error_rate,
                'error_types': self.error_types
            }
        }


@dataclass
class TestSessionSettings:
    """Test-specific session settings."""
    
    # Test session identification
    test_session_prefix: str = "TEST"
    use_random_comp_ids: bool = True
    comp_id_suffix: str = ""
    
    # Test timing settings
    accelerated_time: bool = True
    time_acceleration_factor: float = 10.0
    heartbeat_interval: int = 5  # Shorter for tests
    logon_timeout: int = 3
    logout_timeout: int = 1
    
    # Test behavior settings
    auto_login: bool = True
    auto_logout: bool = True
    reset_sequence_numbers: bool = True
    persist_messages: bool = False
    
    # Test validation settings
    strict_validation: bool = True
    validate_all_fields: bool = True
    require_admin_messages: bool = False
    
    # Test data settings
    use_test_symbols: bool = True
    test_symbols: List[str] = field(default_factory=lambda: [
        "TEST.AAPL", "TEST.MSFT", "TEST.GOOGL"
    ])
    
    def generate_test_comp_ids(self) -> tuple:
        """Generate unique test component IDs."""
        if self.use_random_comp_ids:
            unique_id = str(uuid.uuid4())[:8].upper()
            sender = f"{self.test_session_prefix}_SENDER_{unique_id}"
            target = f"{self.test_session_prefix}_TARGET_{unique_id}"
        else:
            suffix = self.comp_id_suffix or datetime.now().strftime("%H%M%S")
            sender = f"{self.test_session_prefix}_SENDER_{suffix}"
            target = f"{self.test_session_prefix}_TARGET_{suffix}"
        
        return sender, target


class TestEnvironment:
    """
    Test environment management for different types of testing.
    
    Provides isolated test environments with proper setup and cleanup,
    temporary directories, and environment-specific configurations.
    """
    
    def __init__(self, environment_type: str = "unit"):
        """
        Initialize test environment.
        
        Args:
            environment_type: Type of test environment (unit, integration, performance, load)
        """
        self.environment_type = environment_type
        self.logger = get_logger(__name__)
        
        # Environment state
        self.is_setup = False
        self.temp_dir = None
        self.created_files = []
        self.started_services = []
        self.cleanup_callbacks = []
        
        # Environment-specific settings
        self.test_timeout = self._get_default_timeout()
        self.max_memory_mb = self._get_default_memory_limit()
        self.parallel_execution = self._get_parallel_setting()
        
        # Thread safety
        self._lock = threading.Lock()
    
    def setup(self) -> bool:
        """
        Setup test environment.
        
        Returns:
            bool: True if setup successful
        """
        with self._lock:
            if self.is_setup:
                self.logger.warning("Test environment already setup")
                return True
            
            try:
                # Create temporary directory
                self.temp_dir = tempfile.mkdtemp(prefix=f"fix_test_{self.environment_type}_")
                self.logger.info(f"Created test directory: {self.temp_dir}")
                
                # Setup environment-specific resources
                if not self._setup_environment_specific():
                    return False
                
                # Setup logging
                self._setup_test_logging()
                
                # Setup mock services if needed
                if self.environment_type in ['integration', 'performance', 'load']:
                    self._setup_mock_services()
                
                self.is_setup = True
                self.logger.info(f"Test environment '{self.environment_type}' setup complete")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to setup test environment: {e}")
                self.cleanup()
                return False
    
    def cleanup(self):
        """Cleanup test environment."""
        with self._lock:
            if not self.is_setup:
                return
            
            try:
                # Run cleanup callbacks
                for callback in reversed(self.cleanup_callbacks):
                    try:
                        callback()
                    except Exception as e:
                        self.logger.warning(f"Cleanup callback error: {e}")
                
                # Stop services
                for service in reversed(self.started_services):
                    try:
                        if hasattr(service, 'stop'):
                            service.stop()
                        elif hasattr(service, 'shutdown'):
                            service.shutdown()
                    except Exception as e:
                        self.logger.warning(f"Service cleanup error: {e}")
                
                # Remove created files
                for file_path in reversed(self.created_files):
                    try:
                        if os.path.exists(file_path):
                            if os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                            else:
                                os.remove(file_path)
                    except Exception as e:
                        self.logger.warning(f"File cleanup error: {e}")
                
                # Remove temporary directory
                if self.temp_dir and os.path.exists(self.temp_dir):
                    try:
                        shutil.rmtree(self.temp_dir)
                        self.logger.info(f"Removed test directory: {self.temp_dir}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove test directory: {e}")
                
                self.is_setup = False
                self.logger.info(f"Test environment '{self.environment_type}' cleanup complete")
                
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")
    
    def get_temp_path(self, relative_path: str = "") -> str:
        """
        Get path within temporary directory.
        
        Args:
            relative_path: Relative path within temp directory
            
        Returns:
            str: Full path
        """
        if not self.temp_dir:
            raise RuntimeError("Test environment not setup")
        
        if relative_path:
            return os.path.join(self.temp_dir, relative_path)
        return self.temp_dir
    
    def create_temp_file(self, filename: str, content: str = "") -> str:
        """
        Create temporary file with content.
        
        Args:
            filename: Name of file to create
            content: File content
            
        Returns:
            str: Full path to created file
        """
        file_path = self.get_temp_path(filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        self.created_files.append(file_path)
        return file_path
    
    def create_temp_dir(self, dirname: str) -> str:
        """
        Create temporary directory.
        
        Args:
            dirname: Directory name
            
        Returns:
            str: Full path to created directory
        """
        dir_path = self.get_temp_path(dirname)
        os.makedirs(dir_path, exist_ok=True)
        self.created_files.append(dir_path)
        return dir_path
    
    def add_cleanup_callback(self, callback):
        """Add cleanup callback function."""
        self.cleanup_callbacks.append(callback)
    
    def start_service(self, service):
        """Register and start a service."""
        if hasattr(service, 'start'):
            service.start()
        elif hasattr(service, 'run'):
            service.run()
        
        self.started_services.append(service)
    
    def _get_default_timeout(self) -> int:
        """Get default timeout for environment type."""
        timeouts = {
            'unit': 30,
            'integration': 120,
            'performance': 300,
            'load': 600,
            'stress': 1800
        }
        return timeouts.get(self.environment_type, 60)
    
    def _get_default_memory_limit(self) -> int:
        """Get default memory limit for environment type."""
        limits = {
            'unit': 256,
            'integration': 512,
            'performance': 1024,
            'load': 2048,
            'stress': 4096
        }
        return limits.get(self.environment_type, 512)
    
    def _get_parallel_setting(self) -> bool:
        """Get parallel execution setting for environment type."""
        return self.environment_type in ['performance', 'load', 'stress']
    
    def _setup_environment_specific(self) -> bool:
        """Setup environment-specific resources."""
        try:
            if self.environment_type == 'unit':
                return self._setup_unit_environment()
            elif self.environment_type == 'integration':
                return self._setup_integration_environment()
            elif self.environment_type == 'performance':
                return self._setup_performance_environment()
            elif self.environment_type == 'load':
                return self._setup_load_environment()
            elif self.environment_type == 'stress':
                return self._setup_stress_environment()
            else:
                self.logger.warning(f"Unknown environment type: {self.environment_type}")
                return True
        except Exception as e:
            self.logger.error(f"Environment-specific setup failed: {e}")
            return False
    
    def _setup_unit_environment(self) -> bool:
        """Setup unit test environment."""
        # Create basic directories
        self.create_temp_dir("store")
        self.create_temp_dir("logs")
        self.create_temp_dir("data")
        return True
    
    def _setup_integration_environment(self) -> bool:
        """Setup integration test environment."""
        # Create directories with more realistic structure
        self.create_temp_dir("store/sessions")
        self.create_temp_dir("logs/application")
        self.create_temp_dir("logs/events")
        self.create_temp_dir("data/market_data")
        self.create_temp_dir("data/reference")
        self.create_temp_dir("config")
        return True
    
    def _setup_performance_environment(self) -> bool:
        """Setup performance test environment."""
        # Optimize for performance testing
        self.create_temp_dir("store")
        self.create_temp_dir("logs")
        self.create_temp_dir("metrics")
        self.create_temp_dir("profiling")
        return True
    
    def _setup_load_environment(self) -> bool:
        """Setup load test environment."""
        # Setup for high-volume testing
        self.create_temp_dir("store")
        self.create_temp_dir("logs")
        self.create_temp_dir("metrics")
        self.create_temp_dir("reports")
        return True
    
    def _setup_stress_environment(self) -> bool:
        """Setup stress test environment."""
        # Setup for stress testing
        self.create_temp_dir("store")
        self.create_temp_dir("logs")
        self.create_temp_dir("dumps")
        self.create_temp_dir("monitoring")
        return True
    
    def _setup_test_logging(self):
        """Setup test-specific logging."""
        import logging
        
        log_dir = self.get_temp_path("logs")
        log_file = os.path.join(log_dir, "test.log")
        
        # Configure test logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        
        self.add_cleanup_callback(lambda: root_logger.removeHandler(handler))
    
    def _setup_mock_services(self):
        """Setup mock services for testing."""
        # This would be implemented based on available mock services
        pass


class TestConfig:
    """
    Main test configuration class that provides comprehensive test settings
    and environment management for different types of testing scenarios.
    """
    
    def __init__(self, test_type: str = "unit", environment: str = "test"):
        """
        Initialize test configuration.
        
        Args:
            test_type: Type of test (unit, integration, performance, load, stress)
            environment: Test environment name
        """
        self.test_type = test_type
        self.environment = environment
        self.logger = get_logger(__name__)
        
        # Configuration components
        self.fix_config = FIXConfig()
        self.mock_config = MockConfig()
        self.session_settings = TestSessionSettings()
        
        # Test environment
        self.test_env = TestEnvironment(test_type)
        
        # Test metadata
        self.test_run_id = str(uuid.uuid4())
        self.created_at = datetime.now(timezone.utc)
        
        # Initialize with test-appropriate settings
        self._initialize_test_config()
    
    def setup(self) -> bool:
        """
        Setup complete test configuration and environment.
        
        Returns:
            bool: True if setup successful
        """
        try:
            # Setup test environment first
            if not self.test_env.setup():
                return False
            
            # Configure FIX settings for testing
            self._configure_fix_for_testing()
            
            # Setup test-specific directories
            self._setup_test_directories()
            
            # Generate test configuration files
            self._generate_test_config_files()
            
            self.logger.info(f"Test configuration setup complete for {self.test_type} tests")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup test configuration: {e}")
            return False
    
    def cleanup(self):
        """Cleanup test configuration and environment."""
        self.test_env.cleanup()
    
    def get_fix_config(self) -> FIXConfig:
        """Get FIX configuration for testing."""
        return self.fix_config
    
    def get_mock_config(self) -> MockConfig:
        """Get mock configuration."""
        return self.mock_config
    
    def get_test_session_config(self) -> TestSessionSettings:
        """Get test session configuration."""
        return self.session_settings
    
    def get_temp_path(self, relative_path: str = "") -> str:
        """Get temporary path for test files."""
        return self.test_env.get_temp_path(relative_path)
    
    def create_test_config_file(self, filename: str, config_type: str = "fix") -> str:
        """
        Create test configuration file.
        
        Args:
            filename: Configuration filename
            config_type: Type of configuration (fix, mock, session)
            
        Returns:
            str: Path to created file
        """
        if config_type == "fix":
            config_dict = self.fix_config.to_dict(include_sensitive=True)
        elif config_type == "mock":
            config_dict = self.mock_config.to_dict()
        elif config_type == "session":
            config_dict = {"session_settings": self.session_settings.__dict__}
        else:
            raise ValueError(f"Unknown config type: {config_type}")
        
        # Convert to YAML format
        import yaml
        content = yaml.dump(config_dict, default_flow_style=False, indent=2)
        
        return self.test_env.create_temp_file(filename, content)
    
    def generate_test_session_ids(self, count: int = 1) -> List[tuple]:
        """
        Generate test session IDs.
        
        Args:
            count: Number of session ID pairs to generate
            
        Returns:
            List[tuple]: List of (sender_comp_id, target_comp_id) pairs
        """
        session_ids = []
        for i in range(count):
            sender, target = self.session_settings.generate_test_comp_ids()
            session_ids.append((sender, target))
        return session_ids
    
    def create_test_symbols(self, count: int = 10, prefix: str = "TEST") -> List[str]:
        """
        Create test trading symbols.
        
        Args:
            count: Number of symbols to create
            prefix: Symbol prefix
            
        Returns:
            List[str]: List of test symbols
        """
        base_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]
        symbols = []
        
        for i in range(count):
            base = base_symbols[i % len(base_symbols)]
            symbol = f"{prefix}.{base}" if prefix else base
            symbols.append(symbol)
        
        return symbols
    
    def _initialize_test_config(self):
        """Initialize configuration with test-appropriate defaults."""
        # Set test environment
        self.fix_config.environment = f"test_{self.test_type}"
        self.fix_config.debug_mode = True
        
        # Configure session for testing
        self.fix_config.session.heartbeat_interval = self.session_settings.heartbeat_interval
        self.fix_config.session.logon_timeout = self.session_settings.logon_timeout
        self.fix_config.session.logout_timeout = self.session_settings.logout_timeout
        self.fix_config.session.reset_on_logon = self.session_settings.reset_sequence_numbers
        self.fix_config.session.persist_messages = self.session_settings.persist_messages
        
        # Configure network for testing
        if self.test_type in ['performance', 'load', 'stress']:
            # Optimize for performance testing
            self.fix_config.network.connect_timeout = 5
            self.fix_config.network.tcp_no_delay = True
            self.fix_config.network.socket_no_delay = True
        else:
            # Standard test settings
            self.fix_config.network.connect_timeout = 10
        
        # Configure security for testing
        self.fix_config.security.ssl_enabled = False  # Simplify for testing
        self.fix_config.security.authentication_enabled = False
        
        # Configure performance for testing
        if self.test_type == 'performance':
            self.fix_config.performance.low_latency_mode = True
            self.fix_config.performance.message_queue_size = 50000
            self.fix_config.performance.worker_threads = 8
        elif self.test_type in ['load', 'stress']:
            self.fix_config.performance.high_throughput_mode = True
            self.fix_config.performance.message_queue_size = 100000
            self.fix_config.performance.worker_threads = 16
            self.fix_config.performance.batch_processing = True
        else:
            # Unit/integration test settings
            self.fix_config.performance.message_queue_size = 1000
            self.fix_config.performance.worker_threads = 2
    
    def _configure_fix_for_testing(self):
        """Configure FIX settings specifically for testing."""
        # Set paths to test directories
        self.fix_config.store_path = self.test_env.get_temp_path("store")
        self.fix_config.log_path = self.test_env.get_temp_path("logs")
        self.fix_config.data_dictionary_path = self.test_env.get_temp_path("spec")
        
        # Generate test session IDs
        sender, target = self.session_settings.generate_test_comp_ids()
        self.fix_config.session.sender_comp_id = sender
        self.fix_config.session.target_comp_id = target
        
        # Set test-appropriate network settings
        if self.mock_config.enable_mock_exchange:
            self.fix_config.session.socket_connect_host = self.mock_config.exchange_host
            self.fix_config.session.socket_connect_port = self.mock_config.exchange_port
        else:
            self.fix_config.session.socket_connect_host = "localhost"
            self.fix_config.session.socket_connect_port = 19876
    
    def _setup_test_directories(self):
        """Setup test-specific directory structure."""
        # Create FIX-specific directories
        self.test_env.create_temp_dir("store")
        self.test_env.create_temp_dir("logs")
        self.test_env.create_temp_dir("spec")
        self.test_env.create_temp_dir("config")
        
        # Create test data directories
        self.test_env.create_temp_dir("data/orders")
        self.test_env.create_temp_dir("data/executions")
        self.test_env.create_temp_dir("data/market_data")
        
        # Create test result directories
        self.test_env.create_temp_dir("results")
        self.test_env.create_temp_dir("reports")
        self.test_env.create_temp_dir("metrics")
    
    def _generate_test_config_files(self):
        """Generate test configuration files."""
        # Create main FIX configuration
        fix_config_file = self.create_test_config_file("fix_config.yaml", "fix")
        self.logger.info(f"Created FIX config: {fix_config_file}")
        
        # Create mock configuration
        mock_config_file = self.create_test_config_file("mock_config.yaml", "mock")
        self.logger.info(f"Created mock config: {mock_config_file}")
        
        # Create session configuration
        session_config_file = self.create_test_config_file("session_config.yaml", "session")
        self.logger.info(f"Created session config: {session_config_file}")
        
        # Create QuickFIX configuration file
        self._create_quickfix_config_file()
    
    def _create_quickfix_config_file(self):
        """Create QuickFIX-compatible configuration file."""
        quickfix_config = f"""[DEFAULT]
ConnectionType=initiator
ReconnectInterval={self.fix_config.session.reconnect_interval}
FileStorePath={self.fix_config.store_path}
FileLogPath={self.fix_config.log_path}
StartTime={self.fix_config.session.start_time}
EndTime={self.fix_config.session.end_time}
UseDataDictionary=N
LogonTimeout={self.fix_config.session.logon_timeout}
LogoutTimeout={self.fix_config.session.logout_timeout}

[SESSION]
BeginString={self.fix_config.session.begin_string}
SenderCompID={self.fix_config.session.sender_comp_id}
TargetCompID={self.fix_config.session.target_comp_id}
SocketConnectHost={self.fix_config.session.socket_connect_host}
SocketConnectPort={self.fix_config.session.socket_connect_port}
HeartBtInt={self.fix_config.session.heartbeat_interval}
ResetOnLogon={"Y" if self.fix_config.session.reset_on_logon else "N"}
ResetOnLogout={"Y" if self.fix_config.session.reset_on_logout else "N"}
ResetOnDisconnect={"Y" if self.fix_config.session.reset_on_disconnect else "N"}
"""
        
        config_file = self.test_env.create_temp_file("quickfix.cfg", quickfix_config)
        self.logger.info(f"Created QuickFIX config: {config_file}")
        return config_file
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test configuration to dictionary."""
        return {
            'test_metadata': {
                'test_type': self.test_type,
                'environment': self.environment,
                'test_run_id': self.test_run_id,
                'created_at': self.created_at.isoformat()
            },
            'fix_config': self.fix_config.to_dict(include_sensitive=True),
            'mock_config': self.mock_config.to_dict(),
            'session_settings': {
                'test_session_prefix': self.session_settings.test_session_prefix,
                'use_random_comp_ids': self.session_settings.use_random_comp_ids,
                'accelerated_time': self.session_settings.accelerated_time,
                'time_acceleration_factor': self.session_settings.time_acceleration_factor,
                'heartbeat_interval': self.session_settings.heartbeat_interval,
                'auto_login': self.session_settings.auto_login,
                'auto_logout': self.session_settings.auto_logout,
                'strict_validation': self.session_settings.strict_validation,
                'use_test_symbols': self.session_settings.use_test_symbols,
                'test_symbols': self.session_settings.test_symbols
            },
            'test_environment': {
                'type': self.test_env.environment_type,
                'temp_dir': self.test_env.temp_dir,
                'test_timeout': self.test_env.test_timeout,
                'max_memory_mb': self.test_env.max_memory_mb,
                'parallel_execution': self.test_env.parallel_execution
            }
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Predefined test configurations for different scenarios
def create_unit_test_config() -> TestConfig:
    """Create configuration for unit tests."""
    config = TestConfig("unit", "test")
    
    # Minimal settings for fast unit tests
    config.mock_config.enable_mock_exchange = True
    config.mock_config.enable_mock_market_data = True
    config.mock_config.simulate_network_latency = False
    config.mock_config.simulate_errors = False
    
    config.session_settings.heartbeat_interval = 5
    config.session_settings.auto_login = True
    config.session_settings.reset_sequence_numbers = True
    
    return config


def create_integration_test_config() -> TestConfig:
    """Create configuration for integration tests."""
    config = TestConfig("integration", "test")
    
    # More realistic settings for integration tests
    config.mock_config.enable_mock_exchange = True
    config.mock_config.enable_mock_market_data = True
    config.mock_config.simulate_network_latency = True
    config.mock_config.network_latency_ms = 20
    config.mock_config.simulate_errors = True
    config.mock_config.error_rate = 0.001
    
    config.session_settings.heartbeat_interval = 10
    config.session_settings.auto_login = True
    config.session_settings.strict_validation = True
    
    return config


def create_performance_test_config() -> TestConfig:
    """Create configuration for performance tests."""
    config = TestConfig("performance", "test")
    
    # Optimized settings for performance testing
    config.mock_config.enable_mock_exchange = True
    config.mock_config.enable_mock_market_data = True
    config.mock_config.simulate_network_latency = True
    config.mock_config.network_latency_ms = 5
    config.mock_config.exchange_fill_delay_ms = 10
    config.mock_config.simulate_errors = False
    
    config.session_settings.heartbeat_interval = 30
    config.session_settings.accelerated_time = False
    config.session_settings.strict_validation = False
    
    # Performance optimizations
    config.fix_config.performance.low_latency_mode = True
    config.fix_config.performance.message_queue_size = 50000
    config.fix_config.performance.worker_threads = 8
    config.fix_config.performance.batch_processing = True
    config.fix_config.performance.cache_enabled = True
    
    return config


def create_load_test_config() -> TestConfig:
    """Create configuration for load tests."""
    config = TestConfig("load", "test")
    
    # High-throughput settings for load testing
    config.mock_config.enable_mock_exchange = True
    config.mock_config.enable_mock_market_data = True
    config.mock_config.simulate_network_latency = True
    config.mock_config.network_latency_ms = 10
    config.mock_config.exchange_auto_fill = True
    config.mock_config.exchange_fill_delay_ms = 5
    config.mock_config.simulate_errors = True
    config.mock_config.error_rate = 0.0001  # Lower error rate for load tests
    
    config.session_settings.heartbeat_interval = 30
    config.session_settings.accelerated_time = False
    config.session_settings.strict_validation = False
    
    # High-throughput optimizations
    config.fix_config.performance.high_throughput_mode = True
    config.fix_config.performance.message_queue_size = 100000
    config.fix_config.performance.worker_threads = 16
    config.fix_config.performance.batch_processing = True
    config.fix_config.performance.batch_size = 500
    config.fix_config.performance.async_processing = True
    config.fix_config.performance.compression_enabled = True
    
    return config


def create_stress_test_config() -> TestConfig:
    """Create configuration for stress tests."""
    config = TestConfig("stress", "test")
    
    # Extreme settings for stress testing
    config.mock_config.enable_mock_exchange = True
    config.mock_config.enable_mock_market_data = True
    config.mock_config.simulate_network_latency = True
    config.mock_config.network_latency_ms = 50
    config.mock_config.simulate_message_loss = True
    config.mock_config.message_loss_rate = 0.01
    config.mock_config.simulate_disconnections = True
    config.mock_config.disconnection_rate = 0.001
    config.mock_config.simulate_errors = True
    config.mock_config.error_rate = 0.01
    
    config.session_settings.heartbeat_interval = 60
    config.session_settings.accelerated_time = False
    config.session_settings.strict_validation = True
    
    # Resource limits for stress testing
    config.fix_config.performance.message_queue_size = 200000
    config.fix_config.performance.worker_threads = 32
    config.fix_config.performance.max_memory_usage_mb = 4096
    config.fix_config.performance.gc_enabled = True
    config.fix_config.performance.gc_interval_seconds = 30
    
    return config


def create_compliance_test_config() -> TestConfig:
    """Create configuration for compliance/regulatory tests."""
    config = TestConfig("integration", "compliance")
    
    # Strict compliance settings
    config.mock_config.enable_mock_exchange = True
    config.mock_config.enable_mock_market_data = True
    config.mock_config.simulate_network_latency = True
    config.mock_config.network_latency_ms = 100  # Realistic latency
    config.mock_config.simulate_errors = True
    config.mock_config.error_rate = 0.005  # Higher error rate for robustness
    
    config.session_settings.heartbeat_interval = 30
    config.session_settings.strict_validation = True
    config.session_settings.validate_all_fields = True
    config.session_settings.require_admin_messages = True
    
    # Compliance-focused settings
    config.fix_config.security.audit_enabled = True
    config.fix_config.security.compliance_mode = "MIFID_II"
    config.fix_config.session.validate_length_and_checksum = True
    config.fix_config.session.validate_fields_out_of_order = True
    config.fix_config.session.validate_fields_have_values = True
    config.fix_config.session.validate_user_defined_fields = True
    
    return config


class TestConfigFactory:
    """Factory for creating test configurations."""
    
    @staticmethod
    def create_config(test_type: str, **kwargs) -> TestConfig:
        """
        Create test configuration by type.
        
        Args:
            test_type: Type of test configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            TestConfig: Configured test configuration
        """
        if test_type == "unit":
            config = create_unit_test_config()
        elif test_type == "integration":
            config = create_integration_test_config()
        elif test_type == "performance":
            config = create_performance_test_config()
        elif test_type == "load":
            config = create_load_test_config()
        elif test_type == "stress":
            config = create_stress_test_config()
        elif test_type == "compliance":
            config = create_compliance_test_config()
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Apply any custom settings
        if kwargs:
            TestConfigFactory._apply_custom_settings(config, kwargs)
        
        return config
    
    @staticmethod
    def _apply_custom_settings(config: TestConfig, settings: Dict[str, Any]):
        """Apply custom settings to test configuration."""
        for key, value in settings.items():
            if '.' in key:
                # Handle nested settings (e.g., 'mock_config.network_latency_ms')
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        break
                else:
                    if hasattr(obj, parts[-1]):
                        setattr(obj, parts[-1], value)
            else:
                # Handle top-level settings
                if hasattr(config, key):
                    setattr(config, key, value)


class TestConfigValidator:
    """Validator for test configurations."""
    
    @staticmethod
    def validate(config: TestConfig) -> List[str]:
        """
        Validate test configuration.
        
        Args:
            config: Test configuration to validate
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Validate FIX configuration
        fix_errors = config.fix_config.validate()
        errors.extend([f"FIX: {error}" for error in fix_errors])
        
        # Validate mock configuration
        mock_errors = TestConfigValidator._validate_mock_config(config.mock_config)
        errors.extend([f"Mock: {error}" for error in mock_errors])
        
        # Validate session settings
        session_errors = TestConfigValidator._validate_session_settings(config.session_settings)
        errors.extend([f"Session: {error}" for error in session_errors])
        
        # Validate test environment compatibility
        env_errors = TestConfigValidator._validate_environment_compatibility(config)
        errors.extend([f"Environment: {error}" for error in env_errors])
        
        return errors
    
    @staticmethod
    def _validate_mock_config(mock_config: MockConfig) -> List[str]:
        """Validate mock configuration."""
        errors = []
        
        if mock_config.network_latency_ms < 0:
            errors.append("Network latency cannot be negative")
        
        if not 0 <= mock_config.message_loss_rate <= 1:
            errors.append("Message loss rate must be between 0 and 1")
        
        if not 0 <= mock_config.disconnection_rate <= 1:
            errors.append("Disconnection rate must be between 0 and 1")
        
        if not 1 <= mock_config.exchange_port <= 65535:
            errors.append("Exchange port must be between 1 and 65535")
        
        if mock_config.exchange_fill_delay_ms < 0:
            errors.append("Exchange fill delay cannot be negative")
        
        if not 0 <= mock_config.exchange_reject_rate <= 1:
            errors.append("Exchange reject rate must be between 0 and 1")
        
        if not 0 <= mock_config.exchange_partial_fill_rate <= 1:
            errors.append("Exchange partial fill rate must be between 0 and 1")
        
        if mock_config.market_data_update_frequency_ms <= 0:
            errors.append("Market data update frequency must be positive")
        
        if mock_config.market_data_price_volatility < 0:
            errors.append("Market data price volatility cannot be negative")
        
        if not 0 <= mock_config.error_rate <= 1:
            errors.append("Error rate must be between 0 and 1")
        
        return errors
    
    @staticmethod
    def _validate_session_settings(session_settings: TestSessionSettings) -> List[str]:
        """Validate session settings."""
        errors = []
        
        if session_settings.time_acceleration_factor <= 0:
            errors.append("Time acceleration factor must be positive")
        
        if session_settings.heartbeat_interval <= 0:
            errors.append("Heartbeat interval must be positive")
        
        if session_settings.logon_timeout <= 0:
            errors.append("Logon timeout must be positive")
        
        if session_settings.logout_timeout <= 0:
            errors.append("Logout timeout must be positive")
        
        if not session_settings.test_session_prefix:
            errors.append("Test session prefix cannot be empty")
        
        return errors
    
    @staticmethod
    def _validate_environment_compatibility(config: TestConfig) -> List[str]:
        """Validate environment compatibility."""
        errors = []
        
        test_type = config.test_type
        
        # Validate performance settings for test type
        if test_type == "performance":
            if not config.fix_config.performance.low_latency_mode:
                errors.append("Performance tests should enable low latency mode")
            
            if config.mock_config.simulate_errors:
                errors.append("Performance tests should disable error simulation")
        
        elif test_type == "load":
            if not config.fix_config.performance.high_throughput_mode:
                errors.append("Load tests should enable high throughput mode")
            
            if config.fix_config.performance.message_queue_size < 50000:
                errors.append("Load tests should use larger message queue")
        
        elif test_type == "stress":
            if config.fix_config.performance.max_memory_usage_mb < 2048:
                errors.append("Stress tests should allow higher memory usage")
        
        # Validate mock service compatibility
        if test_type in ["performance", "load"] and config.mock_config.simulate_message_loss:
            errors.append("Performance/load tests should not simulate message loss")
        
        return errors


class TestConfigManager:
    """Manager for test configuration lifecycle."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.active_configs: Dict[str, TestConfig] = {}
        self._lock = threading.Lock()
    
    def create_config(self, test_id: str, test_type: str, **kwargs) -> TestConfig:
        """
        Create and register test configuration.
        
        Args:
            test_id: Unique test identifier
            test_type: Type of test configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            TestConfig: Created test configuration
        """
        with self._lock:
            if test_id in self.active_configs:
                raise ValueError(f"Test configuration '{test_id}' already exists")
            
            config = TestConfigFactory.create_config(test_type, **kwargs)
            
            # Validate configuration
            errors = TestConfigValidator.validate(config)
            if errors:
                raise ValueError(f"Invalid test configuration: {errors}")
            
            # Setup configuration
            if not config.setup():
                raise RuntimeError(f"Failed to setup test configuration '{test_id}'")
            
            self.active_configs[test_id] = config
            self.logger.info(f"Created test configuration '{test_id}' of type '{test_type}'")
            
            return config
    
    def get_config(self, test_id: str) -> Optional[TestConfig]:
        """Get test configuration by ID."""
        with self._lock:
            return self.active_configs.get(test_id)
    
    def cleanup_config(self, test_id: str):
        """Cleanup test configuration."""
        with self._lock:
            if test_id in self.active_configs:
                config = self.active_configs[test_id]
                config.cleanup()
                del self.active_configs[test_id]
                self.logger.info(f"Cleaned up test configuration '{test_id}'")
    
    def cleanup_all(self):
        """Cleanup all active test configurations."""
        with self._lock:
            for test_id in list(self.active_configs.keys()):
                self.cleanup_config(test_id)
    
    def list_active_configs(self) -> List[str]:
        """List active test configuration IDs."""
        with self._lock:
            return list(self.active_configs.keys())
    
    def get_config_summary(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of test configuration."""
        config = self.get_config(test_id)
        if not config:
            return None
        
        return {
            'test_id': test_id,
            'test_type': config.test_type,
            'environment': config.environment,
            'created_at': config.created_at.isoformat(),
            'temp_dir': config.test_env.temp_dir,
            'sender_comp_id': config.fix_config.session.sender_comp_id,
            'target_comp_id': config.fix_config.session.target_comp_id,
            'mock_exchange_enabled': config.mock_config.enable_mock_exchange,
            'mock_market_data_enabled': config.mock_config.enable_mock_market_data
        }


# Global test configuration manager
_test_config_manager = None

def get_test_config_manager() -> TestConfigManager:
    """Get global test configuration manager."""
    global _test_config_manager
    if _test_config_manager is None:
        _test_config_manager = TestConfigManager()
    return _test_config_manager


# Convenience functions for test configuration
def create_test_config(test_type: str = "unit", **kwargs) -> TestConfig:
    """
    Create test configuration with automatic cleanup.
    
    Args:
        test_type: Type of test configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        TestConfig: Test configuration with context manager support
    """
    return TestConfigFactory.create_config(test_type, **kwargs)


def with_test_config(test_type: str = "unit", **kwargs):
    """
    Decorator for test methods that need test configuration.
    
    Args:
        test_type: Type of test configuration
        **kwargs: Additional configuration parameters
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **test_kwargs):
            with create_test_config(test_type, **kwargs) as config:
                # Add config as first argument
                return func(config, *args, **test_kwargs)
        
        return wrapper
    return decorator


# Test configuration templates for common scenarios
UNIT_TEST_TEMPLATE = {
    'test_type': 'unit',
    'mock_config.simulate_network_latency': False,
    'mock_config.simulate_errors': False,
    'session_settings.heartbeat_interval': 5,
    'session_settings.auto_login': True
}

INTEGRATION_TEST_TEMPLATE = {
    'test_type': 'integration',
    'mock_config.simulate_network_latency': True,
    'mock_config.network_latency_ms': 20,
    'mock_config.simulate_errors': True,
    'session_settings.strict_validation': True
}

PERFORMANCE_TEST_TEMPLATE = {
    'test_type': 'performance',
    'mock_config.exchange_fill_delay_ms': 10,
    'fix_config.performance.low_latency_mode': True,
    'fix_config.performance.message_queue_size': 50000,
    'session_settings.strict_validation': False
}

# Export commonly used templates
TEST_TEMPLATES = {
    'unit': UNIT_TEST_TEMPLATE,
    'integration': INTEGRATION_TEST_TEMPLATE,
    'performance': PERFORMANCE_TEST_TEMPLATE
}