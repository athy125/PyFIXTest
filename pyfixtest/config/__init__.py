"""
FIX Trading System Configuration Management

This module provides comprehensive configuration management for the FIX trading system,
including environment-specific configurations, security settings, performance tuning,
and deployment configurations.

Configuration Components:
    - FIX protocol settings and session parameters
    - Network and connection configurations
    - Security and authentication settings
    - Performance and optimization parameters
    - Logging and monitoring configurations
    - Environment-specific overrides
    - Deployment and clustering settings
"""

__version__ = "1.0.0"
__author__ = "FIX Trading System Team"

# Core configuration classes
from .fix_config import (
    FIXConfig,
    SessionConfig,
    NetworkConfig,
    SecurityConfig,
    PerformanceConfig
)

# Environment-specific configurations
from .environments import (
    DevelopmentConfig,
    TestingConfig,
    StagingConfig,
    ProductionConfig,
    EnvironmentConfig,
    ConfigurationManager
)

# Specialized configurations
from .security_config import (
    SSLConfig,
    AuthenticationConfig,
    EncryptionConfig,
    CertificateConfig,
    SecurityPolicyConfig
)

from .performance_config import (
    LatencyConfig,
    ThroughputConfig,
    MemoryConfig,
    ConnectionPoolConfig,
    OptimizationConfig
)

from .logging_config import (
    LoggingConfig,
    FileLogConfig,
    DatabaseLogConfig,
    RemoteLogConfig,
    AuditConfig
)

from .monitoring_config import (
    MetricsConfig,
    AlertConfig,
    HealthCheckConfig,
    DashboardConfig,
    NotificationConfig
)

# Deployment configurations
from .deployment_config import (
    ClusterConfig,
    LoadBalancerConfig,
    FailoverConfig,
    ScalingConfig,
    MaintenanceConfig
)

# Configuration validators and utilities
from .config_validators import (
    ConfigValidator,
    EnvironmentValidator,
    SecurityValidator,
    PerformanceValidator,
    ValidationResult
)

from .config_utils import (
    ConfigLoader,
    ConfigMerger,
    ConfigExporter,
    ConfigTemplateGenerator,
    ConfigMigrator
)

# Export all public classes and functions
__all__ = [
    # Core configuration
    'FIXConfig',
    'SessionConfig',
    'NetworkConfig',
    'SecurityConfig',
    'PerformanceConfig',
    
    # Environment configurations
    'DevelopmentConfig',
    'TestingConfig',
    'StagingConfig',
    'ProductionConfig',
    'EnvironmentConfig',
    'ConfigurationManager',
    
    # Specialized configurations
    'SSLConfig',
    'AuthenticationConfig',
    'EncryptionConfig',
    'CertificateConfig',
    'SecurityPolicyConfig',
    'LatencyConfig',
    'ThroughputConfig',
    'MemoryConfig',
    'ConnectionPoolConfig',
    'OptimizationConfig',
    'LoggingConfig',
    'FileLogConfig',
    'DatabaseLogConfig',
    'RemoteLogConfig',
    'AuditConfig',
    'MetricsConfig',
    'AlertConfig',
    'HealthCheckConfig',
    'DashboardConfig',
    'NotificationConfig',
    
    # Deployment configurations
    'ClusterConfig',
    'LoadBalancerConfig',
    'FailoverConfig',
    'ScalingConfig',
    'MaintenanceConfig',
    
    # Validators and utilities
    'ConfigValidator',
    'EnvironmentValidator',
    'SecurityValidator',
    'PerformanceValidator',
    'ValidationResult',
    'ConfigLoader',
    'ConfigMerger',
    'ConfigExporter',
    'ConfigTemplateGenerator',
    'ConfigMigrator',
    
    # Helper functions
    'load_config',
    'validate_config',
    'merge_configs',
    'export_config',
    'create_config_template',
    'get_environment_config',
    'set_environment',
    'get_current_environment',
    'apply_environment_overrides',
]

# Configuration file formats supported
SUPPORTED_FORMATS = ['ini', 'yaml', 'json', 'xml', 'properties', 'toml']

# Default configuration directories
DEFAULT_CONFIG_PATHS = [
    './config',
    './configs',
    '/etc/fix-trading-system',
    '~/.fix-trading-system',
    './examples/config'
]

# Environment variable prefixes
ENV_PREFIX = 'FIX_'
ENV_CONFIG_PREFIX = 'FIX_CONFIG_'

# Default configurations by environment
DEFAULT_ENVIRONMENTS = {
    'development': {
        'debug': True,
        'log_level': 'DEBUG',
        'heartbeat_interval': 30,
        'connection_timeout': 10,
        'ssl_enabled': False,
        'performance_monitoring': True,
        'file_logging': True,
        'console_logging': True,
        'metrics_enabled': True,
        'hot_reload': True
    },
    'testing': {
        'debug': True,
        'log_level': 'INFO',
        'heartbeat_interval': 10,
        'connection_timeout': 5,
        'ssl_enabled': True,
        'performance_monitoring': True,
        'file_logging': True,
        'console_logging': False,
        'metrics_enabled': True,
        'mock_external_services': True
    },
    'staging': {
        'debug': False,
        'log_level': 'INFO',
        'heartbeat_interval': 30,
        'connection_timeout': 15,
        'ssl_enabled': True,
        'performance_monitoring': True,
        'file_logging': True,
        'console_logging': False,
        'metrics_enabled': True,
        'database_logging': True
    },
    'production': {
        'debug': False,
        'log_level': 'WARNING',
        'heartbeat_interval': 30,
        'connection_timeout': 30,
        'ssl_enabled': True,
        'performance_monitoring': True,
        'file_logging': True,
        'console_logging': False,
        'metrics_enabled': True,
        'database_logging': True,
        'audit_logging': True,
        'high_availability': True,
        'clustering_enabled': True
    }
}

# Configuration schema version
CONFIG_SCHEMA_VERSION = "1.0"

# Global configuration manager instance
_config_manager = None

def get_config_manager() -> 'ConfigurationManager':
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

def load_config(
    config_path: str = None,
    environment: str = None,
    format: str = 'auto'
) -> 'FIXConfig':
    """
    Load configuration from file or environment.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name (development, testing, staging, production)
        format: Configuration file format
        
    Returns:
        FIXConfig: Loaded configuration object
    """
    manager = get_config_manager()
    return manager.load_config(config_path, environment, format)

def validate_config(config: 'FIXConfig', environment: str = None) -> 'ValidationResult':
    """
    Validate configuration for completeness and correctness.
    
    Args:
        config: Configuration to validate
        environment: Target environment for validation
        
    Returns:
        ValidationResult: Validation results
    """
    validator = ConfigValidator()
    return validator.validate(config, environment)

def merge_configs(*configs: 'FIXConfig') -> 'FIXConfig':
    """
    Merge multiple configurations with priority order.
    
    Args:
        *configs: Configuration objects to merge
        
    Returns:
        FIXConfig: Merged configuration
    """
    merger = ConfigMerger()
    return merger.merge(*configs)

def export_config(
    config: 'FIXConfig',
    output_path: str,
    format: str = 'yaml',
    include_sensitive: bool = False
) -> bool:
    """
    Export configuration to file.
    
    Args:
        config: Configuration to export
        output_path: Output file path
        format: Export format
        include_sensitive: Whether to include sensitive data
        
    Returns:
        bool: True if successful
    """
    exporter = ConfigExporter()
    return exporter.export(config, output_path, format, include_sensitive)

def create_config_template(
    environment: str = 'development',
    output_path: str = None,
    format: str = 'yaml'
) -> str:
    """
    Create configuration template for specific environment.
    
    Args:
        environment: Target environment
        output_path: Optional output file path
        format: Template format
        
    Returns:
        str: Template content or file path
    """
    generator = ConfigTemplateGenerator()
    return generator.create_template(environment, output_path, format)

def get_environment_config(environment: str) -> dict:
    """
    Get default configuration for specific environment.
    
    Args:
        environment: Environment name
        
    Returns:
        dict: Environment configuration
    """
    return DEFAULT_ENVIRONMENTS.get(environment, {}).copy()

def set_environment(environment: str):
    """
    Set current environment globally.
    
    Args:
        environment: Environment to set
    """
    manager = get_config_manager()
    manager.set_environment(environment)

def get_current_environment() -> str:
    """
    Get current environment setting.
    
    Returns:
        str: Current environment name
    """
    manager = get_config_manager()
    return manager.get_current_environment()

def apply_environment_overrides(config: 'FIXConfig', environment: str = None) -> 'FIXConfig':
    """
    Apply environment-specific overrides to configuration.
    
    Args:
        config: Base configuration
        environment: Environment for overrides
        
    Returns:
        FIXConfig: Configuration with overrides applied
    """
    if environment is None:
        environment = get_current_environment()
    
    env_config = get_environment_config(environment)
    manager = get_config_manager()
    return manager.apply_overrides(config, env_config)

def discover_config_files(search_paths: list = None) -> list:
    """
    Discover configuration files in standard locations.
    
    Args:
        search_paths: Optional custom search paths
        
    Returns:
        list: Found configuration files
    """
    import os
    import glob
    
    if search_paths is None:
        search_paths = DEFAULT_CONFIG_PATHS
    
    config_files = []
    for path in search_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            for format in SUPPORTED_FORMATS:
                pattern = os.path.join(expanded_path, f"*.{format}")
                config_files.extend(glob.glob(pattern))
    
    return config_files

def load_environment_from_file(env_file: str = '.env') -> dict:
    """
    Load environment variables from file.
    
    Args:
        env_file: Environment file path
        
    Returns:
        dict: Environment variables
    """
    import os
    
    env_vars = {}
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    env_vars[key.strip()] = value.strip()
    
    return env_vars

def get_config_from_env() -> dict:
    """
    Extract configuration from environment variables.
    
    Returns:
        dict: Configuration from environment
    """
    import os
    
    config = {}
    for key, value in os.environ.items():
        if key.startswith(ENV_CONFIG_PREFIX):
            config_key = key[len(ENV_CONFIG_PREFIX):].lower()
            config[config_key] = value
    
    return config

def setup_config_monitoring(config: 'FIXConfig', callback=None):
    """
    Setup configuration file monitoring for hot reload.
    
    Args:
        config: Configuration to monitor
        callback: Callback function for changes
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class ConfigChangeHandler(FileSystemEventHandler):
            def __init__(self, config, callback):
                self.config = config
                self.callback = callback
            
            def on_modified(self, event):
                if not event.is_directory and self.callback:
                    self.callback(event.src_path)
        
        # Setup file monitoring
        observer = Observer()
        handler = ConfigChangeHandler(config, callback)
        
        # Monitor config directories
        for path in DEFAULT_CONFIG_PATHS:
            if os.path.exists(path):
                observer.schedule(handler, path, recursive=False)
        
        observer.start()
        return observer
        
    except ImportError:
        # Watchdog not available, skip monitoring
        return None

# Configuration backup and restore
def backup_config(config: 'FIXConfig', backup_path: str = None) -> str:
    """
    Create backup of current configuration.
    
    Args:
        config: Configuration to backup
        backup_path: Optional backup location
        
    Returns:
        str: Backup file path
    """
    import os
    import datetime
    
    if backup_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"./config_backup_{timestamp}.yaml"
    
    success = export_config(config, backup_path, 'yaml', include_sensitive=True)
    return backup_path if success else None

def restore_config(backup_path: str) -> 'FIXConfig':
    """
    Restore configuration from backup.
    
    Args:
        backup_path: Path to backup file
        
    Returns:
        FIXConfig: Restored configuration
    """
    return load_config(backup_path)

# Configuration encryption/decryption
def encrypt_sensitive_config(config: 'FIXConfig', key: str = None) -> 'FIXConfig':
    """
    Encrypt sensitive configuration values.
    
    Args:
        config: Configuration to encrypt
        key: Encryption key
        
    Returns:
        FIXConfig: Configuration with encrypted sensitive values
    """
    from .security_config import ConfigEncryption
    
    encryption = ConfigEncryption(key)
    return encryption.encrypt_config(config)

def decrypt_sensitive_config(config: 'FIXConfig', key: str = None) -> 'FIXConfig':
    """
    Decrypt sensitive configuration values.
    
    Args:
        config: Configuration to decrypt
        key: Decryption key
        
    Returns:
        FIXConfig: Configuration with decrypted values
    """
    from .security_config import ConfigEncryption
    
    encryption = ConfigEncryption(key)
    return encryption.decrypt_config(config)

# Configuration versioning
def get_config_version(config: 'FIXConfig') -> str:
    """
    Get configuration schema version.
    
    Args:
        config: Configuration object
        
    Returns:
        str: Schema version
    """
    return getattr(config, 'schema_version', CONFIG_SCHEMA_VERSION)

def migrate_config(config: 'FIXConfig', target_version: str = None) -> 'FIXConfig':
    """
    Migrate configuration to newer schema version.
    
    Args:
        config: Configuration to migrate
        target_version: Target schema version
        
    Returns:
        FIXConfig: Migrated configuration
    """
    migrator = ConfigMigrator()
    return migrator.migrate(config, target_version or CONFIG_SCHEMA_VERSION)

# Configuration testing utilities
def test_config_connectivity(config: 'FIXConfig') -> dict:
    """
    Test connectivity with given configuration.
    
    Args:
        config: Configuration to test
        
    Returns:
        dict: Test results
    """
    results = {
        'network_connectivity': False,
        'ssl_validation': False,
        'authentication': False,
        'session_creation': False,
        'errors': []
    }
    
    try:
        # Test network connectivity
        import socket
        host = config.get_host()
        port = config.get_port()
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            results['network_connectivity'] = True
        else:
            results['errors'].append(f"Cannot connect to {host}:{port}")
    
    except Exception as e:
        results['errors'].append(f"Network test error: {e}")
    
    return results

def generate_config_report(config: 'FIXConfig') -> dict:
    """
    Generate comprehensive configuration report.
    
    Args:
        config: Configuration to analyze
        
    Returns:
        dict: Configuration report
    """
    report = {
        'schema_version': get_config_version(config),
        'environment': getattr(config, 'environment', 'unknown'),
        'validation_result': validate_config(config),
        'security_settings': {
            'ssl_enabled': getattr(config, 'ssl_enabled', False),
            'authentication_enabled': getattr(config, 'authentication_enabled', False),
            'encryption_enabled': getattr(config, 'encryption_enabled', False)
        },
        'performance_settings': {
            'heartbeat_interval': getattr(config, 'heartbeat_interval', 30),
            'connection_timeout': getattr(config, 'connection_timeout', 30),
            'message_queue_size': getattr(config, 'message_queue_size', 1000)
        },
        'deployment_settings': {
            'clustering_enabled': getattr(config, 'clustering_enabled', False),
            'high_availability': getattr(config, 'high_availability', False),
            'auto_scaling': getattr(config, 'auto_scaling', False)
        }
    }
    
    return report

# Error handling for configuration
class ConfigurationError(Exception):
    """Base exception for configuration errors."""
    pass

class ConfigValidationError(ConfigurationError):
    """Exception raised for configuration validation errors."""
    pass

class ConfigLoadError(ConfigurationError):
    """Exception raised when configuration loading fails."""
    pass

class ConfigSaveError(ConfigurationError):
    """Exception raised when configuration saving fails."""
    pass

class EnvironmentError(ConfigurationError):
    """Exception raised for environment-related errors."""
    pass

# Configuration change tracking
class ConfigChangeTracker:
    """Track configuration changes for audit purposes."""
    
    def __init__(self):
        self.changes = []
        self.enabled = False
    
    def enable_tracking(self):
        """Enable change tracking."""
        self.enabled = True
    
    def disable_tracking(self):
        """Disable change tracking."""
        self.enabled = False
    
    def record_change(self, field: str, old_value, new_value, timestamp=None):
        """Record a configuration change."""
        if not self.enabled:
            return
        
        import datetime
        
        change = {
            'field': field,
            'old_value': old_value,
            'new_value': new_value,
            'timestamp': timestamp or datetime.datetime.now(),
            'user': getattr(self, 'current_user', 'system')
        }
        
        self.changes.append(change)
    
    def get_changes(self, since=None) -> list:
        """Get recorded changes."""
        if since is None:
            return self.changes.copy()
        
        return [c for c in self.changes if c['timestamp'] >= since]

# Global change tracker
config_change_tracker = ConfigChangeTracker()

# Module initialization
def _initialize_config_module():
    """Initialize configuration module."""
    import os
    
    # Set default environment from environment variable
    default_env = os.environ.get('FIX_ENVIRONMENT', 'development')
    set_environment(default_env)
    
    # Load environment variables from .env file if present
    env_vars = load_environment_from_file()
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    
    # Enable change tracking in development
    if get_current_environment() == 'development':
        config_change_tracker.enable_tracking()

# Initialize module
_initialize_config_module()

# Version information
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'schema_version': CONFIG_SCHEMA_VERSION,
    'supported_formats': SUPPORTED_FORMATS
}

def get_version():
    """Get version string."""
    return __version__

def get_version_info():
    """Get detailed version information."""
    return VERSION_INFO.copy()

# Usage documentation
USAGE_EXAMPLES = """
Configuration Usage Examples:

1. Load configuration from file:
   >>> from fix_trading_system.config import load_config
   >>> config = load_config('config/production.yaml')

2. Load environment-specific configuration:
   >>> config = load_config(environment='production')

3. Validate configuration:
   >>> from fix_trading_system.config import validate_config
   >>> result = validate_config(config, 'production')

4. Merge configurations:
   >>> from fix_trading_system.config import merge_configs
   >>> merged = merge_configs(base_config, override_config)

5. Create configuration template:
   >>> from fix_trading_system.config import create_config_template
   >>> template = create_config_template('production', 'config.yaml')

6. Set environment:
   >>> from fix_trading_system.config import set_environment
   >>> set_environment('staging')

7. Export configuration:
   >>> from fix_trading_system.config import export_config
   >>> export_config(config, 'backup.yaml', include_sensitive=False)

Environment Variables:
- FIX_ENVIRONMENT: Set default environment
- FIX_CONFIG_*: Configuration overrides
- FIX_LOG_LEVEL: Set logging level
- FIX_DEBUG: Enable debug mode

Supported Formats:
- YAML (.yaml, .yml)
- JSON (.json)
- INI (.ini, .cfg)
- XML (.xml)
- Properties (.properties)
- TOML (.toml)
"""

# Add usage examples to module documentation
__doc__ += USAGE_EXAMPLES