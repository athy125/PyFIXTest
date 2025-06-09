"""
Core FIX configuration classes for the trading system.

This module provides the main configuration classes for FIX protocol settings,
session management, network configuration, security settings, and performance tuning.
"""

import os
import yaml
import json
import configparser
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..utils.logging_config import get_logger


@dataclass
class SessionConfig:
    """Configuration for FIX session parameters."""
    
    # Session identification
    sender_comp_id: str = ""
    target_comp_id: str = ""
    session_qualifier: str = ""
    begin_string: str = "FIX.4.4"
    
    # Connection settings
    connection_type: str = "initiator"  # initiator or acceptor
    socket_connect_host: str = "localhost"
    socket_connect_port: int = 9876
    socket_accept_port: int = 9876
    
    # Timing settings
    heartbeat_interval: int = 30
    logon_timeout: int = 10
    logout_timeout: int = 2
    reconnect_interval: int = 30
    max_reconnect_attempts: int = 5
    
    # Message handling
    reset_on_logon: bool = True
    reset_on_logout: bool = False
    reset_on_disconnect: bool = False
    refresh_on_logon: bool = False
    
    # Sequence number management
    persist_messages: bool = True
    validate_length_and_checksum: bool = True
    validate_fields_out_of_order: bool = True
    validate_fields_have_values: bool = True
    validate_user_defined_fields: bool = True
    
    # Session qualifiers
    use_data_dictionary: bool = True
    data_dictionary: str = ""
    transport_data_dictionary: str = ""
    app_data_dictionary: str = ""
    
    # Time zone and scheduling
    start_time: str = "00:00:00"
    end_time: str = "00:00:00"
    start_day: str = ""
    end_day: str = ""
    timezone: str = "UTC"
    
    # Session state
    check_latency: bool = True
    max_latency: int = 120
    test_request_delay_multiplier: float = 2.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'SenderCompID': self.sender_comp_id,
            'TargetCompID': self.target_comp_id,
            'SessionQualifier': self.session_qualifier,
            'BeginString': self.begin_string,
            'ConnectionType': self.connection_type,
            'SocketConnectHost': self.socket_connect_host,
            'SocketConnectPort': str(self.socket_connect_port),
            'SocketAcceptPort': str(self.socket_accept_port),
            'HeartBtInt': str(self.heartbeat_interval),
            'LogonTimeout': str(self.logon_timeout),
            'LogoutTimeout': str(self.logout_timeout),
            'ReconnectInterval': str(self.reconnect_interval),
            'ResetOnLogon': 'Y' if self.reset_on_logon else 'N',
            'ResetOnLogout': 'Y' if self.reset_on_logout else 'N',
            'ResetOnDisconnect': 'Y' if self.reset_on_disconnect else 'N',
            'RefreshOnLogon': 'Y' if self.refresh_on_logon else 'N',
            'PersistMessages': 'Y' if self.persist_messages else 'N',
            'ValidateLengthAndChecksum': 'Y' if self.validate_length_and_checksum else 'N',
            'ValidateFieldsOutOfOrder': 'Y' if self.validate_fields_out_of_order else 'N',
            'ValidateFieldsHaveValues': 'Y' if self.validate_fields_have_values else 'N',
            'ValidateUserDefinedFields': 'Y' if self.validate_user_defined_fields else 'N',
            'UseDataDictionary': 'Y' if self.use_data_dictionary else 'N',
            'DataDictionary': self.data_dictionary,
            'TransportDataDictionary': self.transport_data_dictionary,
            'AppDataDictionary': self.app_data_dictionary,
            'StartTime': self.start_time,
            'EndTime': self.end_time,
            'StartDay': self.start_day,
            'EndDay': self.end_day,
            'TimeZone': self.timezone,
            'CheckLatency': 'Y' if self.check_latency else 'N',
            'MaxLatency': str(self.max_latency),
            'TestRequestDelayMultiplier': str(self.test_request_delay_multiplier)
        }


@dataclass
class NetworkConfig:
    """Network and connection configuration."""
    
    # Basic connection settings
    host: str = "localhost"
    port: int = 9876
    bind_address: str = ""
    
    # Connection behavior
    socket_reuse_address: bool = True
    socket_no_delay: bool = True
    socket_send_buffer_size: int = 0
    socket_receive_buffer_size: int = 0
    
    # Timeout settings
    connect_timeout: int = 30
    send_timeout: int = 30
    receive_timeout: int = 30
    keep_alive_timeout: int = 300
    
    # Connection pooling
    max_connections: int = 100
    min_connections: int = 1
    connection_pool_size: int = 10
    connection_idle_timeout: int = 600
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    max_retry_delay: float = 60.0
    
    # Network optimization
    tcp_no_delay: bool = True
    tcp_keep_alive: bool = True
    tcp_keep_alive_idle: int = 300
    tcp_keep_alive_interval: int = 60
    tcp_keep_alive_count: int = 3
    
    # Bandwidth management
    send_rate_limit: int = 0  # bytes per second, 0 = unlimited
    receive_rate_limit: int = 0  # bytes per second, 0 = unlimited
    message_rate_limit: int = 0  # messages per second, 0 = unlimited
    
    def validate(self) -> List[str]:
        """Validate network configuration."""
        errors = []
        
        if not 1 <= self.port <= 65535:
            errors.append("Port must be between 1 and 65535")
        
        if self.connect_timeout <= 0:
            errors.append("Connect timeout must be positive")
        
        if self.max_connections <= 0:
            errors.append("Max connections must be positive")
        
        if self.min_connections < 0:
            errors.append("Min connections cannot be negative")
        
        if self.min_connections > self.max_connections:
            errors.append("Min connections cannot exceed max connections")
        
        return errors


@dataclass
class SecurityConfig:
    """Security and authentication configuration."""
    
    # SSL/TLS settings
    ssl_enabled: bool = False
    ssl_protocol: str = "TLSv1.2"
    ssl_cert_file: str = ""
    ssl_key_file: str = ""
    ssl_ca_file: str = ""
    ssl_verify_mode: str = "CERT_REQUIRED"
    ssl_ciphers: str = ""
    ssl_check_hostname: bool = True
    
    # Authentication
    authentication_enabled: bool = False
    username: str = ""
    password: str = ""
    password_encrypted: bool = False
    encryption_key: str = ""
    
    # Authorization
    authorization_enabled: bool = False
    user_roles: List[str] = field(default_factory=list)
    permissions: Dict[str, List[str]] = field(default_factory=dict)
    
    # Security policies
    enforce_encryption: bool = False
    require_strong_passwords: bool = True
    password_expiry_days: int = 90
    max_login_attempts: int = 3
    lockout_duration_minutes: int = 15
    
    # Audit and compliance
    audit_enabled: bool = False
    audit_log_file: str = ""
    compliance_mode: str = ""  # MIFID_II, SOX, etc.
    
    # IP filtering
    allowed_ips: List[str] = field(default_factory=list)
    blocked_ips: List[str] = field(default_factory=list)
    ip_whitelist_enabled: bool = False
    
    # Message encryption
    message_encryption_enabled: bool = False
    encryption_algorithm: str = "AES256"
    key_rotation_interval_hours: int = 24
    
    def validate(self) -> List[str]:
        """Validate security configuration."""
        errors = []
        
        if self.ssl_enabled:
            if not self.ssl_cert_file:
                errors.append("SSL certificate file required when SSL is enabled")
            if not self.ssl_key_file:
                errors.append("SSL key file required when SSL is enabled")
        
        if self.authentication_enabled:
            if not self.username:
                errors.append("Username required when authentication is enabled")
            if not self.password and not self.password_encrypted:
                errors.append("Password required when authentication is enabled")
        
        if self.password_expiry_days <= 0:
            errors.append("Password expiry days must be positive")
        
        if self.max_login_attempts <= 0:
            errors.append("Max login attempts must be positive")
        
        return errors


@dataclass
class PerformanceConfig:
    """Performance and optimization configuration."""
    
    # Message processing
    message_queue_size: int = 10000
    worker_threads: int = 4
    async_processing: bool = True
    batch_processing: bool = False
    batch_size: int = 100
    batch_timeout_ms: int = 10
    
    # Memory management
    max_memory_usage_mb: int = 1024
    gc_enabled: bool = True
    gc_interval_seconds: int = 60
    object_pool_enabled: bool = True
    
    # Latency optimization
    low_latency_mode: bool = False
    cpu_affinity: List[int] = field(default_factory=list)
    priority_boost: bool = False
    real_time_scheduling: bool = False
    
    # Throughput optimization
    high_throughput_mode: bool = False
    compression_enabled: bool = False
    compression_algorithm: str = "gzip"
    compression_level: int = 6
    
    # Caching
    cache_enabled: bool = True
    cache_size_mb: int = 128
    cache_ttl_seconds: int = 300
    cache_strategy: str = "LRU"
    
    # I/O optimization
    io_buffer_size: int = 8192
    io_async_enabled: bool = True
    io_direct_enabled: bool = False
    io_sync_frequency: int = 100
    
    # Resource limits
    max_open_files: int = 1024
    max_file_size_mb: int = 1024
    max_log_files: int = 10
    max_session_count: int = 100
    
    # Monitoring thresholds
    cpu_threshold_percent: float = 80.0
    memory_threshold_percent: float = 85.0
    disk_threshold_percent: float = 90.0
    network_threshold_mbps: float = 100.0
    
    def validate(self) -> List[str]:
        """Validate performance configuration."""
        errors = []
        
        if self.message_queue_size <= 0:
            errors.append("Message queue size must be positive")
        
        if self.worker_threads <= 0:
            errors.append("Worker threads must be positive")
        
        if self.max_memory_usage_mb <= 0:
            errors.append("Max memory usage must be positive")
        
        if not 0 <= self.cpu_threshold_percent <= 100:
            errors.append("CPU threshold must be between 0 and 100")
        
        if not 0 <= self.memory_threshold_percent <= 100:
            errors.append("Memory threshold must be between 0 and 100")
        
        return errors


class FIXConfig:
    """
    Main FIX configuration class that orchestrates all configuration components.
    
    This class provides a comprehensive configuration management system for FIX
    trading applications, supporting multiple environments, validation, and
    various configuration sources.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize FIX configuration.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.logger = get_logger(__name__)
        
        # Configuration components
        self.session = SessionConfig()
        self.network = NetworkConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        
        # Metadata
        self.environment = "development"
        self.config_version = "1.0"
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        
        # File paths
        self.store_path = "./store"
        self.log_path = "./logs"
        self.data_dictionary_path = "./spec"
        
        # Application settings
        self.app_name = "FIX Trading System"
        self.app_version = "1.0.0"
        self.debug_mode = False
        
        # Load configuration if file provided
        if config_file:
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str, format: str = None) -> bool:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
            format: Configuration format (auto-detected if None)
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                self.logger.error(f"Configuration file not found: {config_file}")
                return False
            
            # Auto-detect format if not specified
            if format is None:
                format = config_path.suffix.lower().lstrip('.')
            
            # Load based on format
            if format in ['yaml', 'yml']:
                return self._load_yaml(config_file)
            elif format == 'json':
                return self._load_json(config_file)
            elif format in ['ini', 'cfg']:
                return self._load_ini(config_file)
            else:
                self.logger.error(f"Unsupported configuration format: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> bool:
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            # Load session configuration
            if 'session' in config_dict:
                self._load_session_config(config_dict['session'])
            
            # Load network configuration
            if 'network' in config_dict:
                self._load_network_config(config_dict['network'])
            
            # Load security configuration
            if 'security' in config_dict:
                self._load_security_config(config_dict['security'])
            
            # Load performance configuration
            if 'performance' in config_dict:
                self._load_performance_config(config_dict['performance'])
            
            # Load general settings
            self._load_general_config(config_dict)
            
            self.updated_at = datetime.now(timezone.utc)
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading configuration from dict: {e}")
            return False
    
    def save_to_file(self, config_file: str, format: str = None, include_sensitive: bool = False) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_file: Path to configuration file
            format: Configuration format (auto-detected if None)
            include_sensitive: Whether to include sensitive data
            
        Returns:
            bool: True if saved successfully
        """
        try:
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Auto-detect format if not specified
            if format is None:
                format = config_path.suffix.lower().lstrip('.')
            
            # Convert to dictionary
            config_dict = self.to_dict(include_sensitive)
            
            # Save based on format
            if format in ['yaml', 'yml']:
                return self._save_yaml(config_file, config_dict)
            elif format == 'json':
                return self._save_json(config_file, config_dict)
            elif format in ['ini', 'cfg']:
                return self._save_ini(config_file, config_dict)
            else:
                self.logger.error(f"Unsupported configuration format: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive data
            
        Returns:
            Dict: Configuration dictionary
        """
        config_dict = {
            'metadata': {
                'environment': self.environment,
                'config_version': self.config_version,
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat(),
                'app_name': self.app_name,
                'app_version': self.app_version,
                'debug_mode': self.debug_mode
            },
            'paths': {
                'store_path': self.store_path,
                'log_path': self.log_path,
                'data_dictionary_path': self.data_dictionary_path
            },
            'session': {
                'sender_comp_id': self.session.sender_comp_id,
                'target_comp_id': self.session.target_comp_id,
                'session_qualifier': self.session.session_qualifier,
                'begin_string': self.session.begin_string,
                'connection_type': self.session.connection_type,
                'socket_connect_host': self.session.socket_connect_host,
                'socket_connect_port': self.session.socket_connect_port,
                'socket_accept_port': self.session.socket_accept_port,
                'heartbeat_interval': self.session.heartbeat_interval,
                'logon_timeout': self.session.logon_timeout,
                'logout_timeout': self.session.logout_timeout,
                'reconnect_interval': self.session.reconnect_interval,
                'max_reconnect_attempts': self.session.max_reconnect_attempts,
                'reset_on_logon': self.session.reset_on_logon,
                'reset_on_logout': self.session.reset_on_logout,
                'reset_on_disconnect': self.session.reset_on_disconnect,
                'refresh_on_logon': self.session.refresh_on_logon,
                'persist_messages': self.session.persist_messages,
                'validate_length_and_checksum': self.session.validate_length_and_checksum,
                'validate_fields_out_of_order': self.session.validate_fields_out_of_order,
                'validate_fields_have_values': self.session.validate_fields_have_values,
                'validate_user_defined_fields': self.session.validate_user_defined_fields,
                'use_data_dictionary': self.session.use_data_dictionary,
                'data_dictionary': self.session.data_dictionary,
                'transport_data_dictionary': self.session.transport_data_dictionary,
                'app_data_dictionary': self.session.app_data_dictionary,
                'start_time': self.session.start_time,
                'end_time': self.session.end_time,
                'start_day': self.session.start_day,
                'end_day': self.session.end_day,
                'timezone': self.session.timezone,
                'check_latency': self.session.check_latency,
                'max_latency': self.session.max_latency,
                'test_request_delay_multiplier': self.session.test_request_delay_multiplier
            },
            'network': {
                'host': self.network.host,
                'port': self.network.port,
                'bind_address': self.network.bind_address,
                'socket_reuse_address': self.network.socket_reuse_address,
                'socket_no_delay': self.network.socket_no_delay,
                'socket_send_buffer_size': self.network.socket_send_buffer_size,
                'socket_receive_buffer_size': self.network.socket_receive_buffer_size,
                'connect_timeout': self.network.connect_timeout,
                'send_timeout': self.network.send_timeout,
                'receive_timeout': self.network.receive_timeout,
                'keep_alive_timeout': self.network.keep_alive_timeout,
                'max_connections': self.network.max_connections,
                'min_connections': self.network.min_connections,
                'connection_pool_size': self.network.connection_pool_size,
                'connection_idle_timeout': self.network.connection_idle_timeout,
                'max_retries': self.network.max_retries,
                'retry_delay': self.network.retry_delay,
                'exponential_backoff': self.network.exponential_backoff,
                'backoff_multiplier': self.network.backoff_multiplier,
                'max_retry_delay': self.network.max_retry_delay,
                'tcp_no_delay': self.network.tcp_no_delay,
                'tcp_keep_alive': self.network.tcp_keep_alive,
                'tcp_keep_alive_idle': self.network.tcp_keep_alive_idle,
                'tcp_keep_alive_interval': self.network.tcp_keep_alive_interval,
                'tcp_keep_alive_count': self.network.tcp_keep_alive_count,
                'send_rate_limit': self.network.send_rate_limit,
                'receive_rate_limit': self.network.receive_rate_limit,
                'message_rate_limit': self.network.message_rate_limit
            },
            'performance': {
                'message_queue_size': self.performance.message_queue_size,
                'worker_threads': self.performance.worker_threads,
                'async_processing': self.performance.async_processing,
                'batch_processing': self.performance.batch_processing,
                'batch_size': self.performance.batch_size,
                'batch_timeout_ms': self.performance.batch_timeout_ms,
                'max_memory_usage_mb': self.performance.max_memory_usage_mb,
                'gc_enabled': self.performance.gc_enabled,
                'gc_interval_seconds': self.performance.gc_interval_seconds,
                'object_pool_enabled': self.performance.object_pool_enabled,
                'low_latency_mode': self.performance.low_latency_mode,
                'cpu_affinity': self.performance.cpu_affinity,
                'priority_boost': self.performance.priority_boost,
                'real_time_scheduling': self.performance.real_time_scheduling,
                'high_throughput_mode': self.performance.high_throughput_mode,
                'compression_enabled': self.performance.compression_enabled,
                'compression_algorithm': self.performance.compression_algorithm,
                'compression_level': self.performance.compression_level,
                'cache_enabled': self.performance.cache_enabled,
                'cache_size_mb': self.performance.cache_size_mb,
                'cache_ttl_seconds': self.performance.cache_ttl_seconds,
                'cache_strategy': self.performance.cache_strategy,
                'io_buffer_size': self.performance.io_buffer_size,
                'io_async_enabled': self.performance.io_async_enabled,
                'io_direct_enabled': self.performance.io_direct_enabled,
                'io_sync_frequency': self.performance.io_sync_frequency,
                'max_open_files': self.performance.max_open_files,
                'max_file_size_mb': self.performance.max_file_size_mb,
                'max_log_files': self.performance.max_log_files,
                'max_session_count': self.performance.max_session_count,
                'cpu_threshold_percent': self.performance.cpu_threshold_percent,
                'memory_threshold_percent': self.performance.memory_threshold_percent,
                'disk_threshold_percent': self.performance.disk_threshold_percent,
                'network_threshold_mbps': self.performance.network_threshold_mbps
            }
        }
        
        # Add security configuration (conditionally include sensitive data)
        security_config = {
            'ssl_enabled': self.security.ssl_enabled,
            'ssl_protocol': self.security.ssl_protocol,
            'ssl_cert_file': self.security.ssl_cert_file,
            'ssl_key_file': self.security.ssl_key_file,
            'ssl_ca_file': self.security.ssl_ca_file,
            'ssl_verify_mode': self.security.ssl_verify_mode,
            'ssl_ciphers': self.security.ssl_ciphers,
            'ssl_check_hostname': self.security.ssl_check_hostname,
            'authentication_enabled': self.security.authentication_enabled,
            'username': self.security.username,
            'password_encrypted': self.security.password_encrypted,
            'authorization_enabled': self.security.authorization_enabled,
            'user_roles': self.security.user_roles,
            'permissions': self.security.permissions,
            'enforce_encryption': self.security.enforce_encryption,
            'require_strong_passwords': self.security.require_strong_passwords,
            'password_expiry_days': self.security.password_expiry_days,
            'max_login_attempts': self.security.max_login_attempts,
            'lockout_duration_minutes': self.security.lockout_duration_minutes,
            'audit_enabled': self.security.audit_enabled,
            'audit_log_file': self.security.audit_log_file,
            'compliance_mode': self.security.compliance_mode,
            'allowed_ips': self.security.allowed_ips,
            'blocked_ips': self.security.blocked_ips,
            'ip_whitelist_enabled': self.security.ip_whitelist_enabled,
            'message_encryption_enabled': self.security.message_encryption_enabled,
            'encryption_algorithm': self.security.encryption_algorithm,
            'key_rotation_interval_hours': self.security.key_rotation_interval_hours
        }
        
        # Include sensitive data if requested
        if include_sensitive:
            security_config.update({
                'password': self.security.password,
                'encryption_key': self.security.encryption_key
            })
        
        config_dict['security'] = security_config
        
        return config_dict
    
    def validate(self) -> List[str]:
        """
        Validate entire configuration.
        
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Validate components
        errors.extend(self.network.validate())
        errors.extend(self.security.validate())
        errors.extend(self.performance.validate())
        
        # Validate paths
        if not self.store_path:
            errors.append("Store path is required")
        
        if not self.log_path:
            errors.append("Log path is required")
        
        # Validate session configuration
        if not self.session.sender_comp_id:
            errors.append("SenderCompID is required")
        
        if not self.session.target_comp_id:
            errors.append("TargetCompID is required")
        
        return errors
    
    def create_directories(self) -> bool:
        """
        Create necessary directories for configuration.
        
        Returns:
            bool: True if successful
        """
        try:
            Path(self.store_path).mkdir(parents=True, exist_ok=True)
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
            Path(self.data_dictionary_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Error creating directories: {e}")
            return False
    
    # Convenience methods for common configuration access
    def get_host(self) -> str:
        """Get connection host."""
        return self.session.socket_connect_host or self.network.host
    
    def get_port(self) -> int:
        """Get connection port."""
        return self.session.socket_connect_port or self.network.port
    
    def get_heartbeat_interval(self) -> int:
        """Get heartbeat interval."""
        return self.session.heartbeat_interval
    
    def get_reconnect_interval(self) -> int:
        """Get reconnect interval."""
        return self.session.reconnect_interval
    
    def get_max_reconnect_attempts(self) -> int:
        """Get maximum reconnect attempts."""
        return self.session.max_reconnect_attempts
    
    def get_store_path(self) -> str:
        """Get message store path."""
        return self.store_path
    
    def get_log_path(self) -> str:
        """Get log file path."""
        return self.log_path
    
    def get_begin_string(self) -> str:
        """Get FIX version string."""
        return self.session.begin_string
    
    def is_ssl_enabled(self) -> bool:
        """Check if SSL is enabled."""
        return self.security.ssl_enabled
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug_mode
    
    # QuickFIX settings generation
    def get_initiator_settings(self) -> 'quickfix.SessionSettings':
        """
        Generate QuickFIX initiator settings.
        
        Returns:
            quickfix.SessionSettings: QuickFIX settings object
        """
        try:
            import quickfix as fix
            
            settings = fix.SessionSettings()
            
            # Default settings
            default_dict = fix.Dictionary()
            default_dict.setString("ConnectionType", "initiator")
            default_dict.setString("ReconnectInterval", str(self.session.reconnect_interval))
            default_dict.setString("FileStorePath", self.store_path)
            default_dict.setString("FileLogPath", self.log_path)
            default_dict.setString("LogonTimeout", str(self.session.logon_timeout))
            default_dict.setString("LogoutTimeout", str(self.session.logout_timeout))
            default_dict.setString("SocketConnectHost", self.get_host())
            default_dict.setString("SocketConnectPort", str(self.get_port()))
            default_dict.setString("StartTime", self.session.start_time)
            default_dict.setString("EndTime", self.session.end_time)
            default_dict.setString("UseDataDictionary", "Y" if self.session.use_data_dictionary else "N")
            
            # SSL settings
            if self.security.ssl_enabled:
                default_dict.setString("SocketUseSSL", "Y")
                if self.security.ssl_cert_file:
                    default_dict.setString("SocketCertificateFile", self.security.ssl_cert_file)
                if self.security.ssl_key_file:
                    default_dict.setString("SocketPrivateKeyFile", self.security.ssl_key_file)
                if self.security.ssl_ca_file:
                    default_dict.setString("SocketCAFile", self.security.ssl_ca_file)
            
            # Data dictionary settings
            if self.session.data_dictionary:
                default_dict.setString("DataDictionary", self.session.data_dictionary)
            if self.session.transport_data_dictionary:
                default_dict.setString("TransportDataDictionary", self.session.transport_data_dictionary)
            if self.session.app_data_dictionary:
                default_dict.setString("AppDataDictionary", self.session.app_data_dictionary)
            
            # Performance settings
            if self.network.socket_no_delay:
                default_dict.setString("SocketNoDelay", "Y")
            if self.network.socket_reuse_address:
                default_dict.setString("SocketReuseAddress", "Y")
            if self.network.socket_send_buffer_size > 0:
                default_dict.setString("SocketSendBufferSize", str(self.network.socket_send_buffer_size))
            if self.network.socket_receive_buffer_size > 0:
                default_dict.setString("SocketReceiveBufferSize", str(self.network.socket_receive_buffer_size))
            
            settings.set(default_dict)
            
            # Session-specific settings
            session_dict = fix.Dictionary()
            session_dict.setString("BeginString", self.session.begin_string)
            session_dict.setString("SenderCompID", self.session.sender_comp_id)
            session_dict.setString("TargetCompID", self.session.target_comp_id)
            session_dict.setString("HeartBtInt", str(self.session.heartbeat_interval))
            
            if self.session.session_qualifier:
                session_dict.setString("SessionQualifier", self.session.session_qualifier)
            
            # Reset settings
            session_dict.setString("ResetOnLogon", "Y" if self.session.reset_on_logon else "N")
            session_dict.setString("ResetOnLogout", "Y" if self.session.reset_on_logout else "N")
            session_dict.setString("ResetOnDisconnect", "Y" if self.session.reset_on_disconnect else "N")
            session_dict.setString("RefreshOnLogon", "Y" if self.session.refresh_on_logon else "N")
            
            # Validation settings
            session_dict.setString("ValidateLengthAndChecksum", "Y" if self.session.validate_length_and_checksum else "N")
            session_dict.setString("ValidateFieldsOutOfOrder", "Y" if self.session.validate_fields_out_of_order else "N")
            session_dict.setString("ValidateFieldsHaveValues", "Y" if self.session.validate_fields_have_values else "N")
            session_dict.setString("ValidateUserDefinedFields", "Y" if self.session.validate_user_defined_fields else "N")
            
            # Authentication
            if self.security.authentication_enabled and self.security.username:
                session_dict.setString("Username", self.security.username)
                if self.security.password and not self.security.password_encrypted:
                    session_dict.setString("Password", self.security.password)
            
            # Create session ID
            session_id = fix.SessionID(
                self.session.begin_string,
                self.session.sender_comp_id,
                self.session.target_comp_id,
                self.session.session_qualifier
            )
            
            settings.set(session_id, session_dict)
            
            return settings
            
        except Exception as e:
            self.logger.error(f"Error creating initiator settings: {e}")
            raise
    
    def get_acceptor_settings(self) -> 'quickfix.SessionSettings':
        """
        Generate QuickFIX acceptor settings.
        
        Returns:
            quickfix.SessionSettings: QuickFIX settings object
        """
        try:
            import quickfix as fix
            
            settings = fix.SessionSettings()
            
            # Default settings
            default_dict = fix.Dictionary()
            default_dict.setString("ConnectionType", "acceptor")
            default_dict.setString("SocketAcceptPort", str(self.session.socket_accept_port))
            default_dict.setString("FileStorePath", self.store_path)
            default_dict.setString("FileLogPath", self.log_path)
            default_dict.setString("StartTime", self.session.start_time)
            default_dict.setString("EndTime", self.session.end_time)
            default_dict.setString("UseDataDictionary", "Y" if self.session.use_data_dictionary else "N")
            
            # SSL settings
            if self.security.ssl_enabled:
                default_dict.setString("SocketUseSSL", "Y")
                if self.security.ssl_cert_file:
                    default_dict.setString("SocketCertificateFile", self.security.ssl_cert_file)
                if self.security.ssl_key_file:
                    default_dict.setString("SocketPrivateKeyFile", self.security.ssl_key_file)
                if self.security.ssl_ca_file:
                    default_dict.setString("SocketCAFile", self.security.ssl_ca_file)
            
            # Data dictionary settings
            if self.session.data_dictionary:
                default_dict.setString("DataDictionary", self.session.data_dictionary)
            if self.session.transport_data_dictionary:
                default_dict.setString("TransportDataDictionary", self.session.transport_data_dictionary)
            if self.session.app_data_dictionary:
                default_dict.setString("AppDataDictionary", self.session.app_data_dictionary)
            
            # Network settings
            if self.network.bind_address:
                default_dict.setString("SocketAcceptAddress", self.network.bind_address)
            if self.network.socket_no_delay:
                default_dict.setString("SocketNoDelay", "Y")
            if self.network.socket_reuse_address:
                default_dict.setString("SocketReuseAddress", "Y")
            
            settings.set(default_dict)
            
            # Session-specific settings
            session_dict = fix.Dictionary()
            session_dict.setString("BeginString", self.session.begin_string)
            session_dict.setString("SenderCompID", self.session.sender_comp_id)
            session_dict.setString("TargetCompID", self.session.target_comp_id)
            session_dict.setString("HeartBtInt", str(self.session.heartbeat_interval))
            
            if self.session.session_qualifier:
                session_dict.setString("SessionQualifier", self.session.session_qualifier)
            
            # Create session ID
            session_id = fix.SessionID(
                self.session.begin_string,
                self.session.sender_comp_id,
                self.session.target_comp_id,
                self.session.session_qualifier
            )
            
            settings.set(session_id, session_dict)
            
            return settings
            
        except Exception as e:
            self.logger.error(f"Error creating acceptor settings: {e}")
            raise
    
    def get_session_id(self) -> 'quickfix.SessionID':
        """
        Get QuickFIX session ID.
        
        Returns:
            quickfix.SessionID: Session ID object
        """
        try:
            import quickfix as fix
            
            return fix.SessionID(
                self.session.begin_string,
                self.session.sender_comp_id,
                self.session.target_comp_id,
                self.session.session_qualifier
            )
            
        except Exception as e:
            self.logger.error(f"Error creating session ID: {e}")
            raise
    
    # Private helper methods for loading different formats
    def _load_yaml(self, config_file: str) -> bool:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            return self.load_from_dict(config_dict)
        except Exception as e:
            self.logger.error(f"Error loading YAML configuration: {e}")
            return False
    
    def _load_json(self, config_file: str) -> bool:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            return self.load_from_dict(config_dict)
        except Exception as e:
            self.logger.error(f"Error loading JSON configuration: {e}")
            return False
    
    def _load_ini(self, config_file: str) -> bool:
        """Load configuration from INI file."""
        try:
            config = configparser.ConfigParser()
            config.read(config_file)
            
            # Convert ConfigParser to dict
            config_dict = {}
            for section in config.sections():
                config_dict[section] = dict(config[section])
            
            return self.load_from_dict(config_dict)
        except Exception as e:
            self.logger.error(f"Error loading INI configuration: {e}")
            return False
    
    def _save_yaml(self, config_file: str, config_dict: Dict[str, Any]) -> bool:
        """Save configuration to YAML file."""
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error saving YAML configuration: {e}")
            return False
    
    def _save_json(self, config_file: str, config_dict: Dict[str, Any]) -> bool:
        """Save configuration to JSON file."""
        try:
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error saving JSON configuration: {e}")
            return False
    
    def _save_ini(self, config_file: str, config_dict: Dict[str, Any]) -> bool:
        """Save configuration to INI file."""
        try:
            config = configparser.ConfigParser()
            
            # Convert dict to ConfigParser format
            for section_name, section_data in config_dict.items():
                if isinstance(section_data, dict):
                    config[section_name] = {}
                    for key, value in section_data.items():
                        config[section_name][key] = str(value)
            
            with open(config_file, 'w') as f:
                config.write(f)
            return True
        except Exception as e:
            self.logger.error(f"Error saving INI configuration: {e}")
            return False
    
    def _load_session_config(self, session_dict: Dict[str, Any]):
        """Load session configuration from dictionary."""
        for key, value in session_dict.items():
            if hasattr(self.session, key):
                setattr(self.session, key, value)
    
    def _load_network_config(self, network_dict: Dict[str, Any]):
        """Load network configuration from dictionary."""
        for key, value in network_dict.items():
            if hasattr(self.network, key):
                setattr(self.network, key, value)
    
    def _load_security_config(self, security_dict: Dict[str, Any]):
        """Load security configuration from dictionary."""
        for key, value in security_dict.items():
            if hasattr(self.security, key):
                setattr(self.security, key, value)
    
    def _load_performance_config(self, performance_dict: Dict[str, Any]):
        """Load performance configuration from dictionary."""
        for key, value in performance_dict.items():
            if hasattr(self.performance, key):
                setattr(self.performance, key, value)
    
    def _load_general_config(self, config_dict: Dict[str, Any]):
        """Load general configuration settings."""
        # Load metadata
        if 'metadata' in config_dict:
            metadata = config_dict['metadata']
            self.environment = metadata.get('environment', self.environment)
            self.config_version = metadata.get('config_version', self.config_version)
            self.app_name = metadata.get('app_name', self.app_name)
            self.app_version = metadata.get('app_version', self.app_version)
            self.debug_mode = metadata.get('debug_mode', self.debug_mode)
            
            # Parse timestamps
            if 'created_at' in metadata:
                try:
                    self.created_at = datetime.fromisoformat(metadata['created_at'])
                except:
                    pass
            
            if 'updated_at' in metadata:
                try:
                    self.updated_at = datetime.fromisoformat(metadata['updated_at'])
                except:
                    pass
        
        # Load paths
        if 'paths' in config_dict:
            paths = config_dict['paths']
            self.store_path = paths.get('store_path', self.store_path)
            self.log_path = paths.get('log_path', self.log_path)
            self.data_dictionary_path = paths.get('data_dictionary_path', self.data_dictionary_path)
        
        # Load top-level settings for backward compatibility
        for key in ['environment', 'debug_mode', 'store_path', 'log_path', 'data_dictionary_path']:
            if key in config_dict:
                setattr(self, key, config_dict[key])
    
    def clone(self) -> 'FIXConfig':
        """
        Create a deep copy of the configuration.
        
        Returns:
            FIXConfig: Cloned configuration
        """
        config_dict = self.to_dict(include_sensitive=True)
        new_config = FIXConfig()
        new_config.load_from_dict(config_dict)
        return new_config
    
    def merge(self, other: 'FIXConfig', override: bool = True) -> 'FIXConfig':
        """
        Merge another configuration into this one.
        
        Args:
            other: Configuration to merge
            override: Whether to override existing values
            
        Returns:
            FIXConfig: Merged configuration (new instance)
        """
        # Create a copy of this configuration
        merged = self.clone()
        other_dict = other.to_dict(include_sensitive=True)
        
        if override:
            # Other configuration takes precedence
            merged.load_from_dict(other_dict)
        else:
            # Only fill in missing values
            merged_dict = merged.to_dict(include_sensitive=True)
            self._merge_dicts(merged_dict, other_dict, override=False)
            merged.load_from_dict(merged_dict)
        
        return merged
    
    def _merge_dicts(self, target: dict, source: dict, override: bool = True):
        """Recursively merge dictionaries."""
        for key, value in source.items():
            if key in target:
                if isinstance(target[key], dict) and isinstance(value, dict):
                    self._merge_dicts(target[key], value, override)
                elif override:
                    target[key] = value
            else:
                target[key] = value
    
    def apply_environment_overrides(self, env_overrides: Dict[str, Any]):
        """
        Apply environment-specific overrides.
        
        Args:
            env_overrides: Dictionary of environment overrides
        """
        # Apply overrides to configuration
        for key, value in env_overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif '.' in key:
                # Handle nested attributes (e.g., 'session.heartbeat_interval')
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        break
                else:
                    if hasattr(obj, parts[-1]):
                        setattr(obj, parts[-1], value)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"FIXConfig(environment={self.environment}, "
                f"session={self.session.sender_comp_id}->{self.session.target_comp_id}, "
                f"host={self.get_host()}:{self.get_port()}, "
                f"ssl={self.security.ssl_enabled})")
    
    def __repr__(self) -> str:
        """Detailed representation of configuration."""
        return (f"FIXConfig(environment='{self.environment}', "
                f"version='{self.config_version}', "
                f"debug={self.debug_mode}, "
                f"session=SessionConfig(sender='{self.session.sender_comp_id}', "
                f"target='{self.session.target_comp_id}', "
                f"heartbeat={self.session.heartbeat_interval}), "
                f"network=NetworkConfig(host='{self.get_host()}', "
                f"port={self.get_port()}), "
                f"security=SecurityConfig(ssl={self.security.ssl_enabled}, "
                f"auth={self.security.authentication_enabled}))")