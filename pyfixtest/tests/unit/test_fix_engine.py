"""
Enhanced test configuration extensions for the FIX trading system.

This module provides additional functionality for test configuration management,
including test data generation, advanced mock behaviors, test result tracking,
and enhanced validation capabilities.
"""

import asyncio
import json
import pickle
import threading
import time
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Union, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod

from ..utils.logging_config import get_logger


class TestDataType(Enum):
    """Types of test data that can be generated."""
    ORDER = "order"
    EXECUTION = "execution"
    MARKET_DATA = "market_data"
    SECURITY_DEFINITION = "security_definition"
    POSITION = "position"
    ACCOUNT = "account"
    TRADE_REPORT = "trade_report"
    RISK_LIMIT = "risk_limit"


@dataclass
class TestDataSpec:
    """Specification for generating test data."""
    data_type: TestDataType
    count: int = 100
    start_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=1))
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    price_range: tuple = (50.0, 500.0)
    quantity_range: tuple = (100, 10000)
    randomization_seed: Optional[int] = None
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


class TestDataGenerator:
    """Generates realistic test data for FIX testing scenarios."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize test data generator.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        self.logger = get_logger(__name__)
        self.random = random.Random(seed)
        self.order_id_counter = 1
        self.exec_id_counter = 1
        self.trade_id_counter = 1
        
        # Market data state for realistic price movements
        self.current_prices = {}
        self.price_trends = {}
    
    def generate_test_data(self, spec: TestDataSpec) -> List[Dict[str, Any]]:
        """
        Generate test data based on specification.
        
        Args:
            spec: Test data specification
            
        Returns:
            List[Dict]: Generated test data
        """
        if spec.randomization_seed is not None:
            self.random.seed(spec.randomization_seed)
        
        generators = {
            TestDataType.ORDER: self._generate_orders,
            TestDataType.EXECUTION: self._generate_executions,
            TestDataType.MARKET_DATA: self._generate_market_data,
            TestDataType.SECURITY_DEFINITION: self._generate_security_definitions,
            TestDataType.POSITION: self._generate_positions,
            TestDataType.ACCOUNT: self._generate_accounts,
            TestDataType.TRADE_REPORT: self._generate_trade_reports,
            TestDataType.RISK_LIMIT: self._generate_risk_limits
        }
        
        generator = generators.get(spec.data_type)
        if not generator:
            raise ValueError(f"Unknown test data type: {spec.data_type}")
        
        return generator(spec)
    
    def _generate_orders(self, spec: TestDataSpec) -> List[Dict[str, Any]]:
        """Generate test order data."""
        orders = []
        
        for _ in range(spec.count):
            symbol = self.random.choice(spec.symbols)
            side = self.random.choice(['1', '2'])  # Buy/Sell
            order_type = self.random.choice(['1', '2', '3', '4'])  # Market, Limit, Stop, Stop Limit
            
            price = self.random.uniform(*spec.price_range)
            quantity = self.random.randint(*spec.quantity_range)
            
            order = {
                'ClOrdID': f"TEST_ORDER_{self.order_id_counter:06d}",
                'Symbol': symbol,
                'Side': side,
                'OrderQty': str(quantity),
                'OrdType': order_type,
                'TransactTime': self._random_timestamp(spec.start_date, spec.end_date),
                'TimeInForce': self.random.choice(['0', '1', '3', '4']),  # DAY, GTC, IOC, FOK
            }
            
            if order_type in ['2', '3', '4']:  # Orders that need price
                order['Price'] = f"{price:.2f}"
            
            if order_type in ['3', '4']:  # Stop orders
                order['StopPx'] = f"{price * self.random.uniform(0.95, 1.05):.2f}"
            
            # Add custom parameters
            for key, value in spec.custom_parameters.items():
                order[key] = value
            
            orders.append(order)
            self.order_id_counter += 1
        
        return orders
    
    def _generate_executions(self, spec: TestDataSpec) -> List[Dict[str, Any]]:
        """Generate test execution data."""
        executions = []
        
        for _ in range(spec.count):
            symbol = self.random.choice(spec.symbols)
            side = self.random.choice(['1', '2'])
            
            price = self.random.uniform(*spec.price_range)
            quantity = self.random.randint(*spec.quantity_range)
            
            execution = {
                'ExecID': f"TEST_EXEC_{self.exec_id_counter:06d}",
                'OrderID': f"TEST_ORDER_{self.random.randint(1, spec.count):06d}",
                'ClOrdID': f"TEST_ORDER_{self.random.randint(1, spec.count):06d}",
                'Symbol': symbol,
                'Side': side,
                'LastQty': str(quantity),
                'LastPx': f"{price:.2f}",
                'AvgPx': f"{price:.2f}",
                'CumQty': str(quantity),
                'LeavesQty': '0',
                'ExecType': self.random.choice(['0', '1', '2', 'F']),  # New, Partial, Fill, Trade
                'OrdStatus': self.random.choice(['0', '1', '2']),  # New, Partial, Filled
                'TransactTime': self._random_timestamp(spec.start_date, spec.end_date),
                'ExecTransType': '0',  # New
            }
            
            executions.append(execution)
            self.exec_id_counter += 1
        
        return executions
    
    def _generate_market_data(self, spec: TestDataSpec) -> List[Dict[str, Any]]:
        """Generate test market data."""
        market_data = []
        
        # Initialize current prices if not already done
        for symbol in spec.symbols:
            if symbol not in self.current_prices:
                self.current_prices[symbol] = self.random.uniform(*spec.price_range)
                self.price_trends[symbol] = self.random.choice([-1, 0, 1])
        
        time_delta = (spec.end_date - spec.start_date) / spec.count
        
        for i in range(spec.count):
            for symbol in spec.symbols:
                # Simulate realistic price movement
                current_price = self.current_prices[symbol]
                trend = self.price_trends[symbol]
                
                # Random walk with trend
                change_percent = self.random.gauss(0, 0.01)  # 1% volatility
                if trend != 0:
                    change_percent += trend * 0.001  # Small trend bias
                
                new_price = current_price * (1 + change_percent)
                new_price = max(new_price, spec.price_range[0])
                new_price = min(new_price, spec.price_range[1])
                
                self.current_prices[symbol] = new_price
                
                # Occasionally change trend
                if self.random.random() < 0.05:
                    self.price_trends[symbol] = self.random.choice([-1, 0, 1])
                
                # Generate bid/ask around current price
                spread = new_price * 0.001  # 0.1% spread
                bid_price = new_price - spread / 2
                ask_price = new_price + spread / 2
                
                bid_size = self.random.randint(100, 5000)
                ask_size = self.random.randint(100, 5000)
                
                timestamp = spec.start_date + i * time_delta
                
                market_data.append({
                    'Symbol': symbol,
                    'MDEntryType': '0',  # Bid
                    'MDEntryPx': f"{bid_price:.2f}",
                    'MDEntrySize': str(bid_size),
                    'MDEntryTime': timestamp.strftime('%H:%M:%S'),
                    'MDEntryDate': timestamp.strftime('%Y%m%d'),
                })
                
                market_data.append({
                    'Symbol': symbol,
                    'MDEntryType': '1',  # Ask
                    'MDEntryPx': f"{ask_price:.2f}",
                    'MDEntrySize': str(ask_size),
                    'MDEntryTime': timestamp.strftime('%H:%M:%S'),
                    'MDEntryDate': timestamp.strftime('%Y%m%d'),
                })
        
        return market_data
    
    def _generate_security_definitions(self, spec: TestDataSpec) -> List[Dict[str, Any]]:
        """Generate test security definition data."""
        securities = []
        
        security_types = ['CS', 'PS', 'CD', 'WI', 'CB']  # Common stock, Preferred, etc.
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD']
        exchanges = ['NASDAQ', 'NYSE', 'BATS', 'ARCA']
        
        for symbol in spec.symbols:
            security = {
                'Symbol': symbol,
                'SecurityID': f"{symbol}_ID",
                'SecurityIDSource': '8',  # Exchange Symbol
                'SecurityType': self.random.choice(security_types),
                'Currency': self.random.choice(currencies),
                'SecurityExchange': self.random.choice(exchanges),
                'SecurityDesc': f"{symbol} Common Stock",
                'MaturityDate': '',
                'IssueDate': '',
                'SecuritySubType': '',
                'Product': '1',  # Equity
            }
            
            securities.append(security)
        
        return securities
    
    def _generate_positions(self, spec: TestDataSpec) -> List[Dict[str, Any]]:
        """Generate test position data."""
        positions = []
        
        accounts = [f"TEST_ACCOUNT_{i:03d}" for i in range(1, 11)]
        
        for symbol in spec.symbols:
            for account in accounts:
                if self.random.random() > 0.7:  # 30% chance of having position
                    continue
                
                quantity = self.random.randint(-10000, 10000)
                avg_price = self.random.uniform(*spec.price_range)
                
                position = {
                    'Account': account,
                    'Symbol': symbol,
                    'LongQty': str(max(quantity, 0)),
                    'ShortQty': str(abs(min(quantity, 0))),
                    'PosQtyStatus': '0',  # Submitted
                    'PosMaintRptID': f"POS_{account}_{symbol}",
                    'PosReqResult': '0',  # Valid request
                    'ClearingBusinessDate': datetime.now(timezone.utc).strftime('%Y%m%d'),
                    'SettlPrice': f"{avg_price:.2f}",
                }
                
                positions.append(position)
        
        return positions
    
    def _generate_accounts(self, spec: TestDataSpec) -> List[Dict[str, Any]]:
        """Generate test account data."""
        accounts = []
        
        for i in range(spec.count):
            account = {
                'Account': f"TEST_ACCOUNT_{i+1:03d}",
                'CustomerOrFirm': '0',  # Customer
                'AcctIDSource': '99',  # Other
                'AccountType': self.random.choice(['1', '2', '3']),  # Individual, Corporate, etc.
                'AccruedInterestAmt': f"{self.random.uniform(0, 1000):.2f}",
                'AccruedInterestRate': f"{self.random.uniform(0, 0.1):.4f}",
                'Commission': f"{self.random.uniform(0, 50):.2f}",
                'Currency': 'USD',
            }
            
            accounts.append(account)
        
        return accounts
    
    def _generate_trade_reports(self, spec: TestDataSpec) -> List[Dict[str, Any]]:
        """Generate test trade report data."""
        trade_reports = []
        
        for _ in range(spec.count):
            symbol = self.random.choice(spec.symbols)
            side = self.random.choice(['1', '2'])
            
            price = self.random.uniform(*spec.price_range)
            quantity = self.random.randint(*spec.quantity_range)
            
            trade_report = {
                'TradeReportID': f"TRADE_{self.trade_id_counter:06d}",
                'TradeReportType': '0',  # Submit
                'TrdType': self.random.choice(['0', '1', '2']),  # Regular, Block, EFP
                'Symbol': symbol,
                'Side': side,
                'LastQty': str(quantity),
                'LastPx': f"{price:.2f}",
                'TradeDate': self._random_timestamp(spec.start_date, spec.end_date).strftime('%Y%m%d'),
                'TransactTime': self._random_timestamp(spec.start_date, spec.end_date),
                'PreviouslyReported': 'N',
                'TrdMatchID': f"MATCH_{self.trade_id_counter:06d}",
            }
            
            trade_reports.append(trade_report)
            self.trade_id_counter += 1
        
        return trade_reports
    
    def _generate_risk_limits(self, spec: TestDataSpec) -> List[Dict[str, Any]]:
        """Generate test risk limit data."""
        risk_limits = []
        
        accounts = [f"TEST_ACCOUNT_{i:03d}" for i in range(1, 11)]
        risk_types = ['1', '2', '3', '4']  # Position, Order, Net Position, etc.
        
        for account in accounts:
            for symbol in spec.symbols:
                for risk_type in risk_types:
                    if self.random.random() > 0.5:  # 50% chance
                        continue
                    
                    limit_amount = self.random.uniform(100000, 10000000)
                    warning_level = limit_amount * 0.8
                    
                    risk_limit = {
                        'RiskLimitReportID': f"RISK_{account}_{symbol}_{risk_type}",
                        'RiskLimitRequestID': f"REQ_{account}_{symbol}_{risk_type}",
                        'Account': account,
                        'Symbol': symbol,
                        'RiskLimitType': risk_type,
                        'RiskLimitAmount': f"{limit_amount:.2f}",
                        'RiskLimitCurrency': 'USD',
                        'RiskLimitPlatform': 'TRADING_SYSTEM',
                        'RiskLimitScope': '1',  # Firm level
                        'RiskWarningLevel': f"{warning_level:.2f}",
                    }
                    
                    risk_limits.append(risk_limit)
        
        return risk_limits
    
    def _random_timestamp(self, start_date: datetime, end_date: datetime) -> datetime:
        """Generate random timestamp between start and end dates."""
        delta = end_date - start_date
        random_seconds = self.random.uniform(0, delta.total_seconds())
        return start_date + timedelta(seconds=random_seconds)


class TestMetrics:
    """Tracks and manages test execution metrics."""
    
    def __init__(self):
        """Initialize test metrics tracker."""
        self.logger = get_logger(__name__)
        self.metrics = {}
        self.timers = {}
        self.counters = {}
        self.histograms = {}
        self._lock = threading.Lock()
    
    def start_timer(self, name: str) -> float:
        """
        Start a named timer.
        
        Args:
            name: Timer name
            
        Returns:
            float: Start time
        """
        start_time = time.time()
        with self._lock:
            self.timers[name] = start_time
        return start_time
    
    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and record duration.
        
        Args:
            name: Timer name
            
        Returns:
            float: Duration in seconds
        """
        end_time = time.time()
        with self._lock:
            if name not in self.timers:
                self.logger.warning(f"Timer '{name}' was not started")
                return 0.0
            
            duration = end_time - self.timers[name]
            del self.timers[name]
            
            # Record in histogram
            if name not in self.histograms:
                self.histograms[name] = []
            self.histograms[name].append(duration)
            
            return duration
    
    def increment_counter(self, name: str, value: int = 1):
        """
        Increment a named counter.
        
        Args:
            name: Counter name
            value: Increment value
        """
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + value
    
    def record_value(self, name: str, value: float):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
        """
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
    
    def get_counter(self, name: str) -> int:
        """Get counter value."""
        with self._lock:
            return self.counters.get(name, 0)
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """
        Get timer statistics.
        
        Args:
            name: Timer name
            
        Returns:
            Dict: Timer statistics
        """
        with self._lock:
            if name not in self.histograms or not self.histograms[name]:
                return {}
            
            values = self.histograms[name]
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'p95': self._percentile(values, 95),
                'p99': self._percentile(values, 99),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0
            }
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """
        Get metric statistics.
        
        Args:
            name: Metric name
            
        Returns:
            Dict: Metric statistics
        """
        with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = self.metrics[name]
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'sum': sum(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all metrics and statistics."""
        with self._lock:
            stats = {
                'counters': self.counters.copy(),
                'timers': {},
                'metrics': {}
            }
            
            for timer_name in self.histograms:
                stats['timers'][timer_name] = self.get_timer_stats(timer_name)
            
            for metric_name in self.metrics:
                stats['metrics'][metric_name] = self.get_metric_stats(metric_name)
            
            return stats
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()
            self.timers.clear()
            self.counters.clear()
            self.histograms.clear()
    
    def export_to_file(self, file_path: str, format: str = 'json'):
        """
        Export metrics to file.
        
        Args:
            file_path: Output file path
            format: Export format (json, csv)
        """
        stats = self.get_all_stats()
        
        if format.lower() == 'json':
            with open(file_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
        elif format.lower() == 'csv':
            import csv
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Type', 'Name', 'Statistic', 'Value'])
                
                # Write counters
                for name, value in stats['counters'].items():
                    writer.writerow(['Counter', name, 'value', value])
                
                # Write timer stats
                for name, timer_stats in stats['timers'].items():
                    for stat, value in timer_stats.items():
                        writer.writerow(['Timer', name, stat, value])
                
                # Write metric stats
                for name, metric_stats in stats['metrics'].items():
                    for stat, value in metric_stats.items():
                        writer.writerow(['Metric', name, stat, value])
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = k - f
        
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations."""
        self.start_timer(name)
        try:
            yield
        finally:
            self.stop_timer(name)


class TestResultRecorder:
    """Records and manages test execution results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize test result recorder.
        
        Args:
            output_dir: Directory for test result files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        
        self.test_results = []
        self.test_session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now(timezone.utc)
        
        self._lock = threading.Lock()
    
    def record_test_start(self, test_name: str, test_type: str, metadata: Dict[str, Any] = None):
        """
        Record test start.
        
        Args:
            test_name: Name of the test
            test_type: Type of test
            metadata: Additional test metadata
        """
        with self._lock:
            test_record = {
                'test_name': test_name,
                'test_type': test_type,
                'session_id': self.test_session_id,
                'start_time': datetime.now(timezone.utc),
                'end_time': None,
                'duration_seconds': None,
                'status': 'RUNNING',
                'result': None,
                'error_message': None,
                'metadata': metadata or {},
                'metrics': {},
                'artifacts': []
            }
            
            self.test_results.append(test_record)
            self.logger.info(f"Started test: {test_name} ({test_type})")
            
            return len(self.test_results) - 1  # Return index
    
    def record_test_end(self, test_index: int, status: str, result: Any = None, error_message: str = None, metrics: Dict[str, Any] = None):
        """
        Record test end.
        
        Args:
            test_index: Index of test record
            status: Test status (PASSED, FAILED, SKIPPED, ERROR)
            result: Test result data
            error_message: Error message if failed
            metrics: Test metrics
        """
        with self._lock:
            if test_index >= len(self.test_results):
                self.logger.error(f"Invalid test index: {test_index}")
                return
            
            test_record = self.test_results[test_index]
            end_time = datetime.now(timezone.utc)
            
            test_record.update({
                'end_time': end_time,
                'duration_seconds': (end_time - test_record['start_time']).total_seconds(),
                'status': status,
                'result': result,
                'error_message': error_message,
                'metrics': metrics or {}
            })
            
            self.logger.info(f"Completed test: {test_record['test_name']} - {status}")
    
    def add_test_artifact(self, test_index: int, artifact_path: str, artifact_type: str, description: str = ""):
        """
        Add test artifact.
        
        Args:
            test_index: Index of test record
            artifact_path: Path to artifact file
            artifact_type: Type of artifact (log, screenshot, data, etc.)
            description: Artifact description
        """
        with self._lock:
            if test_index >= len(self.test_results):
                return
            
            artifact = {
                'path': artifact_path,
                'type': artifact_type,
                'description': description,
                'created_at': datetime.now(timezone.utc)
            }
            
            self.test_results[test_index]['artifacts'].append(artifact)
    
    def generate_test_report(self, report_format: str = 'json') -> str:
        """
        Generate test execution report.
        
        Args:
            report_format: Report format (json, html, xml)
            
        Returns:
            str: Path to generated report
        """
        session_end_time = datetime.now(timezone.utc)
        session_duration = (session_end_time - self.session_start_time).total_seconds()
        
        # Calculate summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for t in self.test_results if t['status'] == 'PASSED')
        failed_tests = sum(1 for t in self.test_results if t['status'] == 'FAILED')
        error_tests = sum(1 for t in self.test_results if t['status'] == 'ERROR')
        skipped_tests = sum(1 for t in self.test_results if t['status'] == 'SKIPPED')
        
        durations = [t['duration_seconds'] for t in self.test_results if t['duration_seconds'] is not None]
        avg_duration = statistics.mean(durations) if durations else 0
        
        report_data = {
            'session_id': self.test_session_id,
            'session_start_time': self.session_start_time.isoformat(),
            'session_end_time': session_end_time.isoformat(),
            'session_duration_seconds': session_duration,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'error_tests': error_tests,
                'skipped_tests': skipped_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'average_test_duration': avg_duration
            },
            'test_results': self.test_results
        }
        
        if report_format.lower() == 'json':
            report_file = self.output_dir / f"test_report_{self.test_session_id}.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        
        elif report_format.lower() == 'html':
            report_file = self.output_dir / f"test_report_{self.test_session_id}.html"
            html_content = self._generate_html_report(report_data)
            with open(report_file, 'w') as f:
                f.write(html_content)
        
        elif report_format.lower() == 'xml':
            report_file = self.output_dir / f"test_report_{self.test_session_id}.xml"
            xml_content = self._generate_xml_report(report_data)
            with open(report_file, 'w') as f:
                f.write(xml_content)
        
        else:
            raise ValueError(f"Unsupported report format: {report_format}")
        
        self.logger.info(f"Generated test report: {report_file}")
        return str(report_file)
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML test report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Execution Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .passed {{ color: green; }}