"""
Time utilities for FIX protocol timestamp handling.
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Union


def get_utc_timestamp() -> str:
    """
    Get current UTC timestamp in FIX format.
    
    Returns:
        str: UTC timestamp in FIX YYYYMMDD-HH:MM:SS format
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d-%H:%M:%S")


def get_utc_timestamp_with_millis() -> str:
    """
    Get current UTC timestamp with milliseconds in FIX format.
    
    Returns:
        str: UTC timestamp in FIX YYYYMMDD-HH:MM:SS.sss format
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d-%H:%M:%S.%f")[:-3]


def format_fix_time(dt: datetime) -> str:
    """
    Format datetime object to FIX timestamp format.
    
    Args:
        dt: Datetime object to format
        
    Returns:
        str: FIX formatted timestamp
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y%m%d-%H:%M:%S")


def parse_fix_time(fix_timestamp: str) -> datetime:
    """
    Parse FIX timestamp to datetime object.
    
    Args:
        fix_timestamp: FIX timestamp string
        
    Returns:
        datetime: Parsed datetime object in UTC
    """
    try:
        # Handle format with milliseconds
        if '.' in fix_timestamp:
            return datetime.strptime(fix_timestamp, "%Y%m%d-%H:%M:%S.%f").replace(tzinfo=timezone.utc)
        else:
            return datetime.strptime(fix_timestamp, "%Y%m%d-%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(f"Invalid FIX timestamp format: {fix_timestamp}") from e


def get_sending_time() -> str:
    """
    Get current time for SendingTime field (52).
    
    Returns:
        str: Current time in FIX format
    """
    return get_utc_timestamp()


def is_timestamp_recent(fix_timestamp: str, max_age_seconds: float = 60.0) -> bool:
    """
    Check if FIX timestamp is recent (within specified age).
    
    Args:
        fix_timestamp: FIX timestamp to check
        max_age_seconds: Maximum age in seconds
        
    Returns:
        bool: True if timestamp is recent
    """
    try:
        timestamp_dt = parse_fix_time(fix_timestamp)
        current_dt = datetime.now(timezone.utc)
        age_seconds = (current_dt - timestamp_dt).total_seconds()
        return age_seconds <= max_age_seconds
    except Exception:
        return False


def time_difference_seconds(timestamp1: str, timestamp2: str) -> float:
    """
    Calculate time difference between two FIX timestamps.
    
    Args:
        timestamp1: First FIX timestamp
        timestamp2: Second FIX timestamp
        
    Returns:
        float: Time difference in seconds (timestamp2 - timestamp1)
    """
    dt1 = parse_fix_time(timestamp1)
    dt2 = parse_fix_time(timestamp2)
    return (dt2 - dt1).total_seconds()


def add_seconds_to_fix_time(fix_timestamp: str, seconds: float) -> str:
    """
    Add seconds to FIX timestamp.
    
    Args:
        fix_timestamp: Original FIX timestamp
        seconds: Seconds to add (can be negative)
        
    Returns:
        str: New FIX timestamp
    """
    dt = parse_fix_time(fix_timestamp)
    new_dt = dt + timedelta(seconds=seconds)
    return format_fix_time(new_dt)


def get_market_open_time(date: Optional[datetime] = None) -> str:
    """
    Get market open time for given date.
    
    Args:
        date: Date for market open (defaults to today)
        
    Returns:
        str: Market open time in FIX format
    """
    if date is None:
        date = datetime.now(timezone.utc)
    
    # Assuming 9:30 AM UTC market open (simplified)
    market_open = date.replace(hour=9, minute=30, second=0, microsecond=0)
    return format_fix_time(market_open)


def get_market_close_time(date: Optional[datetime] = None) -> str:
    """
    Get market close time for given date.
    
    Args:
        date: Date for market close (defaults to today)
        
    Returns:
        str: Market close time in FIX format
    """
    if date is None:
        date = datetime.now(timezone.utc)
    
    # Assuming 4:00 PM UTC market close (simplified)
    market_close = date.replace(hour=16, minute=0, second=0, microsecond=0)
    return format_fix_time(market_close)


def is_market_hours(timestamp: Optional[str] = None) -> bool:
    """
    Check if timestamp falls within market hours.
    
    Args:
        timestamp: FIX timestamp to check (defaults to current time)
        
    Returns:
        bool: True if within market hours
    """
    if timestamp is None:
        timestamp = get_utc_timestamp()
    
    dt = parse_fix_time(timestamp)
    market_open = parse_fix_time(get_market_open_time(dt))
    market_close = parse_fix_time(get_market_close_time(dt))
    
    return market_open <= dt <= market_close

def get_fix_utcdate() -> str:
    """
    Get current UTC date in FIX format (YYYYMMDD).
    
    Returns:
        str: Current date in FIX format
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def get_fix_utctime() -> str:
    """
    Get current UTC time in FIX format (HH:MM:SS).
    
    Returns:
        str: Current time in FIX format
    """
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def parse_fix_date(fix_date: str) -> datetime:
    """
    Parse FIX date string to datetime object.
    
    Args:
        fix_date: FIX date string (YYYYMMDD)
        
    Returns:
        datetime: Parsed date as datetime object
    """
    try:
        return datetime.strptime(fix_date, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(f"Invalid FIX date format: {fix_date}") from e


def format_fix_date(dt: datetime) -> str:
    """
    Format datetime object to FIX date format.
    
    Args:
        dt: Datetime object to format
        
    Returns:
        str: FIX formatted date (YYYYMMDD)
    """
    return dt.strftime("%Y%m%d")


def get_business_days_between(start_date: str, end_date: str) -> int:
    """
    Calculate business days between two FIX dates.
    
    Args:
        start_date: Start date in FIX format (YYYYMMDD)
        end_date: End date in FIX format (YYYYMMDD)
        
    Returns:
        int: Number of business days
    """
    start_dt = parse_fix_date(start_date)
    end_dt = parse_fix_date(end_date)
    
    # Simple business day calculation (excludes weekends)
    business_days = 0
    current_date = start_dt
    
    while current_date <= end_dt:
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            business_days += 1
        current_date += timedelta(days=1)
    
    return business_days


def is_weekend(fix_date: str) -> bool:
    """
    Check if FIX date falls on weekend.
    
    Args:
        fix_date: Date in FIX format (YYYYMMDD)
        
    Returns:
        bool: True if weekend
    """
    dt = parse_fix_date(fix_date)
    return dt.weekday() >= 5  # Saturday = 5, Sunday = 6


def add_business_days(fix_date: str, days: int) -> str:
    """
    Add business days to FIX date.
    
    Args:
        fix_date: Starting date in FIX format
        days: Number of business days to add
        
    Returns:
        str: New date in FIX format
    """
    dt = parse_fix_date(fix_date)
    added_days = 0
    
    while added_days < days:
        dt += timedelta(days=1)
        if dt.weekday() < 5:  # Skip weekends
            added_days += 1
    
    return format_fix_date(dt)