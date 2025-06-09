"""
Message utilities for FIX protocol message handling and manipulation.
"""

import re
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import quickfix as fix

from .logging_config import get_logger


class FIXFieldType(Enum):
    """FIX field data types."""
    INT = "INT"
    FLOAT = "FLOAT"
    CHAR = "CHAR"
    STRING = "STRING"
    MULTIPLEVALUESTRING = "MULTIPLEVALUESTRING"
    BOOLEAN = "BOOLEAN"
    UTCTIMESTAMP = "UTCTIMESTAMP"
    UTCDATEONLY = "UTCDATEONLY"
    UTCTIMEONLY = "UTCTIMEONLY"
    DATA = "DATA"
    LENGTH = "LENGTH"


class MessageUtils:
    """
    Utility class for FIX message manipulation and analysis.
    
    Provides utilities for:
    - Message parsing and formatting
    - Field extraction and validation
    - Message comparison and analysis
    - Protocol-specific operations
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.field_definitions = self._load_field_definitions()
    
    def message_to_dict(self, message: fix.Message) -> Dict[str, Any]:
        """
        Convert FIX message to dictionary representation.
        
        Args:
            message: FIX message to convert
            
        Returns:
            Dict: Dictionary representation of message
        """
        result = {
            'header': {},
            'body': {},
            'trailer': {}
        }
        
        try:
            # Extract header fields
            header = message.getHeader()
            header_iterator = header.iterator()
            
            while header_iterator.hasNext():
                field = header_iterator.next()
                tag = field.getTag()
                value = field.getValue()
                field_name = self.get_field_name(tag)
                result['header'][f"{tag}_{field_name}"] = value
            
            # Extract body fields
            body_iterator = message.iterator()
            while body_iterator.hasNext():
                field = body_iterator.next()
                tag = field.getTag()
                value = field.getValue()
                field_name = self.get_field_name(tag)
                result['body'][f"{tag}_{field_name}"] = value
            
            # Extract trailer fields
            trailer = message.getTrailer()
            trailer_iterator = trailer.iterator()
            
            while trailer_iterator.hasNext():
                field = trailer_iterator.next()
                tag = field.getTag()
                value = field.getValue()
                field_name = self.get_field_name(tag)
                result['trailer'][f"{tag}_{field_name}"] = value
        
        except Exception as e:
            self.logger.error(f"Error converting message to dict: {e}")
            raise
        
        return result
    
    def dict_to_message(self, message_dict: Dict[str, Any]) -> fix.Message:
        """
        Convert dictionary to FIX message.
        
        Args:
            message_dict: Dictionary representation of message
            
        Returns:
            fix.Message: Reconstructed FIX message
        """
        message = fix.Message()
        
        try:
            # Set header fields
            header = message.getHeader()
            for field_key, value in message_dict.get('header', {}).items():
                tag = int(field_key.split('_')[0])
                header.setField(tag, str(value))
            
            # Set body fields
            for field_key, value in message_dict.get('body', {}).items():
                tag = int(field_key.split('_')[0])
                message.setField(tag, str(value))
            
            # Set trailer fields
            trailer = message.getTrailer()
            for field_key, value in message_dict.get('trailer', {}).items():
                tag = int(field_key.split('_')[0])
                trailer.setField(tag, str(value))
        
        except Exception as e:
            self.logger.error(f"Error converting dict to message: {e}")
            raise
        
        return message
    
    def extract_field_value(
        self,
        message: fix.Message,
        field_tag: int,
        default: Optional[str] = None
    ) -> Optional[str]:
        """
        Safely extract field value from message.
        
        Args:
            message: FIX message
            field_tag: Field tag to extract
            default: Default value if field not found
            
        Returns:
            Field value or default
        """
        try:
            return message.getField(field_tag)
        except fix.FieldNotFound:
            return default
        except Exception as e:
            self.logger.error(f"Error extracting field {field_tag}: {e}")
            return default
    
    def extract_typed_field_value(
        self,
        message: fix.Message,
        field_tag: int,
        field_type: FIXFieldType,
        default: Any = None
    ) -> Any:
        """
        Extract and convert field value to appropriate type.
        
        Args:
            message: FIX message
            field_tag: Field tag to extract
            field_type: Expected field type
            default: Default value if field not found
            
        Returns:
            Typed field value or default
        """
        raw_value = self.extract_field_value(message, field_tag)
        
        if raw_value is None:
            return default
        
        try:
            if field_type == FIXFieldType.INT:
                return int(raw_value)
            elif field_type == FIXFieldType.FLOAT:
                return float(raw_value)
            elif field_type == FIXFieldType.BOOLEAN:
                return raw_value.upper() in ('Y', 'TRUE', '1')
            elif field_type == FIXFieldType.CHAR:
                return raw_value[0] if raw_value else ''
            else:
                return raw_value
        
        except (ValueError, IndexError) as e:
            self.logger.error(f"Error converting field {field_tag} to {field_type}: {e}")
            return default
    
    def get_message_type(self, message: fix.Message) -> Optional[str]:
        """
        Get message type from message.
        
        Args:
            message: FIX message
            
        Returns:
            Message type or None
        """
        return self.extract_field_value(message, 35)  # MsgType
    
    def get_message_type_name(self, msg_type: str) -> str:
        """
        Get human-readable name for message type.
        
        Args:
            msg_type: FIX message type
            
        Returns:
            Human-readable message type name
        """
        msg_type_names = {
            '0': 'Heartbeat',
            '1': 'TestRequest',
            '2': 'ResendRequest',
            '3': 'Reject',
            '4': 'SequenceReset',
            '5': 'Logout',
            '6': 'IOI',
            '7': 'Advertisement',
            '8': 'ExecutionReport',
            '9': 'OrderCancelReject',
            'A': 'Logon',
            'B': 'News',
            'C': 'Email',
            'D': 'NewOrderSingle',
            'E': 'NewOrderList',
            'F': 'OrderCancelRequest',
            'G': 'OrderCancelReplaceRequest',
            'H': 'OrderStatusRequest',
            'I': 'Allocation',
            'J': 'ListCancelRequest',
            'K': 'ListExecute',
            'L': 'ListStatusRequest',
            'M': 'ListStatus',
            'N': 'AllocationAck',
            'P': 'DontKnowTrade',
            'Q': 'QuoteRequest',
            'R': 'Quote',
            'S': 'SettlementInstructions',
            'T': 'MarketDataRequest',
            'U': 'MarketDataIncrementalRefresh',
            'V': 'MarketDataRequest',
            'W': 'MarketDataSnapshot',
            'X': 'MarketDataIncrementalRefresh',
            'Y': 'MarketDataRequestReject',
            'Z': 'QuoteCancel',
            'a': 'QuoteStatusRequest',
            'b': 'MassQuoteAcknowledgement',
            'c': 'SecurityDefinitionRequest',
            'd': 'SecurityDefinition',
            'e': 'SecurityStatusRequest',
            'f': 'SecurityStatus',
            'g': 'TradingSessionStatusRequest',
            'h': 'TradingSessionStatus',
            'i': 'MassQuote',
            'j': 'BusinessMessageReject',
            'k': 'BidRequest',
            'l': 'BidResponse',
            'm': 'ListStrikePrice'
        }
        
        return msg_type_names.get(msg_type, f"Unknown ({msg_type})")
    
    def compare_messages(
        self,
        message1: fix.Message,
        message2: fix.Message,
        ignore_fields: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Compare two FIX messages and return differences.
        
        Args:
            message1: First message
            message2: Second message
            ignore_fields: List of field tags to ignore in comparison
            
        Returns:
            Dictionary containing comparison results
        """
        if ignore_fields is None:
            ignore_fields = [52, 60]  # SendingTime, TransactTime by default
        
        dict1 = self.message_to_dict(message1)
        dict2 = self.message_to_dict(message2)
        
        differences = {
            'are_equal': True,
            'field_differences': [],
            'only_in_message1': [],
            'only_in_message2': []
        }
        
        # Compare all sections
        for section in ['header', 'body', 'trailer']:
            fields1 = dict1.get(section, {})
            fields2 = dict2.get(section, {})
            
            all_fields = set(fields1.keys()) | set(fields2.keys())
            
            for field_key in all_fields:
                tag = int(field_key.split('_')[0])
                
                if tag in ignore_fields:
                    continue
                
                if field_key in fields1 and field_key in fields2:
                    if fields1[field_key] != fields2[field_key]:
                        differences['field_differences'].append({
                            'section': section,
                            'field': field_key,
                            'message1_value': fields1[field_key],
                            'message2_value': fields2[field_key]
                        })
                        differences['are_equal'] = False
                elif field_key in fields1:
                    differences['only_in_message1'].append({
                        'section': section,
                        'field': field_key,
                        'value': fields1[field_key]
                    })
                    differences['are_equal'] = False
                else:
                    differences['only_in_message2'].append({
                        'section': section,
                        'field': field_key,
                        'value': fields2[field_key]
                    })
                    differences['are_equal'] = False
        
        return differences
    
    def format_message_for_display(self, message: fix.Message) -> str:
        """
        Format FIX message for human-readable display.
        
        Args:
            message: FIX message to format
            
        Returns:
            Formatted string representation
        """
        msg_dict = self.message_to_dict(message)
        msg_type = self.get_message_type(message)
        msg_type_name = self.get_message_type_name(msg_type) if msg_type else "Unknown"
        
        lines = [
            f"=== FIX Message ({msg_type_name}) ===",
            "",
            "HEADER:"
        ]
        
        for field_key, value in msg_dict.get('header', {}).items():
            lines.append(f"  {field_key}: {value}")
        
        lines.extend(["", "BODY:"])
        for field_key, value in msg_dict.get('body', {}).items():
            lines.append(f"  {field_key}: {value}")
        
        if msg_dict.get('trailer'):
            lines.extend(["", "TRAILER:"])
            for field_key, value in msg_dict.get('trailer', {}).items():
                lines.append(f"  {field_key}: {value}")
        
        lines.append("=" * 40)
        
        return "\n".join(lines)
    
    def extract_repeating_group(
        self,
        message: fix.Message,
        group_count_tag: int,
        group_fields: List[int]
    ) -> List[Dict[int, str]]:
        """
        Extract repeating group from FIX message.
        
        Args:
            message: FIX message
            group_count_tag: Tag indicating number of groups
            group_fields: List of field tags in each group entry
            
        Returns:
            List of dictionaries, each representing one group entry
        """
        try:
            group_count = int(self.extract_field_value(message, group_count_tag, '0'))
            groups = []
            
            # This is a simplified implementation
            # Real implementation would need to properly parse FIX groups
            for i in range(group_count):
                group_entry = {}
                for field_tag in group_fields:
                    value = self.extract_field_value(message, field_tag)
                    if value is not None:
                        group_entry[field_tag] = value
                
                if group_entry:
                    groups.append(group_entry)
            
            return groups
        
        except Exception as e:
            self.logger.error(f"Error extracting repeating group: {e}")
            return []
    
    def validate_checksum(self, message: fix.Message) -> bool:
        """
        Validate FIX message checksum.
        
        Args:
            message: FIX message to validate
            
        Returns:
            bool: True if checksum is valid
        """
        try:
            # Get message as string
            msg_str = message.toString()
            
            # Find checksum field
            checksum_match = re.search(r'10=(\d{3})', msg_str)
            if not checksum_match:
                return False
            
            stated_checksum = int(checksum_match.group(1))
            
            # Calculate actual checksum (sum of all bytes except checksum field)
            checksum_pos = msg_str.find('10=')
            if checksum_pos == -1:
                return False
            
            message_without_checksum = msg_str[:checksum_pos]
            calculated_checksum = sum(ord(c) for c in message_without_checksum) % 256
            
            return stated_checksum == calculated_checksum
        
        except Exception as e:
            self.logger.error(f"Error validating checksum: {e}")
            return False
    
    def calculate_body_length(self, message: fix.Message) -> int:
        """
        Calculate FIX message body length.
        
        Args:
            message: FIX message
            
        Returns:
            int: Body length in bytes
        """
        try:
            msg_str = message.toString()
            
            # Find start of body (after BodyLength field)
            body_length_match = re.search(r'9=\d+\x01', msg_str)
            if not body_length_match:
                return 0
            
            body_start = body_length_match.end()
            
            # Find end of body (before CheckSum field)
            checksum_pos = msg_str.find('10=')
            if checksum_pos == -1:
                body_end = len(msg_str)
            else:
                body_end = checksum_pos
            
            return body_end - body_start
        
        except Exception as e:
            self.logger.error(f"Error calculating body length: {e}")
            return 0
    
    def get_field_name(self, field_tag: int) -> str:
        """
        Get field name for field tag.
        
        Args:
            field_tag: FIX field tag
            
        Returns:
            Field name or tag as string
        """
        return self.field_definitions.get(field_tag, str(field_tag))
    
    def sanitize_message_for_logging(self, message: fix.Message) -> str:
        """
        Sanitize message for safe logging (remove sensitive data).
        
        Args:
            message: FIX message to sanitize
            
        Returns:
            Sanitized message string
        """
        sensitive_fields = [
            96,   # RawData
            212,  # XmlData
            213,  # XmlDataLen
            354,  # EncodedText
            355,  # EncodedTextLen
        ]
        
        msg_dict = self.message_to_dict(message)
        
        # Remove sensitive fields
        for section in ['header', 'body', 'trailer']:
            section_data = msg_dict.get(section, {})
            for field_key in list(section_data.keys()):
                tag = int(field_key.split('_')[0])
                if tag in sensitive_fields:
                    section_data[field_key] = "[REDACTED]"
        
        # Reconstruct message and return formatted version
        sanitized_message = self.dict_to_message(msg_dict)
        return self.format_message_for_display(sanitized_message)
    
    def _load_field_definitions(self) -> Dict[int, str]:
        """Load FIX field tag to name mappings."""
        return {
            1: "Account",
            2: "AdvId",
            3: "AdvRefID",
            4: "AdvSide",
            5: "AdvTransType",
            6: "AvgPx",
            7: "BeginSeqNo",
            8: "BeginString",
            9: "BodyLength",
            10: "CheckSum",
            11: "ClOrdID",
            12: "Commission",
            13: "CommType",
            14: "CumQty",
            15: "Currency",
            16: "EndSeqNo",
            17: "ExecID",
            18: "ExecInst",
            19: "ExecRefID",
            20: "ExecTransType",
            21: "HandlInst",
            22: "SecurityIDSource",
            23: "IOIID",
            24: "IOIQltyInd",
            25: "IOIRefID",
            26: "IOIQty",
            27: "IOITransType",
            28: "IOIShares",
            29: "LastCapacity",
            30: "LastMkt",
            31: "LastPx",
            32: "LastQty",
            33: "NoLinesOfText",
            34: "MsgSeqNum",
            35: "MsgType",
            36: "NewSeqNo",
            37: "OrderID",
            38: "OrderQty",
            39: "OrdStatus",
            40: "OrdType",
            41: "OrigClOrdID",
            42: "OrigTime",
            43: "PossDupFlag",
            44: "Price",
            45: "RefSeqNum",
            46: "RelatdSym",
            47: "Rule80A",
            48: "SecurityID",
            49: "SenderCompID",
            50: "SenderSubID",
            51: "SendingDate",
            52: "SendingTime",
            53: "Quantity",
            54: "Side",
            55: "Symbol",
            56: "TargetCompID",
            57: "TargetSubID",
            58: "Text",
            59: "TimeInForce",
            60: "TransactTime",
            61: "Urgency",
            62: "ValidUntilTime",
            63: "SettlmntTyp",
            64: "FutSettDate",
            65: "SymbolSfx",
            66: "ListID",
            67: "ListSeqNo",
            68: "TotNoOrders",
            69: "ListExecInst",
            70: "AllocID",
            71: "AllocTransType",
            72: "RefAllocID",
            73: "NoOrders",
            74: "AvgPrxPrecision",
            75: "TradeDate",
            76: "ExecBroker",
            77: "PositionEffect",
            78: "NoAllocs",
            79: "AllocAccount",
            80: "AllocQty",
            81: "ProcessCode",
            82: "NoRpts",
            83: "RptSeq",
            84: "CxlQty",
            85: "NoDlvyInst",
            86: "DlvyInst",
            87: "AllocStatus",
            88: "AllocRejCode",
            89: "Signature",
            90: "SecureDataLen",
            91: "SecureData",
            92: "BrokerOfCredit",
            93: "SignatureLength",
            94: "EmailType",
            95: "RawDataLength",
            96: "RawData",
            97: "PossResend",
            98: "EncryptMethod",
            99: "StopPx",
            100: "ExDestination",
            101: "CxlRejReason",
            102: "CxlRejResponseTo",
            103: "OrdRejReason",
            104: "IOIQualifier",
            105: "WaveNo",
            106: "Issuer",
            107: "SecurityDesc",
            108: "HeartBtInt",
            109: "ClientID",
            110: "MinQty",
            111: "MaxFloor",
            112: "TestReqID",
            113: "ReportToExch",
            114: "LocateReqd",
            115: "OnBehalfOfCompID",
            116: "OnBehalfOfSubID",
            117: "QuoteID",
            118: "NetMoney",
            119: "SettlCurrAmt",
            120: "SettlCurrency"},
            # ... continues with more field definitions