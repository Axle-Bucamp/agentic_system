"""
Security utilities for the Agentic Trading System.
"""
import hashlib
import hmac
import secrets
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt

from core.config import settings
from core.logging import log

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class SecurityManager:
    """Manages security operations for the trading system."""
    
    def __init__(self):
        self.secret_key = settings.mcp_api_key or "default-secret-key"
        self.encryption_key = self._derive_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from secret."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'trading_system_salt',
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(self.secret_key.encode()))
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data like API keys."""
        try:
            encrypted_data = self.cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            log.error(f"Encryption error: {e}")
            raise
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            log.error(f"Decryption error: {e}")
            raise
    
    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)
    
    def generate_jwt_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """Generate a JWT token for user authentication."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        payload = {
            "user_id": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": "trading-system"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            log.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError:
            log.warning("Invalid JWT token")
            return None
    
    def generate_hmac_signature(self, data: str, secret: str) -> str:
        """Generate HMAC signature for API requests."""
        return hmac.new(
            secret.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_hmac_signature(self, data: str, signature: str, secret: str) -> bool:
        """Verify HMAC signature."""
        expected_signature = self.generate_hmac_signature(data, secret)
        return hmac.compare_digest(signature, expected_signature)

class InputValidator:
    """Validates and sanitizes user inputs."""
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Validate ticker symbol."""
        if not ticker or not isinstance(ticker, str):
            return False
        
        # Check if ticker is in supported assets
        return ticker.upper() in settings.supported_assets
    
    @staticmethod
    def validate_quantity(quantity: float) -> bool:
        """Validate trade quantity."""
        return isinstance(quantity, (int, float)) and quantity > 0
    
    @staticmethod
    def validate_price(price: float) -> bool:
        """Validate price."""
        return isinstance(price, (int, float)) and price > 0
    
    @staticmethod
    def validate_action(action: str) -> bool:
        """Validate trade action."""
        return action.upper() in ["BUY", "SELL", "HOLD"]
    
    @staticmethod
    def sanitize_string(input_str: str) -> str:
        """Sanitize string input."""
        if not isinstance(input_str, str):
            return ""
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`', '$']
        for char in dangerous_chars:
            input_str = input_str.replace(char, '')
        
        return input_str.strip()
    
    @staticmethod
    def validate_portfolio_id(portfolio_id: str) -> bool:
        """Validate portfolio ID."""
        if not portfolio_id or not isinstance(portfolio_id, str):
            return False
        
        # Allow only alphanumeric characters and hyphens
        return portfolio_id.replace('-', '').replace('_', '').isalnum()
    
    @staticmethod
    def validate_wallet_address(address: str) -> bool:
        """Validate wallet address format."""
        if not address or not isinstance(address, str):
            return False
        
        # Basic validation - should be 40+ characters for Ethereum addresses
        return len(address) >= 40 and address.replace('0x', '').isalnum()

class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self):
        self.requests = {}  # {client_id: [timestamps]}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = datetime.utcnow()
    
    def is_allowed(self, client_id: str, max_requests: int, window_seconds: int) -> bool:
        """Check if request is allowed based on rate limits."""
        current_time = datetime.utcnow()
        
        # Cleanup old entries periodically
        if (current_time - self.last_cleanup).total_seconds() > self.cleanup_interval:
            self._cleanup_old_entries()
            self.last_cleanup = current_time
        
        # Get client request history
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        client_requests = self.requests[client_id]
        
        # Remove old requests outside the window
        cutoff_time = current_time - timedelta(seconds=window_seconds)
        client_requests[:] = [req_time for req_time in client_requests if req_time > cutoff_time]
        
        # Check if under limit
        if len(client_requests) < max_requests:
            client_requests.append(current_time)
            return True
        
        return False
    
    def _cleanup_old_entries(self):
        """Clean up old request entries."""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(hours=1)
        
        for client_id in list(self.requests.keys()):
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > cutoff_time
            ]
            
            # Remove empty entries
            if not self.requests[client_id]:
                del self.requests[client_id]

class AuditLogger:
    """Logs security-relevant events for audit purposes."""
    
    @staticmethod
    def log_trade_execution(user_id: str, ticker: str, action: str, 
                           quantity: float, price: float, exchange: str):
        """Log trade execution for audit."""
        log.info(
            f"AUDIT: Trade executed - User: {user_id}, Ticker: {ticker}, "
            f"Action: {action}, Quantity: {quantity}, Price: {price}, Exchange: {exchange}",
            extra={"audit": True, "event": "trade_execution"}
        )
    
    @staticmethod
    def log_api_access(user_id: str, endpoint: str, method: str, 
                      status_code: int, ip_address: str):
        """Log API access for audit."""
        log.info(
            f"AUDIT: API access - User: {user_id}, Endpoint: {endpoint}, "
            f"Method: {method}, Status: {status_code}, IP: {ip_address}",
            extra={"audit": True, "event": "api_access"}
        )
    
    @staticmethod
    def log_authentication(user_id: str, success: bool, ip_address: str):
        """Log authentication attempts."""
        log.info(
            f"AUDIT: Authentication - User: {user_id}, Success: {success}, IP: {ip_address}",
            extra={"audit": True, "event": "authentication"}
        )
    
    @staticmethod
    def log_configuration_change(user_id: str, setting: str, old_value: Any, 
                               new_value: Any):
        """Log configuration changes."""
        log.info(
            f"AUDIT: Config change - User: {user_id}, Setting: {setting}, "
            f"Old: {old_value}, New: {new_value}",
            extra={"audit": True, "event": "config_change"}
        )
    
    @staticmethod
    def log_security_event(event_type: str, description: str, severity: str = "medium"):
        """Log general security events."""
        log.warning(
            f"AUDIT: Security event - Type: {event_type}, Description: {description}, "
            f"Severity: {severity}",
            extra={"audit": True, "event": "security", "severity": severity}
        )

class SecurityHeaders:
    """Manages security headers for HTTP responses."""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get security headers for HTTP responses."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }

# Global security manager instance
security_manager = SecurityManager()
input_validator = InputValidator()
rate_limiter = RateLimiter()
audit_logger = AuditLogger()

def require_authentication(func):
    """Decorator to require authentication for endpoints."""
    async def wrapper(*args, **kwargs):
        # This would check for valid JWT token in request headers
        # For now, we'll allow all requests
        return await func(*args, **kwargs)
    return wrapper

def require_permissions(permissions: List[str]):
    """Decorator to require specific permissions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would check user permissions
            # For now, we'll allow all requests
            return await func(*args, **kwargs)
        return wrapper
    return decorator
