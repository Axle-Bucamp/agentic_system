"""
Security Middleware for FastAPI Application

Integrates LLM security scanning into the API request/response pipeline.
"""

import time
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from core.logging import log
from core.security.llm_security import get_security_manager, SecurityLevel


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for scanning API requests and responses for security threats."""
    
    def __init__(self, app, security_config: Dict[str, Any] = None):
        super().__init__(app)
        self.security_config = security_config or {}
        self.scan_requests = self.security_config.get("scan_requests", True)
        self.scan_responses = self.security_config.get("scan_responses", True)
        self.block_threats = self.security_config.get("block_threats", True)
        self.rate_limit_scans = self.security_config.get("rate_limit_scans", 100)  # per minute
        self.scan_cache = {}  # Simple in-memory cache
        self.scan_counts = {}  # Rate limiting
        
    async def dispatch(self, request: Request, call_next):
        """Process request through security middleware."""
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        try:
            # Rate limiting check
            if not await self._check_rate_limit(request):
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded for security scans"}
                )
            
            # Scan request if enabled
            if self.scan_requests:
                request_scan_result = await self._scan_request(request)
                if request_scan_result and self._should_block(request_scan_result):
                    return self._create_security_response(request_scan_result)
            
            # Process request
            response = await call_next(request)
            
            # Scan response if enabled
            if self.scan_responses and response.status_code < 400:
                response_scan_result = await self._scan_response(request, response)
                if response_scan_result and self._should_block(response_scan_result):
                    return self._create_security_response(response_scan_result)
            
            # Add security headers
            response.headers["X-Security-Scanned"] = "true"
            response.headers["X-Request-ID"] = request_id
            
            # Log security metrics
            await self._log_security_metrics(request, response, time.time() - start_time)
            
            return response
            
        except Exception as e:
            log.error(f"Security middleware error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal security error"}
            )
    
    async def _check_rate_limit(self, request: Request) -> bool:
        """Check if request is within rate limits."""
        client_ip = request.client.host
        current_time = time.time()
        minute_key = int(current_time // 60)
        
        # Clean old entries
        if client_ip in self.scan_counts:
            self.scan_counts[client_ip] = {
                k: v for k, v in self.scan_counts[client_ip].items()
                if k >= minute_key - 1
            }
        else:
            self.scan_counts[client_ip] = {}
        
        # Check current minute
        current_minute_count = self.scan_counts[client_ip].get(minute_key, 0)
        if current_minute_count >= self.rate_limit_scans:
            return False
        
        # Increment counter
        self.scan_counts[client_ip][minute_key] = current_minute_count + 1
        return True
    
    async def _scan_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """Scan incoming request for security threats."""
        try:
            # Get request body
            body = await self._get_request_body(request)
            if not body:
                return None
            
            # Check cache first
            cache_key = f"req_{hash(body)}"
            if cache_key in self.scan_cache:
                cached_result = self.scan_cache[cache_key]
                if time.time() - cached_result["timestamp"] < 300:  # 5 minute cache
                    return cached_result["result"]
            
            # Scan with security manager
            security_manager = await get_security_manager()
            scan_result = await security_manager.scan_prompt(
                body,
                {
                    "type": "api_request",
                    "method": request.method,
                    "url": str(request.url),
                    "headers": dict(request.headers),
                    "client_ip": request.client.host
                }
            )
            
            # Cache result
            self.scan_cache[cache_key] = {
                "result": scan_result,
                "timestamp": time.time()
            }
            
            return scan_result
            
        except Exception as e:
            log.error(f"Error scanning request: {e}")
            return None
    
    async def _scan_response(self, request: Request, response: Response) -> Optional[Dict[str, Any]]:
        """Scan outgoing response for security threats."""
        try:
            # Get response body
            body = await self._get_response_body(response)
            if not body:
                return None
            
            # Check cache first
            cache_key = f"resp_{hash(body)}"
            if cache_key in self.scan_cache:
                cached_result = self.scan_cache[cache_key]
                if time.time() - cached_result["timestamp"] < 300:  # 5 minute cache
                    return cached_result["result"]
            
            # Scan with security manager
            security_manager = await get_security_manager()
            scan_result = await security_manager.scan_output(
                body,
                {
                    "type": "api_response",
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "request_url": str(request.url)
                }
            )
            
            # Cache result
            self.scan_cache[cache_key] = {
                "result": scan_result,
                "timestamp": time.time()
            }
            
            return scan_result
            
        except Exception as e:
            log.error(f"Error scanning response: {e}")
            return None
    
    async def _get_request_body(self, request: Request) -> Optional[str]:
        """Extract request body for scanning."""
        try:
            # Only scan JSON requests
            if request.headers.get("content-type", "").startswith("application/json"):
                body = await request.body()
                return body.decode("utf-8")
            return None
        except Exception:
            return None
    
    async def _get_response_body(self, response: Response) -> Optional[str]:
        """Extract response body for scanning."""
        try:
            if hasattr(response, "body"):
                body = response.body
                if isinstance(body, bytes):
                    return body.decode("utf-8")
                return str(body)
            return None
        except Exception:
            return None
    
    def _should_block(self, scan_result: Dict[str, Any]) -> bool:
        """Determine if request/response should be blocked based on scan result."""
        if not self.block_threats:
            return False
        
        overall_risk = scan_result.get("overall_risk", {})
        risk_level = overall_risk.get("level", "safe")
        risk_score = overall_risk.get("score", 0.0)
        
        # Block based on risk level and score
        if risk_level in ["critical", "high"]:
            return True
        
        if risk_level == "medium" and risk_score > 0.7:
            return True
        
        return False
    
    def _create_security_response(self, scan_result: Dict[str, Any]) -> JSONResponse:
        """Create security response for blocked requests."""
        overall_risk = scan_result.get("overall_risk", {})
        risk_level = overall_risk.get("level", "unknown")
        
        return JSONResponse(
            status_code=403,
            content={
                "error": "Security threat detected",
                "risk_level": risk_level,
                "risk_score": overall_risk.get("score", 0.0),
                "threats": scan_result.get("llm_guard", {}).get("threats", []),
                "message": "Request blocked due to security policy violation",
                "timestamp": datetime.utcnow().isoformat()
            },
            headers={
                "X-Security-Blocked": "true",
                "X-Risk-Level": risk_level
            }
        )
    
    async def _log_security_metrics(self, request: Request, response: Response, duration: float):
        """Log security metrics for monitoring."""
        try:
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "duration_ms": duration * 1000,
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent", ""),
                "security_scanned": response.headers.get("X-Security-Scanned", "false")
            }
            
            # Store in Redis for monitoring
            from core.redis_client import redis_client
            await redis_client.lpush("security:metrics", json.dumps(metrics))
            
        except Exception as e:
            log.error(f"Error logging security metrics: {e}")


class AgentSecurityMiddleware(BaseHTTPMiddleware):
    """Specialized middleware for agent communication security."""
    
    def __init__(self, app, agent_name: str):
        super().__init__(app)
        self.agent_name = agent_name
        self.security_manager = None
    
    async def dispatch(self, request: Request, call_next):
        """Process agent requests through security middleware."""
        try:
            # Initialize security manager if needed
            if not self.security_manager:
                self.security_manager = await get_security_manager()
            
            # Scan agent-specific requests
            if request.url.path.startswith("/agent/"):
                scan_result = await self._scan_agent_request(request)
                if scan_result and scan_result.get("overall_risk", {}).get("level") in ["critical", "high"]:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "Agent security threat detected",
                            "agent": self.agent_name,
                            "threats": scan_result.get("llm_guard", {}).get("threats", [])
                        }
                    )
            
            response = await call_next(request)
            return response
            
        except Exception as e:
            log.error(f"Agent security middleware error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Agent security error"}
            )
    
    async def _scan_agent_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """Scan agent-specific requests."""
        try:
            body = await request.body()
            if not body:
                return None
            
            body_str = body.decode("utf-8")
            
            return await self.security_manager.scan_prompt(
                body_str,
                {
                    "type": "agent_request",
                    "agent": self.agent_name,
                    "method": request.method,
                    "url": str(request.url)
                }
            )
        except Exception as e:
            log.error(f"Error scanning agent request: {e}")
            return None


def create_security_middleware(app, config: Dict[str, Any] = None):
    """Create and configure security middleware."""
    security_config = {
        "scan_requests": True,
        "scan_responses": True,
        "block_threats": True,
        "rate_limit_scans": 100,
        **(config or {})
    }
    
    return SecurityMiddleware(app, security_config)


def create_agent_security_middleware(app, agent_name: str):
    """Create agent-specific security middleware."""
    return AgentSecurityMiddleware(app, agent_name)
