"""
LLM Security Module for Agentic Trading System

Integrates multiple security tools for comprehensive LLM and agentic pipeline protection:
- LLM Guard: Input/output sanitization and detection
- Agentic Radar: Agentic workflow security analysis
- Garak: LLM vulnerability testing
- Rebuff: Prompt injection protection
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

try:
    from llm_guard import scan_prompt, scan_output
    from llm_guard.scanners import (
        BanSubstrings, BanTopics, PromptInjection, Toxicity,
        Bias, MaliciousURLs, Sensitive, JSON, Language
    )
    LLM_GUARD_AVAILABLE = True
except ImportError:
    LLM_GUARD_AVAILABLE = False

try:
    import agentic_radar
    AGENTIC_RADAR_AVAILABLE = True
except ImportError:
    AGENTIC_RADAR_AVAILABLE = False

try:
    import garak
    GARAK_AVAILABLE = True
except ImportError:
    GARAK_AVAILABLE = False

try:
    from rebuff import Rebuff
    REBUFF_AVAILABLE = True
except ImportError:
    REBUFF_AVAILABLE = False

from core.logging import log
from core.redis_client import redis_client


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    PROMPT_INJECTION = "prompt_injection"
    DATA_LEAKAGE = "data_leakage"
    TOXICITY = "toxicity"
    BIAS = "bias"
    MALICIOUS_URL = "malicious_url"
    SENSITIVE_DATA = "sensitive_data"
    CODE_INJECTION = "code_injection"
    WORKFLOW_ANOMALY = "workflow_anomaly"


@dataclass
class SecurityAlert:
    """Security alert data structure."""
    alert_id: str
    threat_type: ThreatType
    severity: SecurityLevel
    message: str
    context: Dict[str, Any]
    timestamp: datetime
    agent: Optional[str] = None
    user_id: Optional[str] = None
    resolved: bool = False


@dataclass
class SecurityMetrics:
    """Security metrics for monitoring."""
    total_scans: int = 0
    threats_detected: int = 0
    false_positives: int = 0
    scan_latency_ms: float = 0.0
    last_scan: Optional[datetime] = None


class LLMSecurityManager:
    """Comprehensive LLM security manager for agentic trading system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alerts: List[SecurityAlert] = []
        self.metrics = SecurityMetrics()
        self.redis_client = redis_client
        
        # Initialize security tools
        self._initialize_llm_guard()
        self._initialize_agentic_radar()
        self._initialize_garak()
        self._initialize_rebuff()
        
        # Security policies
        self.policies = self._load_security_policies()
        
        log.info("LLM Security Manager initialized")
    
    def _initialize_llm_guard(self):
        """Initialize LLM Guard scanners."""
        if not LLM_GUARD_AVAILABLE:
            log.warning("LLM Guard not available - install with: pip install llm-guard")
            return
        
        try:
            # Configure prompt scanners
            self.prompt_scanners = [
                BanSubstrings(substrings=["password", "secret", "key", "token"]),
                BanTopics(topics=["violence", "hate", "self-harm"]),
                PromptInjection(),
                Toxicity(),
                Language(valid_languages=["en"]),
            ]
            
            # Configure output scanners
            self.output_scanners = [
                Bias(),
                MaliciousURLs(),
                Sensitive(),
                JSON(),
                Toxicity(),
            ]
            
            log.info("LLM Guard initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize LLM Guard: {e}")
    
    def _initialize_agentic_radar(self):
        """Initialize Agentic Radar for workflow analysis."""
        if not AGENTIC_RADAR_AVAILABLE:
            log.warning("Agentic Radar not available - install with: pip install agentic-radar")
            return
        
        try:
            # Configure agentic radar for trading workflow analysis
            self.agentic_radar = agentic_radar.AgenticRadar(
                config={
                    "workflow_path": "agents/",
                    "security_level": "high",
                    "scan_patterns": [
                        "prompt_injection",
                        "data_leakage",
                        "unauthorized_access",
                        "workflow_anomaly"
                    ]
                }
            )
            log.info("Agentic Radar initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize Agentic Radar: {e}")
    
    def _initialize_garak(self):
        """Initialize Garak for LLM vulnerability testing."""
        if not GARAK_AVAILABLE:
            log.warning("Garak not available - install with: pip install garak")
            return
        
        try:
            # Configure Garak for trading-specific vulnerability testing
            self.garak_config = {
                "model_name": "trading_agent",
                "probes": [
                    "promptinjection",
                    "promptleak",
                    "toxicity",
                    "bias"
                ],
                "output_dir": "security/garak_reports"
            }
            log.info("Garak initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize Garak: {e}")
    
    def _initialize_rebuff(self):
        """Initialize Rebuff for prompt injection protection."""
        if not REBUFF_AVAILABLE:
            log.warning("Rebuff not available - install with: pip install rebuff")
            return
        
        try:
            self.rebuff = Rebuff(
                api_key=self.config.get("rebuff_api_key"),
                api_url=self.config.get("rebuff_api_url", "https://api.rebuff.ai")
            )
            log.info("Rebuff initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize Rebuff: {e}")
    
    def _load_security_policies(self) -> Dict[str, Any]:
        """Load security policies from configuration."""
        return {
            "max_prompt_length": 10000,
            "allowed_file_types": [".txt", ".json", ".csv"],
            "blocked_patterns": [
                r"password\s*[:=]\s*\w+",
                r"api[_-]?key\s*[:=]\s*\w+",
                r"secret\s*[:=]\s*\w+",
                r"token\s*[:=]\s*\w+"
            ],
            "rate_limits": {
                "scans_per_minute": 100,
                "alerts_per_hour": 50
            },
            "auto_block_threshold": 0.8,
            "notification_channels": ["redis", "log"]
        }
    
    async def scan_prompt(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Scan prompt for security threats using multiple tools."""
        start_time = datetime.utcnow()
        context = context or {}
        
        try:
            # LLM Guard scan
            llm_guard_result = await self._scan_with_llm_guard(prompt, "prompt")
            
            # Rebuff scan
            rebuff_result = await self._scan_with_rebuff(prompt)
            
            # Custom pattern matching
            pattern_result = await self._scan_patterns(prompt)
            
            # Combine results
            scan_result = {
                "prompt": prompt,
                "context": context,
                "timestamp": start_time.isoformat(),
                "llm_guard": llm_guard_result,
                "rebuff": rebuff_result,
                "patterns": pattern_result,
                "overall_risk": self._calculate_overall_risk([
                    llm_guard_result, rebuff_result, pattern_result
                ]),
                "scan_duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
            
            # Update metrics
            self.metrics.total_scans += 1
            self.metrics.last_scan = datetime.utcnow()
            self.metrics.scan_latency_ms = scan_result["scan_duration_ms"]
            
            if scan_result["overall_risk"]["level"] != "safe":
                self.metrics.threats_detected += 1
                await self._create_alert(scan_result, context)
            
            # Store scan result
            await self._store_scan_result(scan_result)
            
            return scan_result
            
        except Exception as e:
            log.error(f"Error scanning prompt: {e}")
            return {
                "prompt": prompt,
                "error": str(e),
                "timestamp": start_time.isoformat(),
                "overall_risk": {"level": "error", "score": 1.0}
            }
    
    async def scan_output(self, output: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Scan LLM output for security threats."""
        start_time = datetime.utcnow()
        context = context or {}
        
        try:
            # LLM Guard scan
            llm_guard_result = await self._scan_with_llm_guard(output, "output")
            
            # Pattern matching for sensitive data
            pattern_result = await self._scan_patterns(output)
            
            # Combine results
            scan_result = {
                "output": output,
                "context": context,
                "timestamp": start_time.isoformat(),
                "llm_guard": llm_guard_result,
                "patterns": pattern_result,
                "overall_risk": self._calculate_overall_risk([
                    llm_guard_result, pattern_result
                ]),
                "scan_duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
            
            # Update metrics
            self.metrics.total_scans += 1
            self.metrics.last_scan = datetime.utcnow()
            self.metrics.scan_latency_ms = scan_result["scan_duration_ms"]
            
            if scan_result["overall_risk"]["level"] != "safe":
                self.metrics.threats_detected += 1
                await self._create_alert(scan_result, context)
            
            # Store scan result
            await self._store_scan_result(scan_result)
            
            return scan_result
            
        except Exception as e:
            log.error(f"Error scanning output: {e}")
            return {
                "output": output,
                "error": str(e),
                "timestamp": start_time.isoformat(),
                "overall_risk": {"level": "error", "score": 1.0}
            }
    
    async def scan_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scan agentic workflow for security anomalies using Agentic Radar."""
        if not AGENTIC_RADAR_AVAILABLE:
            return {"error": "Agentic Radar not available"}
        
        try:
            # Use Agentic Radar to analyze workflow
            radar_result = await self.agentic_radar.scan_workflow(workflow_data)
            
            # Analyze workflow patterns
            workflow_analysis = await self._analyze_workflow_patterns(workflow_data)
            
            scan_result = {
                "workflow": workflow_data,
                "timestamp": datetime.utcnow().isoformat(),
                "radar_result": radar_result,
                "pattern_analysis": workflow_analysis,
                "overall_risk": self._calculate_workflow_risk(radar_result, workflow_analysis)
            }
            
            if scan_result["overall_risk"]["level"] != "safe":
                await self._create_alert(scan_result, {"type": "workflow"})
            
            return scan_result
            
        except Exception as e:
            log.error(f"Error scanning workflow: {e}")
            return {"error": str(e)}
    
    async def run_vulnerability_test(self, model_endpoint: str) -> Dict[str, Any]:
        """Run comprehensive vulnerability tests using Garak."""
        if not GARAK_AVAILABLE:
            return {"error": "Garak not available"}
        
        try:
            # Configure Garak for trading model testing
            garak_config = {
                **self.garak_config,
                "model_name": model_endpoint,
                "probes": [
                    "promptinjection",
                    "promptleak", 
                    "toxicity",
                    "bias",
                    "trading_specific"  # Custom probe for trading vulnerabilities
                ]
            }
            
            # Run Garak tests
            test_result = await self._run_garak_tests(garak_config)
            
            return {
                "model_endpoint": model_endpoint,
                "timestamp": datetime.utcnow().isoformat(),
                "garak_result": test_result,
                "vulnerabilities": self._parse_garak_results(test_result)
            }
            
        except Exception as e:
            log.error(f"Error running vulnerability test: {e}")
            return {"error": str(e)}
    
    async def _scan_with_llm_guard(self, text: str, scan_type: str) -> Dict[str, Any]:
        """Scan text using LLM Guard."""
        if not LLM_GUARD_AVAILABLE:
            return {"error": "LLM Guard not available"}
        
        try:
            if scan_type == "prompt":
                result = scan_prompt(self.prompt_scanners, text)
            else:
                result = scan_output(self.output_scanners, text)
            
            return {
                "sanitized": result.sanitized,
                "risk_score": result.risk_score,
                "threats": [threat.name for threat in result.threats],
                "is_safe": result.is_safe
            }
        except Exception as e:
            log.error(f"LLM Guard scan error: {e}")
            return {"error": str(e), "is_safe": False, "risk_score": 1.0}
    
    async def _scan_with_rebuff(self, prompt: str) -> Dict[str, Any]:
        """Scan prompt using Rebuff."""
        if not REBUFF_AVAILABLE:
            return {"error": "Rebuff not available"}
        
        try:
            result = await self.rebuff.detect_injection(prompt)
            return {
                "is_injection": result.is_injection,
                "risk_score": result.risk_score,
                "suggestions": result.suggestions
            }
        except Exception as e:
            log.error(f"Rebuff scan error: {e}")
            return {"error": str(e), "is_injection": True, "risk_score": 1.0}
    
    async def _scan_patterns(self, text: str) -> Dict[str, Any]:
        """Scan text for custom security patterns."""
        import re
        
        threats = []
        risk_score = 0.0
        
        for pattern in self.policies["blocked_patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append(f"Pattern match: {pattern}")
                risk_score += 0.3
        
        # Check for excessive length
        if len(text) > self.policies["max_prompt_length"]:
            threats.append("Excessive length")
            risk_score += 0.2
        
        return {
            "threats": threats,
            "risk_score": min(risk_score, 1.0),
            "is_safe": risk_score < 0.5
        }
    
    async def _analyze_workflow_patterns(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow patterns for anomalies."""
        anomalies = []
        risk_score = 0.0
        
        # Check for suspicious agent interactions
        if "agents" in workflow_data:
            agent_count = len(workflow_data["agents"])
            if agent_count > 10:  # Too many agents
                anomalies.append("Excessive agent count")
                risk_score += 0.2
        
        # Check for rapid successive operations
        if "operations" in workflow_data:
            ops = workflow_data["operations"]
            if len(ops) > 100:  # Too many operations
                anomalies.append("Excessive operations")
                risk_score += 0.3
        
        # Check for suspicious data access patterns
        if "data_access" in workflow_data:
            data_access = workflow_data["data_access"]
            if len(data_access) > 50:  # Too much data access
                anomalies.append("Excessive data access")
                risk_score += 0.4
        
        return {
            "anomalies": anomalies,
            "risk_score": min(risk_score, 1.0),
            "is_safe": risk_score < 0.5
        }
    
    def _calculate_overall_risk(self, scan_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall risk score from multiple scan results."""
        total_risk = 0.0
        threat_count = 0
        errors = 0
        
        for result in scan_results:
            if "error" in result:
                errors += 1
                total_risk += 1.0
            elif "risk_score" in result:
                total_risk += result["risk_score"]
                if result["risk_score"] > 0.5:
                    threat_count += 1
        
        if errors > 0:
            return {"level": "error", "score": 1.0, "threats": threat_count}
        
        avg_risk = total_risk / len(scan_results) if scan_results else 0.0
        
        if avg_risk >= 0.8:
            level = "critical"
        elif avg_risk >= 0.6:
            level = "high"
        elif avg_risk >= 0.4:
            level = "medium"
        elif avg_risk >= 0.2:
            level = "low"
        else:
            level = "safe"
        
        return {
            "level": level,
            "score": avg_risk,
            "threats": threat_count
        }
    
    def _calculate_workflow_risk(self, radar_result: Dict[str, Any], pattern_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate workflow-specific risk score."""
        radar_risk = radar_result.get("risk_score", 0.0)
        pattern_risk = pattern_analysis.get("risk_score", 0.0)
        
        overall_risk = (radar_risk + pattern_risk) / 2
        
        if overall_risk >= 0.8:
            level = "critical"
        elif overall_risk >= 0.6:
            level = "high"
        elif overall_risk >= 0.4:
            level = "medium"
        elif overall_risk >= 0.2:
            level = "low"
        else:
            level = "safe"
        
        return {
            "level": level,
            "score": overall_risk,
            "radar_risk": radar_risk,
            "pattern_risk": pattern_risk
        }
    
    async def _create_alert(self, scan_result: Dict[str, Any], context: Dict[str, Any]):
        """Create security alert."""
        alert = SecurityAlert(
            alert_id=f"alert_{datetime.utcnow().timestamp()}",
            threat_type=ThreatType.PROMPT_INJECTION,  # Default, should be determined from scan
            severity=SecurityLevel.HIGH,  # Should be determined from risk score
            message=f"Security threat detected: {scan_result.get('overall_risk', {}).get('level', 'unknown')}",
            context=context,
            timestamp=datetime.utcnow(),
            agent=context.get("agent"),
            user_id=context.get("user_id")
        )
        
        self.alerts.append(alert)
        
        # Store in Redis
        await self.redis_client.set_json(
            f"security:alert:{alert.alert_id}",
            {
                "alert_id": alert.alert_id,
                "threat_type": alert.threat_type.value,
                "severity": alert.severity.value,
                "message": alert.message,
                "context": alert.context,
                "timestamp": alert.timestamp.isoformat(),
                "agent": alert.agent,
                "user_id": alert.user_id,
                "resolved": alert.resolved
            }
        )
        
        # Publish alert
        await self.redis_client.publish(
            "security:alerts",
            json.dumps(alert.__dict__, default=str)
        )
        
        log.warning(f"Security alert created: {alert.alert_id}")
    
    async def _store_scan_result(self, scan_result: Dict[str, Any]):
        """Store scan result in Redis."""
        scan_id = f"scan_{datetime.utcnow().timestamp()}"
        await self.redis_client.set_json(f"security:scan:{scan_id}", scan_result)
    
    async def _run_garak_tests(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run Garak vulnerability tests."""
        # This would integrate with Garak's testing framework
        # For now, return a mock result
        return {
            "tests_run": 10,
            "vulnerabilities_found": 0,
            "test_duration": 30.5,
            "status": "completed"
        }
    
    def _parse_garak_results(self, test_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Garak test results for vulnerabilities."""
        # This would parse actual Garak results
        return []
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get current security metrics."""
        return {
            "total_scans": self.metrics.total_scans,
            "threats_detected": self.metrics.threats_detected,
            "false_positives": self.metrics.false_positives,
            "scan_latency_ms": self.metrics.scan_latency_ms,
            "last_scan": self.metrics.last_scan.isoformat() if self.metrics.last_scan else None,
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "tools_available": {
                "llm_guard": LLM_GUARD_AVAILABLE,
                "agentic_radar": AGENTIC_RADAR_AVAILABLE,
                "garak": GARAK_AVAILABLE,
                "rebuff": REBUFF_AVAILABLE
            }
        }
    
    async def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent security alerts."""
        recent_alerts = sorted(
            self.alerts,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
        
        return [
            {
                "alert_id": alert.alert_id,
                "threat_type": alert.threat_type.value,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "agent": alert.agent,
                "resolved": alert.resolved
            }
            for alert in recent_alerts
        ]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a security alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                await self.redis_client.set_json(
                    f"security:alert:{alert_id}",
                    {
                        "alert_id": alert.alert_id,
                        "threat_type": alert.threat_type.value,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "context": alert.context,
                        "timestamp": alert.timestamp.isoformat(),
                        "agent": alert.agent,
                        "user_id": alert.user_id,
                        "resolved": True
                    }
                )
                return True
        return False


# Global security manager instance
_security_manager: Optional[LLMSecurityManager] = None


async def get_security_manager() -> LLMSecurityManager:
    """Get global security manager instance."""
    global _security_manager
    if _security_manager is None:
        from core.config import settings
        _security_manager = LLMSecurityManager({
            "rebuff_api_key": getattr(settings, "rebuff_api_key", None),
            "rebuff_api_url": getattr(settings, "rebuff_api_url", "https://api.rebuff.ai")
        })
    return _security_manager


async def scan_agent_prompt(prompt: str, agent_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function to scan agent prompts."""
    security_manager = await get_security_manager()
    context = context or {}
    context["agent"] = agent_name
    return await security_manager.scan_prompt(prompt, context)


async def scan_agent_output(output: str, agent_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function to scan agent outputs."""
    security_manager = await get_security_manager()
    context = context or {}
    context["agent"] = agent_name
    return await security_manager.scan_output(output, context)
