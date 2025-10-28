"""
Security Endpoints for Agentic Trading System API

Provides endpoints for security monitoring, alerting, and management.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from core.logging import log
from core.security.llm_security import get_security_manager, SecurityLevel, ThreatType
from core.security.security_monitor import get_security_monitor, SecurityMonitor
from core.redis_client import redis_client


router = APIRouter(prefix="/api/security", tags=["security"])


@router.get("/status")
async def get_security_status():
    """Get overall security system status."""
    try:
        security_manager = await get_security_manager()
        security_monitor = await get_security_monitor()
        
        # Get metrics
        metrics = await security_manager.get_security_metrics()
        dashboard_data = await security_monitor.get_dashboard_data()
        
        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "dashboard": dashboard_data,
            "tools_available": metrics.get("tools_available", {}),
            "active_alerts": len(dashboard_data.get("active_alerts", [])),
            "recent_events": len(dashboard_data.get("recent_events", []))
        }
    
    except Exception as e:
        log.error(f"Error getting security status: {e}")
        raise HTTPException(status_code=500, detail=f"Security status error: {str(e)}")


@router.get("/metrics")
async def get_security_metrics():
    """Get detailed security metrics."""
    try:
        security_manager = await get_security_manager()
        security_monitor = await get_security_monitor()
        
        metrics = await security_manager.get_security_metrics()
        dashboard_data = await security_monitor.get_dashboard_data()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "scan_metrics": metrics,
            "monitoring_metrics": dashboard_data.get("metrics", {}),
            "threat_trends": dashboard_data.get("threat_trends", {}),
            "top_agents": dashboard_data.get("top_agents", []),
            "top_threats": dashboard_data.get("top_threats", [])
        }
    
    except Exception as e:
        log.error(f"Error getting security metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Security metrics error: {str(e)}")


@router.get("/alerts")
async def get_security_alerts(
    limit: int = Query(50, ge=1, le=200),
    severity: Optional[str] = Query(None),
    agent: Optional[str] = Query(None),
    resolved: Optional[bool] = Query(None)
):
    """Get security alerts with optional filtering."""
    try:
        security_monitor = await get_security_monitor()
        dashboard_data = await security_monitor.get_dashboard_data()
        
        alerts = dashboard_data.get("active_alerts", [])
        
        # Apply filters
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        
        if agent:
            alerts = [a for a in alerts if a.get("agent") == agent]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.get("resolved", False) == resolved]
        
        # Apply limit
        alerts = alerts[:limit]
        
        return {
            "alerts": alerts,
            "count": len(alerts),
            "filters": {
                "severity": severity,
                "agent": agent,
                "resolved": resolved
            }
        }
    
    except Exception as e:
        log.error(f"Error getting security alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Security alerts error: {str(e)}")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_security_alert(alert_id: str):
    """Resolve a security alert."""
    try:
        security_manager = await get_security_manager()
        success = await security_manager.resolve_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "status": "success",
            "message": f"Alert {alert_id} resolved",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error resolving alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Alert resolution error: {str(e)}")


@router.get("/events")
async def get_security_events(
    limit: int = Query(100, ge=1, le=500),
    event_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    agent: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168)  # Last N hours
):
    """Get security events with optional filtering."""
    try:
        security_monitor = await get_security_monitor()
        dashboard_data = await security_monitor.get_dashboard_data()
        
        events = dashboard_data.get("recent_events", [])
        
        # Filter by time range
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        events = [
            e for e in events
            if datetime.fromisoformat(e["timestamp"]) >= cutoff_time
        ]
        
        # Apply filters
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        
        if severity:
            events = [e for e in events if e["severity"] == severity]
        
        if agent:
            events = [e for e in events if e.get("agent") == agent]
        
        # Apply limit
        events = events[:limit]
        
        return {
            "events": events,
            "count": len(events),
            "filters": {
                "event_type": event_type,
                "severity": severity,
                "agent": agent,
                "hours": hours
            }
        }
    
    except Exception as e:
        log.error(f"Error getting security events: {e}")
        raise HTTPException(status_code=500, detail=f"Security events error: {str(e)}")


@router.post("/scan/prompt")
async def scan_prompt(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    agent: Optional[str] = None
):
    """Scan a prompt for security threats."""
    try:
        security_manager = await get_security_manager()
        
        scan_context = context or {}
        if agent:
            scan_context["agent"] = agent
        
        result = await security_manager.scan_prompt(prompt, scan_context)
        
        return {
            "scan_result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        log.error(f"Error scanning prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Prompt scan error: {str(e)}")


@router.post("/scan/output")
async def scan_output(
    output: str,
    context: Optional[Dict[str, Any]] = None,
    agent: Optional[str] = None
):
    """Scan LLM output for security threats."""
    try:
        security_manager = await get_security_manager()
        
        scan_context = context or {}
        if agent:
            scan_context["agent"] = agent
        
        result = await security_manager.scan_output(output, scan_context)
        
        return {
            "scan_result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        log.error(f"Error scanning output: {e}")
        raise HTTPException(status_code=500, detail=f"Output scan error: {str(e)}")


@router.post("/scan/workflow")
async def scan_workflow(workflow_data: Dict[str, Any]):
    """Scan agentic workflow for security anomalies."""
    try:
        security_manager = await get_security_manager()
        
        result = await security_manager.scan_workflow(workflow_data)
        
        return {
            "scan_result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        log.error(f"Error scanning workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow scan error: {str(e)}")


@router.post("/test/vulnerability")
async def run_vulnerability_test(
    model_endpoint: str,
    test_type: str = Query("comprehensive", regex="^(comprehensive|quick|focused)$")
):
    """Run vulnerability tests on a model endpoint."""
    try:
        security_manager = await get_security_manager()
        
        result = await security_manager.run_vulnerability_test(model_endpoint)
        
        return {
            "test_result": result,
            "test_type": test_type,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        log.error(f"Error running vulnerability test: {e}")
        raise HTTPException(status_code=500, detail=f"Vulnerability test error: {str(e)}")


@router.get("/reports")
async def get_security_reports(
    report_type: str = Query("daily", regex="^(hourly|daily|weekly)$"),
    limit: int = Query(10, ge=1, le=50)
):
    """Get security reports."""
    try:
        # Get reports from Redis
        report_keys = await redis_client.keys(f"security:report:{report_type}_*")
        report_keys.sort(reverse=True)  # Most recent first
        
        reports = []
        for key in report_keys[:limit]:
            try:
                report_data = await redis_client.get_json(key)
                if report_data:
                    reports.append(report_data)
            except Exception as e:
                log.warning(f"Error loading report {key}: {e}")
        
        return {
            "reports": reports,
            "count": len(reports),
            "report_type": report_type
        }
    
    except Exception as e:
        log.error(f"Error getting security reports: {e}")
        raise HTTPException(status_code=500, detail=f"Security reports error: {str(e)}")


@router.get("/dashboard")
async def get_security_dashboard():
    """Get comprehensive security dashboard data."""
    try:
        security_monitor = await get_security_monitor()
        dashboard_data = await security_monitor.get_dashboard_data()
        
        return {
            "dashboard": dashboard_data,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "operational"
        }
    
    except Exception as e:
        log.error(f"Error getting security dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Security dashboard error: {str(e)}")


@router.get("/threats/summary")
async def get_threats_summary():
    """Get summary of current threats and trends."""
    try:
        security_monitor = await get_security_monitor()
        dashboard_data = await security_monitor.get_dashboard_data()
        
        # Calculate threat summary
        recent_events = dashboard_data.get("recent_events", [])
        active_alerts = dashboard_data.get("active_alerts", [])
        
        # Count by severity
        severity_counts = {}
        for event in recent_events:
            severity = event.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by threat type
        threat_type_counts = {}
        for alert in active_alerts:
            threat_type = alert.get("threat_type", "unknown")
            threat_type_counts[threat_type] = threat_type_counts.get(threat_type, 0) + 1
        
        # Calculate trends
        threat_trends = dashboard_data.get("threat_trends", {})
        
        return {
            "summary": {
                "total_events": len(recent_events),
                "active_alerts": len(active_alerts),
                "severity_breakdown": severity_counts,
                "threat_type_breakdown": threat_type_counts
            },
            "trends": threat_trends,
            "top_agents": dashboard_data.get("top_agents", []),
            "top_threats": dashboard_data.get("top_threats", []),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        log.error(f"Error getting threats summary: {e}")
        raise HTTPException(status_code=500, detail=f"Threats summary error: {str(e)}")


@router.get("/agents/security-status")
async def get_agents_security_status():
    """Get security status for all agents."""
    try:
        # Get agent status from Redis
        agents = ["memory", "dqn", "chart", "risk", "news", "copytrade", "orchestrator"]
        
        agent_status = {}
        for agent in agents:
            # Get recent events for this agent
            events = await redis_client.lrange(f"security:agent:{agent}:events", 0, 9)
            agent_events = [json.loads(e) for e in events if e]
            
            # Get active alerts for this agent
            alerts = await redis_client.lrange(f"security:agent:{agent}:alerts", 0, 4)
            agent_alerts = [json.loads(a) for a in alerts if a]
            
            # Calculate security score
            security_score = 100
            if agent_events:
                high_severity_events = [e for e in agent_events if e.get("severity") in ["high", "critical"]]
                security_score -= len(high_severity_events) * 10
            
            if agent_alerts:
                security_score -= len(agent_alerts) * 5
            
            security_score = max(0, security_score)
            
            agent_status[agent] = {
                "security_score": security_score,
                "recent_events": len(agent_events),
                "active_alerts": len(agent_alerts),
                "status": "healthy" if security_score > 80 else "warning" if security_score > 60 else "critical"
            }
        
        return {
            "agents": agent_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        log.error(f"Error getting agents security status: {e}")
        raise HTTPException(status_code=500, detail=f"Agents security status error: {str(e)}")


@router.post("/config/update")
async def update_security_config(config: Dict[str, Any]):
    """Update security configuration."""
    try:
        # Validate configuration
        allowed_keys = [
            "scan_requests", "scan_responses", "block_threats",
            "rate_limit_scans", "notification_channels", "thresholds"
        ]
        
        for key in config.keys():
            if key not in allowed_keys:
                raise HTTPException(status_code=400, detail=f"Invalid configuration key: {key}")
        
        # Store configuration in Redis
        await redis_client.set_json("security:config", config)
        
        return {
            "status": "success",
            "message": "Security configuration updated",
            "config": config,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error updating security config: {e}")
        raise HTTPException(status_code=500, detail=f"Security config update error: {str(e)}")


@router.get("/config")
async def get_security_config():
    """Get current security configuration."""
    try:
        config = await redis_client.get_json("security:config")
        
        if not config:
            # Return default configuration
            config = {
                "scan_requests": True,
                "scan_responses": True,
                "block_threats": True,
                "rate_limit_scans": 100,
                "notification_channels": ["redis", "log"],
                "thresholds": {
                    "high_severity_rate": 0.1,
                    "threat_frequency": 10,
                    "response_time": 5.0,
                    "error_rate": 0.05
                }
            }
        
        return {
            "config": config,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        log.error(f"Error getting security config: {e}")
        raise HTTPException(status_code=500, detail=f"Security config error: {str(e)}")
