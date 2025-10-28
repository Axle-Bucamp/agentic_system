"""
Security Monitoring and Alerting System

Provides real-time security monitoring, alerting, and reporting for the agentic trading system.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from core.logging import log
from core.redis_client import redis_client
from core.security.llm_security import SecurityAlert, ThreatType, SecurityLevel


class AlertStatus(Enum):
    """Alert status enumeration."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    event_type: str
    severity: SecurityLevel
    message: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]
    agent: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class SecurityReport:
    """Security report data structure."""
    report_id: str
    period_start: datetime
    period_end: datetime
    total_events: int
    threats_by_type: Dict[str, int]
    threats_by_severity: Dict[str, int]
    top_agents: List[Dict[str, Any]]
    top_threats: List[Dict[str, Any]]
    recommendations: List[str]


class SecurityMonitor:
    """Real-time security monitoring and alerting system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.events: List[SecurityEvent] = []
        self.alerts: List[SecurityAlert] = []
        self.metrics = {
            "total_events": 0,
            "threats_detected": 0,
            "alerts_generated": 0,
            "false_positives": 0,
            "avg_response_time": 0.0
        }
        self.running = False
        self.monitor_task = None
        
        # Alert thresholds
        self.thresholds = {
            "high_severity_rate": 0.1,  # 10% of events
            "threat_frequency": 10,  # per minute
            "response_time": 5.0,  # seconds
            "error_rate": 0.05  # 5% of requests
        }
        
        # Notification channels
        self.notification_channels = self.config.get("notification_channels", ["redis", "log"])
        
        log.info("Security Monitor initialized")
    
    async def start_monitoring(self):
        """Start the security monitoring system."""
        if self.running:
            log.warning("Security monitoring already running")
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        log.info("Security monitoring started")
    
    async def stop_monitoring(self):
        """Stop the security monitoring system."""
        if not self.running:
            return
        
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        log.info("Security monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Process security events
                await self._process_security_events()
                
                # Check for new alerts
                await self._check_for_alerts()
                
                # Update metrics
                await self._update_metrics()
                
                # Generate reports if needed
                await self._generate_periodic_reports()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Wait before next iteration
                await asyncio.sleep(1.0)
                
            except Exception as e:
                log.error(f"Error in security monitoring loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_security_events(self):
        """Process security events from Redis."""
        try:
            # Get events from Redis
            events_data = await redis_client.lrange("security:events", 0, 99)
            
            for event_data in events_data:
                try:
                    event_dict = json.loads(event_data)
                    event = SecurityEvent(
                        event_id=event_dict["event_id"],
                        event_type=event_dict["event_type"],
                        severity=SecurityLevel(event_dict["severity"]),
                        message=event_dict["message"],
                        source=event_dict["source"],
                        timestamp=datetime.fromisoformat(event_dict["timestamp"]),
                        metadata=event_dict["metadata"],
                        agent=event_dict.get("agent"),
                        user_id=event_dict.get("user_id")
                    )
                    
                    # Process event
                    await self._process_event(event)
                    
                except Exception as e:
                    log.error(f"Error processing security event: {e}")
            
        except Exception as e:
            log.error(f"Error processing security events: {e}")
    
    async def _process_event(self, event: SecurityEvent):
        """Process a single security event."""
        # Add to events list
        self.events.append(event)
        self.metrics["total_events"] += 1
        
        # Check if event requires alert
        if self._should_create_alert(event):
            await self._create_alert_from_event(event)
        
        # Check for patterns
        await self._check_event_patterns(event)
        
        # Store event
        await self._store_event(event)
    
    def _should_create_alert(self, event: SecurityEvent) -> bool:
        """Determine if event should create an alert."""
        # Always alert on critical events
        if event.severity == SecurityLevel.CRITICAL:
            return True
        
        # Alert on high severity events
        if event.severity == SecurityLevel.HIGH:
            return True
        
        # Check for pattern-based alerts
        if self._check_threat_patterns(event):
            return True
        
        return False
    
    def _check_threat_patterns(self, event: SecurityEvent) -> bool:
        """Check for threat patterns that require alerts."""
        # Check for rapid successive events
        recent_events = [
            e for e in self.events[-10:]
            if e.timestamp > datetime.utcnow() - timedelta(minutes=1)
        ]
        
        if len(recent_events) > 5:  # More than 5 events in 1 minute
            return True
        
        # Check for same agent multiple threats
        agent_events = [
            e for e in self.events[-20:]
            if e.agent == event.agent and e.timestamp > datetime.utcnow() - timedelta(minutes=5)
        ]
        
        if len(agent_events) > 3:  # More than 3 events from same agent in 5 minutes
            return True
        
        return False
    
    async def _create_alert_from_event(self, event: SecurityEvent):
        """Create alert from security event."""
        alert = SecurityAlert(
            alert_id=f"alert_{int(time.time())}_{event.event_id}",
            threat_type=self._map_event_to_threat_type(event),
            severity=event.severity,
            message=f"Security event: {event.message}",
            context={
                "event_id": event.event_id,
                "source": event.source,
                "metadata": event.metadata
            },
            timestamp=event.timestamp,
            agent=event.agent,
            user_id=event.user_id
        )
        
        self.alerts.append(alert)
        self.metrics["alerts_generated"] += 1
        
        # Store alert
        await self._store_alert(alert)
        
        # Send notifications
        await self._send_notifications(alert)
        
        log.warning(f"Security alert created: {alert.alert_id}")
    
    def _map_event_to_threat_type(self, event: SecurityEvent) -> ThreatType:
        """Map security event to threat type."""
        event_type = event.event_type.lower()
        
        if "injection" in event_type:
            return ThreatType.PROMPT_INJECTION
        elif "leak" in event_type or "exposure" in event_type:
            return ThreatType.DATA_LEAKAGE
        elif "toxic" in event_type or "hate" in event_type:
            return ThreatType.TOXICITY
        elif "bias" in event_type:
            return ThreatType.BIAS
        elif "url" in event_type or "malicious" in event_type:
            return ThreatType.MALICIOUS_URL
        elif "sensitive" in event_type or "secret" in event_type:
            return ThreatType.SENSITIVE_DATA
        elif "code" in event_type:
            return ThreatType.CODE_INJECTION
        else:
            return ThreatType.WORKFLOW_ANOMALY
    
    async def _check_event_patterns(self, event: SecurityEvent):
        """Check for security event patterns."""
        # Check for attack patterns
        await self._check_attack_patterns(event)
        
        # Check for anomaly patterns
        await self._check_anomaly_patterns(event)
    
    async def _check_attack_patterns(self, event: SecurityEvent):
        """Check for attack patterns."""
        # Check for coordinated attacks
        recent_events = [
            e for e in self.events[-50:]
            if e.timestamp > datetime.utcnow() - timedelta(minutes=10)
        ]
        
        # Group by source IP or agent
        source_groups = {}
        for e in recent_events:
            source = e.source
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(e)
        
        # Check for coordinated attacks
        for source, events in source_groups.items():
            if len(events) > 10:  # More than 10 events from same source
                await self._create_coordinated_attack_alert(source, events)
    
    async def _check_anomaly_patterns(self, event: SecurityEvent):
        """Check for anomaly patterns."""
        # Check for unusual timing patterns
        if event.timestamp.hour < 6 or event.timestamp.hour > 22:  # Outside business hours
            await self._create_anomaly_alert("unusual_timing", event)
        
        # Check for unusual agent behavior
        if event.agent:
            agent_events = [e for e in self.events if e.agent == event.agent]
            if len(agent_events) > 20:  # Agent generating too many events
                await self._create_anomaly_alert("agent_anomaly", event)
    
    async def _create_coordinated_attack_alert(self, source: str, events: List[SecurityEvent]):
        """Create alert for coordinated attack."""
        alert = SecurityAlert(
            alert_id=f"coordinated_attack_{int(time.time())}",
            threat_type=ThreatType.WORKFLOW_ANOMALY,
            severity=SecurityLevel.CRITICAL,
            message=f"Coordinated attack detected from {source}",
            context={
                "source": source,
                "event_count": len(events),
                "time_window": "10 minutes"
            },
            timestamp=datetime.utcnow()
        )
        
        self.alerts.append(alert)
        await self._store_alert(alert)
        await self._send_notifications(alert)
    
    async def _create_anomaly_alert(self, anomaly_type: str, event: SecurityEvent):
        """Create alert for anomaly."""
        alert = SecurityAlert(
            alert_id=f"anomaly_{anomaly_type}_{int(time.time())}",
            threat_type=ThreatType.WORKFLOW_ANOMALY,
            severity=SecurityLevel.MEDIUM,
            message=f"Security anomaly detected: {anomaly_type}",
            context={
                "anomaly_type": anomaly_type,
                "event_id": event.event_id,
                "agent": event.agent
            },
            timestamp=datetime.utcnow()
        )
        
        self.alerts.append(alert)
        await self._store_alert(alert)
        await self._send_notifications(alert)
    
    async def _check_for_alerts(self):
        """Check for new alerts from Redis."""
        try:
            # Get alerts from Redis
            alerts_data = await redis_client.lrange("security:alerts", 0, 99)
            
            for alert_data in alerts_data:
                try:
                    alert_dict = json.loads(alert_data)
                    alert = SecurityAlert(
                        alert_id=alert_dict["alert_id"],
                        threat_type=ThreatType(alert_dict["threat_type"]),
                        severity=SecurityLevel(alert_dict["severity"]),
                        message=alert_dict["message"],
                        context=alert_dict["context"],
                        timestamp=datetime.fromisoformat(alert_dict["timestamp"]),
                        agent=alert_dict.get("agent"),
                        user_id=alert_dict.get("user_id"),
                        resolved=alert_dict.get("resolved", False)
                    )
                    
                    # Check if alert already exists
                    if not any(a.alert_id == alert.alert_id for a in self.alerts):
                        self.alerts.append(alert)
                        await self._send_notifications(alert)
                
                except Exception as e:
                    log.error(f"Error processing alert: {e}")
        
        except Exception as e:
            log.error(f"Error checking for alerts: {e}")
    
    async def _update_metrics(self):
        """Update security metrics."""
        try:
            # Calculate metrics
            total_events = len(self.events)
            threats_detected = len([e for e in self.events if e.severity != SecurityLevel.LOW])
            alerts_generated = len(self.alerts)
            
            # Calculate response time (mock for now)
            avg_response_time = 1.0
            
            self.metrics.update({
                "total_events": total_events,
                "threats_detected": threats_detected,
                "alerts_generated": alerts_generated,
                "avg_response_time": avg_response_time
            })
            
            # Store metrics in Redis
            await redis_client.set_json("security:metrics", self.metrics)
            
        except Exception as e:
            log.error(f"Error updating metrics: {e}")
    
    async def _generate_periodic_reports(self):
        """Generate periodic security reports."""
        try:
            # Generate hourly reports
            now = datetime.utcnow()
            if now.minute == 0:  # Top of the hour
                await self._generate_hourly_report(now)
            
            # Generate daily reports
            if now.hour == 0 and now.minute == 0:  # Midnight
                await self._generate_daily_report(now)
        
        except Exception as e:
            log.error(f"Error generating reports: {e}")
    
    async def _generate_hourly_report(self, timestamp: datetime):
        """Generate hourly security report."""
        try:
            hour_start = timestamp.replace(minute=0, second=0, microsecond=0)
            hour_end = hour_start + timedelta(hours=1)
            
            # Get events for this hour
            hour_events = [
                e for e in self.events
                if hour_start <= e.timestamp < hour_end
            ]
            
            if not hour_events:
                return
            
            # Generate report
            report = SecurityReport(
                report_id=f"hourly_{int(timestamp.timestamp())}",
                period_start=hour_start,
                period_end=hour_end,
                total_events=len(hour_events),
                threats_by_type=self._count_threats_by_type(hour_events),
                threats_by_severity=self._count_threats_by_severity(hour_events),
                top_agents=self._get_top_agents(hour_events),
                top_threats=self._get_top_threats(hour_events),
                recommendations=self._generate_recommendations(hour_events)
            )
            
            # Store report
            await self._store_report(report)
            
            log.info(f"Generated hourly security report: {report.report_id}")
        
        except Exception as e:
            log.error(f"Error generating hourly report: {e}")
    
    async def _generate_daily_report(self, timestamp: datetime):
        """Generate daily security report."""
        try:
            day_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            
            # Get events for this day
            day_events = [
                e for e in self.events
                if day_start <= e.timestamp < day_end
            ]
            
            if not day_events:
                return
            
            # Generate report
            report = SecurityReport(
                report_id=f"daily_{int(timestamp.timestamp())}",
                period_start=day_start,
                period_end=day_end,
                total_events=len(day_events),
                threats_by_type=self._count_threats_by_type(day_events),
                threats_by_severity=self._count_threats_by_severity(day_events),
                top_agents=self._get_top_agents(day_events),
                top_threats=self._get_top_threats(day_events),
                recommendations=self._generate_recommendations(day_events)
            )
            
            # Store report
            await self._store_report(report)
            
            log.info(f"Generated daily security report: {report.report_id}")
        
        except Exception as e:
            log.error(f"Error generating daily report: {e}")
    
    def _count_threats_by_type(self, events: List[SecurityEvent]) -> Dict[str, int]:
        """Count threats by type."""
        counts = {}
        for event in events:
            event_type = event.event_type
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts
    
    def _count_threats_by_severity(self, events: List[SecurityEvent]) -> Dict[str, int]:
        """Count threats by severity."""
        counts = {}
        for event in events:
            severity = event.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def _get_top_agents(self, events: List[SecurityEvent], limit: int = 5) -> List[Dict[str, Any]]:
        """Get top agents by event count."""
        agent_counts = {}
        for event in events:
            if event.agent:
                agent_counts[event.agent] = agent_counts.get(event.agent, 0) + 1
        
        return [
            {"agent": agent, "count": count}
            for agent, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]
    
    def _get_top_threats(self, events: List[SecurityEvent], limit: int = 5) -> List[Dict[str, Any]]:
        """Get top threats by count."""
        threat_counts = {}
        for event in events:
            threat_type = event.event_type
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        return [
            {"threat_type": threat_type, "count": count}
            for threat_type, count in sorted(threat_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]
    
    def _generate_recommendations(self, events: List[SecurityEvent]) -> List[str]:
        """Generate security recommendations based on events."""
        recommendations = []
        
        # Check for high threat rate
        high_severity_events = [e for e in events if e.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]]
        if len(high_severity_events) > len(events) * 0.1:  # More than 10% high severity
            recommendations.append("High threat rate detected - consider increasing security measures")
        
        # Check for agent-specific issues
        agent_events = {}
        for event in events:
            if event.agent:
                agent_events[event.agent] = agent_events.get(event.agent, 0) + 1
        
        for agent, count in agent_events.items():
            if count > 10:  # Agent generating too many events
                recommendations.append(f"Agent {agent} generating excessive events - investigate")
        
        # Check for timing patterns
        night_events = [e for e in events if e.timestamp.hour < 6 or e.timestamp.hour > 22]
        if len(night_events) > len(events) * 0.3:  # More than 30% at night
            recommendations.append("Unusual timing pattern detected - consider monitoring off-hours activity")
        
        return recommendations
    
    async def _cleanup_old_data(self):
        """Clean up old data to prevent memory issues."""
        try:
            # Keep only last 1000 events
            if len(self.events) > 1000:
                self.events = self.events[-1000:]
            
            # Keep only last 500 alerts
            if len(self.alerts) > 500:
                self.alerts = self.alerts[-500:]
            
            # Clean up old Redis data
            await redis_client.ltrim("security:events", 0, 999)
            await redis_client.ltrim("security:alerts", 0, 499)
        
        except Exception as e:
            log.error(f"Error cleaning up old data: {e}")
    
    async def _store_event(self, event: SecurityEvent):
        """Store security event in Redis."""
        try:
            event_data = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "severity": event.severity.value,
                "message": event.message,
                "source": event.source,
                "timestamp": event.timestamp.isoformat(),
                "metadata": event.metadata,
                "agent": event.agent,
                "user_id": event.user_id
            }
            
            await redis_client.lpush("security:events", json.dumps(event_data))
        
        except Exception as e:
            log.error(f"Error storing event: {e}")
    
    async def _store_alert(self, alert: SecurityAlert):
        """Store security alert in Redis."""
        try:
            alert_data = {
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
            
            await redis_client.lpush("security:alerts", json.dumps(alert_data))
        
        except Exception as e:
            log.error(f"Error storing alert: {e}")
    
    async def _store_report(self, report: SecurityReport):
        """Store security report in Redis."""
        try:
            report_data = asdict(report)
            report_data["period_start"] = report.period_start.isoformat()
            report_data["period_end"] = report.period_end.isoformat()
            
            await redis_client.set_json(f"security:report:{report.report_id}", report_data)
        
        except Exception as e:
            log.error(f"Error storing report: {e}")
    
    async def _send_notifications(self, alert: SecurityAlert):
        """Send security notifications."""
        try:
            for channel in self.notification_channels:
                if channel == "redis":
                    await self._send_redis_notification(alert)
                elif channel == "log":
                    await self._send_log_notification(alert)
                elif channel == "webhook":
                    await self._send_webhook_notification(alert)
        
        except Exception as e:
            log.error(f"Error sending notifications: {e}")
    
    async def _send_redis_notification(self, alert: SecurityAlert):
        """Send notification via Redis."""
        try:
            notification = {
                "type": "security_alert",
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "agent": alert.agent
            }
            
            await redis_client.publish("security:notifications", json.dumps(notification))
        
        except Exception as e:
            log.error(f"Error sending Redis notification: {e}")
    
    async def _send_log_notification(self, alert: SecurityAlert):
        """Send notification via log."""
        log.warning(f"SECURITY ALERT: {alert.alert_id} - {alert.message}")
    
    async def _send_webhook_notification(self, alert: SecurityAlert):
        """Send notification via webhook."""
        # Implementation would depend on webhook configuration
        pass
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for security dashboard."""
        try:
            # Get recent events
            recent_events = self.events[-50:] if self.events else []
            
            # Get active alerts
            active_alerts = [a for a in self.alerts if not a.resolved]
            
            # Get metrics
            metrics = self.metrics.copy()
            
            # Get threat trends
            threat_trends = self._calculate_threat_trends()
            
            return {
                "metrics": metrics,
                "recent_events": [
                    {
                        "event_id": e.event_id,
                        "event_type": e.event_type,
                        "severity": e.severity.value,
                        "message": e.message,
                        "timestamp": e.timestamp.isoformat(),
                        "agent": e.agent
                    }
                    for e in recent_events
                ],
                "active_alerts": [
                    {
                        "alert_id": a.alert_id,
                        "threat_type": a.threat_type.value,
                        "severity": a.severity.value,
                        "message": a.message,
                        "timestamp": a.timestamp.isoformat(),
                        "agent": a.agent
                    }
                    for a in active_alerts
                ],
                "threat_trends": threat_trends,
                "top_agents": self._get_top_agents(recent_events),
                "top_threats": self._get_top_threats(recent_events)
            }
        
        except Exception as e:
            log.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}
    
    def _calculate_threat_trends(self) -> Dict[str, Any]:
        """Calculate threat trends over time."""
        try:
            # Group events by hour for last 24 hours
            now = datetime.utcnow()
            hourly_counts = {}
            
            for i in range(24):
                hour_start = now - timedelta(hours=i+1)
                hour_end = now - timedelta(hours=i)
                
                hour_events = [
                    e for e in self.events
                    if hour_start <= e.timestamp < hour_end
                ]
                
                hourly_counts[hour_start.strftime("%H:00")] = len(hour_events)
            
            return {
                "hourly_counts": hourly_counts,
                "trend": "increasing" if len(self.events) > 0 else "stable"
            }
        
        except Exception as e:
            log.error(f"Error calculating threat trends: {e}")
            return {}


# Global security monitor instance
_security_monitor: Optional[SecurityMonitor] = None


async def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor instance."""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor


async def start_security_monitoring():
    """Start security monitoring system."""
    monitor = await get_security_monitor()
    await monitor.start_monitoring()


async def stop_security_monitoring():
    """Stop security monitoring system."""
    monitor = await get_security_monitor()
    await monitor.stop_monitoring()
