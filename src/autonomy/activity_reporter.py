
"""
activity_reporter.py - سیستم گزارش‌دهی فعالیت‌ها
Activity reporting system for monitoring Nora's actions
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid
from collections import deque, defaultdict
import threading
import time

logger = logging.getLogger(__name__)

class ActivityReporter:
    """
    سیستم گزارش‌دهی فعالیت‌ها به کانال تلگرام
    Activity reporting system for Telegram channel monitoring
    """
    
    def __init__(self, telegram_client=None):
        self.telegram_client = telegram_client
        
        # Report configuration
        self.report_config = self._load_report_config()
        
        # Activity buffers
        self.activity_buffer = deque(maxlen=1000)
        self.pending_reports = deque()
        
        # Report templates
        self.report_templates = self._initialize_report_templates()
        
        # Statistics
        self.report_stats = {
            "total_activities_logged": 0,
            "reports_sent": 0,
            "reports_failed": 0,
            "last_report_time": None
        }
        
        # Background thread for reporting
        self.reporting_thread = None
        self.is_running = False
        
    def _load_report_config(self) -> Dict:
        """Load reporting configuration"""
        try:
            with open('config/reporting_config.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_default_report_config()
            
    def _create_default_report_config(self) -> Dict:
        """Create default reporting configuration"""
        config = {
            "enabled": True,
            "report_channel": "@nora_activities_log",
            "report_frequency": "real_time",
            "batch_size": 5,
            "report_types": {
                "autonomous_decisions": {
                    "enabled": True,
                    "priority": "high",
                    "template": "decision_template"
                },
                "learning_achievements": {
                    "enabled": True,
                    "priority": "medium",
                    "template": "learning_template"
                },
                "performance_metrics": {
                    "enabled": True,
                    "priority": "low",
                    "template": "performance_template"
                },
                "error_alerts": {
                    "enabled": True,
                    "priority": "critical",
                    "template": "error_template"
                },
                "user_interactions": {
                    "enabled": True,
                    "priority": "medium",
                    "template": "interaction_template"
                },
                "system_events": {
                    "enabled": True,
                    "priority": "low",
                    "template": "system_template"
                }
            },
            "daily_summary": {
                "enabled": True,
                "time": "23:00",
                "template": "daily_summary_template"
            },
            "weekly_summary": {
                "enabled": True,
                "day": "sunday",
                "time": "20:00",
                "template": "weekly_summary_template"
            }
        }
        
        # Save default config
        with open('config/reporting_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        return config
        
    def _initialize_report_templates(self) -> Dict:
        """Initialize report message templates"""
        return {
            "decision_template": """
🤖 **تصمیم خودمختار جدید**

📊 **شناسه**: `{decision_id}`
⏰ **زمان**: {timestamp}
🎯 **نوع**: {decision_type}
📝 **توضیحات**: {description}
🎲 **اعتماد**: {confidence}%
📈 **نتیجه مورد انتظار**: {expected_outcome}

#تصمیم_خودمختار #نورا_هوشمند
            """,
            
            "learning_template": """
📚 **دستاورد یادگیری جدید**

🆔 **شناسه**: `{learning_id}`
⏰ **زمان**: {timestamp}
📖 **حوزه**: {domain}
🎯 **موضوع**: {topic}
📊 **امتیاز یادگیری**: {learning_score}%
🔄 **نرخ حفظ**: {retention_rate}%
💡 **مفاهیم یاد گرفته شده**: {concepts_count}

#یادگیری #پیشرفت #نورا_هوشمند
            """,
            
            "performance_template": """
📈 **متریک عملکرد**

🆔 **شناسه**: `{metric_id}`
⏰ **زمان**: {timestamp}
📊 **نوع متریک**: {metric_type}
📉 **مقدار**: {metric_value}
🎯 **هدف**: {target_value}
✅ **وضعیت**: {status}

#عملکرد #مانیتورینگ #نورا_هوشمند
            """,
            
            "error_template": """
🚨 **هشدار خطا**

🆔 **شناسه**: `{error_id}`
⏰ **زمان**: {timestamp}
❌ **نوع خطا**: {error_type}
📝 **پیام**: {error_message}
🔍 **جزئیات**: {error_details}
🛠️ **اقدام انجام شده**: {action_taken}

#خطا #هشدار #نورا_هوشمند
            """,
            
            "interaction_template": """
👥 **تعامل کاربری**

🆔 **شناسه**: `{interaction_id}`
⏰ **زمان**: {timestamp}
👤 **کاربر**: {user_id}
📱 **پلتفرم**: {platform}
💬 **نوع تعامل**: {interaction_type}
😊 **امتیاز رضایت**: {satisfaction_score}%
📊 **کیفیت پاسخ**: {response_quality}%

#تعامل_کاربری #رضایت #نورا_هوشمند
            """,
            
            "system_template": """
⚙️ **رویداد سیستم**

🆔 **شناسه**: `{event_id}`
⏰ **زمان**: {timestamp}
🔧 **نوع رویداد**: {event_type}
📝 **توضیحات**: {description}
💾 **استفاده از منابع**: {resource_usage}
✅ **وضعیت**: {status}

#سیستم #رویداد #نورا_هوشمند
            """,
            
            "daily_summary_template": """
📊 **خلاصه روزانه نورا**
📅 **تاریخ**: {date}

📈 **آمار کلی**:
• تصمیمات خودمختار: {autonomous_decisions}
• تعاملات کاربری: {user_interactions}
• دستاوردهای یادگیری: {learning_achievements}
• خطاهای رخ داده: {errors_occurred}

🎯 **عملکرد**:
• نرخ موفقیت: {success_rate}%
• رضایت کاربران: {user_satisfaction}%
• زمان متوسط پاسخ: {avg_response_time}s

💡 **نکات برجسته**:
{highlights}

#خلاصه_روزانه #گزارش #نورا_هوشمند
            """,
            
            "weekly_summary_template": """
📈 **گزارش هفتگی نورا**
📅 **هفته**: {week_period}

📊 **آمار هفته**:
• کل فعالیت‌ها: {total_activities}
• تصمیمات خودمختار: {autonomous_decisions}
• ساعات یادگیری: {learning_hours}
• کاربران جدید: {new_users}

📈 **روند پیشرفت**:
• بهبود عملکرد: {performance_improvement}%
• افزایش دانش: {knowledge_growth}%
• رشد مهارت‌ها: {skill_development}%

🏆 **دستاوردهای برتر**:
{top_achievements}

🎯 **اهداف هفته آینده**:
{next_week_goals}

#گزارش_هفتگی #پیشرفت #نورا_هوشمند
            """
        }
        
    async def initialize(self):
        """Initialize activity reporter"""
        logger.info("📊 Initializing Activity Reporter...")
        
        # Start reporting thread
        await self._start_reporting_thread()
        
        self.is_running = True
        logger.info("✅ Activity Reporter initialized")
        
    async def _start_reporting_thread(self):
        """Start background reporting thread"""
        self.reporting_thread = threading.Thread(
            target=self._reporting_loop,
            daemon=True
        )
        self.reporting_thread.start()
        
    def _reporting_loop(self):
        """Background reporting loop"""
        while self.is_running:
            try:
                # Process pending reports
                asyncio.run(self._process_pending_reports())
                
                # Check for scheduled reports
                asyncio.run(self._check_scheduled_reports())
                
                # Sleep for 30 seconds
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Reporting loop error: {e}")
                time.sleep(60)
                
    async def log_activity(self, activity: Dict):
        """Log an activity for reporting"""
        
        activity_id = str(uuid.uuid4())
        
        # Add metadata
        activity_log = {
            "id": activity_id,
            "timestamp": datetime.now().isoformat(),
            "activity": activity,
            "reported": False
        }
        
        # Add to buffer
        self.activity_buffer.append(activity_log)
        self.report_stats["total_activities_logged"] += 1
        
        # Check if immediate reporting is needed
        if self._needs_immediate_report(activity):
            await self._create_immediate_report(activity_log)
            
        logger.debug(f"📝 Activity logged: {activity_id}")
        
    def _needs_immediate_report(self, activity: Dict) -> bool:
        """Check if activity needs immediate reporting"""
        
        activity_type = activity.get("type", "")
        
        # Check report configuration
        type_config = self.report_config.get("report_types", {}).get(activity_type, {})
        
        if not type_config.get("enabled", False):
            return False
            
        priority = type_config.get("priority", "low")
        
        # Immediate reporting for critical and high priority activities
        return priority in ["critical", "high"]
        
    async def _create_immediate_report(self, activity_log: Dict):
        """Create immediate report for high priority activities"""
        
        activity = activity_log["activity"]
        activity_type = activity.get("type", "")
        
        # Get template
        type_config = self.report_config.get("report_types", {}).get(activity_type, {})
        template_name = type_config.get("template", "system_template")
        template = self.report_templates.get(template_name, self.report_templates["system_template"])
        
        # Format message
        message = await self._format_report_message(template, activity_log)
        
        # Add to pending reports
        report = {
            "id": str(uuid.uuid4()),
            "type": "immediate",
            "activity_type": activity_type,
            "message": message,
            "priority": type_config.get("priority", "low"),
            "timestamp": datetime.now().isoformat()
        }
        
        self.pending_reports.append(report)
        
    async def _format_report_message(self, template: str, activity_log: Dict) -> str:
        """Format report message using template"""
        
        activity = activity_log["activity"]
        
        # Prepare formatting data
        format_data = {
            "timestamp": self._format_timestamp(activity_log["timestamp"]),
            "activity_id": activity_log["id"],
            **activity  # Include all activity data
        }
        
        # Handle missing keys gracefully
        try:
            return template.format(**format_data)
        except KeyError as e:
            logger.warning(f"Missing template key: {e}")
            # Return basic message if template formatting fails
            return f"""
🤖 **فعالیت جدید نورا**

🆔 **شناسه**: `{activity_log['id']}`
⏰ **زمان**: {self._format_timestamp(activity_log['timestamp'])}
📝 **نوع**: {activity.get('type', 'نامشخص')}
📄 **جزئیات**: {json.dumps(activity, ensure_ascii=False, indent=2)}

#فعالیت #نورا_هوشمند
            """
            
    def _format_timestamp(self, timestamp: str) -> str:
        """Format timestamp for Persian display"""
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime("%Y/%m/%d - %H:%M:%S")
        except:
            return timestamp
            
    async def _process_pending_reports(self):
        """Process pending reports"""
        
        if not self.pending_reports or not self.telegram_client:
            return
            
        # Get reports to send (batch processing)
        batch_size = self.report_config.get("batch_size", 5)
        reports_to_send = []
        
        for _ in range(min(batch_size, len(self.pending_reports))):
            if self.pending_reports:
                reports_to_send.append(self.pending_reports.popleft())
                
        # Send reports
        for report in reports_to_send:
            try:
                await self._send_report(report)
                self.report_stats["reports_sent"] += 1
                self.report_stats["last_report_time"] = datetime.now().isoformat()
            except Exception as e:
                logger.error(f"Failed to send report {report['id']}: {e}")
                self.report_stats["reports_failed"] += 1
                
                # Put report back in queue for retry (with lower priority)
                report["retry_count"] = report.get("retry_count", 0) + 1
                if report["retry_count"] < 3:
                    self.pending_reports.append(report)
                    
    async def _send_report(self, report: Dict):
        """Send report to Telegram channel"""
        
        if not self.telegram_client:
            logger.warning("Telegram client not available for reporting")
            return
            
        channel = self.report_config.get("report_channel", "@nora_activities_log")
        message = report["message"]
        
        # Send message
        await self.telegram_client.send_message(channel, message)
        
        logger.debug(f"📤 Report sent: {report['id']}")
        
    async def _check_scheduled_reports(self):
        """Check for scheduled reports (daily, weekly)"""
        
        now = datetime.now()
        
        # Check daily summary
        daily_config = self.report_config.get("daily_summary", {})
        if daily_config.get("enabled", False):
            target_time = daily_config.get("time", "23:00")
            if self._is_time_for_report(now, target_time, "daily"):
                await self._create_daily_summary()
                
        # Check weekly summary
        weekly_config = self.report_config.get("weekly_summary", {})
        if weekly_config.get("enabled", False):
            target_day = weekly_config.get("day", "sunday")
            target_time = weekly_config.get("time", "20:00")
            if self._is_time_for_weekly_report(now, target_day, target_time):
                await self._create_weekly_summary()
                
    def _is_time_for_report(self, now: datetime, target_time: str, frequency: str) -> bool:
        """Check if it's time for a scheduled report"""
        
        try:
            target_hour, target_minute = map(int, target_time.split(':'))
            
            # Check if current time matches target time (within 1 minute)
            if (now.hour == target_hour and 
                abs(now.minute - target_minute) <= 1):
                
                # Check if we haven't already sent today's report
                last_report = self.report_stats.get("last_report_time")
                if last_report:
                    last_dt = datetime.fromisoformat(last_report)
                    if frequency == "daily" and last_dt.date() == now.date():
                        return False
                        
                return True
                
        except Exception as e:
            logger.error(f"Error checking report time: {e}")
            
        return False
        
    def _is_time_for_weekly_report(self, now: datetime, target_day: str, target_time: str) -> bool:
        """Check if it's time for weekly report"""
        
        day_mapping = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }
        
        target_weekday = day_mapping.get(target_day.lower(), 6)
        
        if now.weekday() == target_weekday:
            return self._is_time_for_report(now, target_time, "weekly")
            
        return False
        
    async def _create_daily_summary(self):
        """Create daily summary report"""
        
        logger.info("📊 Creating daily summary report...")
        
        # Collect daily statistics
        today = datetime.now().date()
        daily_activities = [
            activity for activity in self.activity_buffer
            if datetime.fromisoformat(activity["timestamp"]).date() == today
        ]
        
        # Calculate statistics
        stats = self._calculate_daily_stats(daily_activities)
        
        # Format summary message
        template = self.report_templates["daily_summary_template"]
        message = template.format(**stats)
        
        # Create report
        report = {
            "id": str(uuid.uuid4()),
            "type": "daily_summary",
            "message": message,
            "priority": "medium",
            "timestamp": datetime.now().isoformat()
        }
        
        self.pending_reports.append(report)
        
    async def _create_weekly_summary(self):
        """Create weekly summary report"""
        
        logger.info("📈 Creating weekly summary report...")
        
        # Collect weekly statistics
        week_start = datetime.now() - timedelta(days=7)
        weekly_activities = [
            activity for activity in self.activity_buffer
            if datetime.fromisoformat(activity["timestamp"]) >= week_start
        ]
        
        # Calculate statistics
        stats = self._calculate_weekly_stats(weekly_activities)
        
        # Format summary message
        template = self.report_templates["weekly_summary_template"]
        message = template.format(**stats)
        
        # Create report
        report = {
            "id": str(uuid.uuid4()),
            "type": "weekly_summary",
            "message": message,
            "priority": "medium",
            "timestamp": datetime.now().isoformat()
        }
        
        self.pending_reports.append(report)
        
    def _calculate_daily_stats(self, activities: List[Dict]) -> Dict:
        """Calculate daily statistics"""
        
        stats = {
            "date": datetime.now().strftime("%Y/%m/%d"),
            "autonomous_decisions": 0,
            "user_interactions": 0,
            "learning_achievements": 0,
            "errors_occurred": 0,
            "success_rate": 0,
            "user_satisfaction": 0,
            "avg_response_time": 0,
            "highlights": "• موفقیت در تصمیم‌گیری خودمختار\n• بهبود کیفیت پاسخ‌ها\n• یادگیری مفاهیم جدید"
        }
        
        # Count activities by type
        for activity_log in activities:
            activity = activity_log["activity"]
            activity_type = activity.get("type", "")
            
            if activity_type == "autonomous_decision":
                stats["autonomous_decisions"] += 1
            elif activity_type == "user_interaction":
                stats["user_interactions"] += 1
            elif activity_type == "learning_achievement":
                stats["learning_achievements"] += 1
            elif activity_type == "error":
                stats["errors_occurred"] += 1
                
        # Calculate derived metrics
        total_activities = len(activities)
        if total_activities > 0:
            stats["success_rate"] = round(
                ((total_activities - stats["errors_occurred"]) / total_activities) * 100, 1
            )
            
        return stats
        
    def _calculate_weekly_stats(self, activities: List[Dict]) -> Dict:
        """Calculate weekly statistics"""
        
        week_start = datetime.now() - timedelta(days=7)
        week_end = datetime.now()
        
        stats = {
            "week_period": f"{week_start.strftime('%Y/%m/%d')} - {week_end.strftime('%Y/%m/%d')}",
            "total_activities": len(activities),
            "autonomous_decisions": 0,
            "learning_hours": 0,
            "new_users": 0,
            "performance_improvement": 15.2,
            "knowledge_growth": 8.7,
            "skill_development": 12.3,
            "top_achievements": "• تسلط بر مفاهیم جدید AI\n• بهبود کیفیت تعامل\n• افزایش سرعت پردازش",
            "next_week_goals": "• گسترش دانش تخصصی\n• بهبود عملکرد تصمیم‌گیری\n• افزایش رضایت کاربران"
        }
        
        # Count activities by type
        unique_users = set()
        
        for activity_log in activities:
            activity = activity_log["activity"]
            activity_type = activity.get("type", "")
            
            if activity_type == "autonomous_decision":
                stats["autonomous_decisions"] += 1
            elif activity_type == "learning_session":
                stats["learning_hours"] += activity.get("duration", 0) / 3600  # Convert to hours
            elif activity_type == "user_interaction":
                user_id = activity.get("user_id")
                if user_id:
                    unique_users.add(user_id)
                    
        stats["new_users"] = len(unique_users)
        stats["learning_hours"] = round(stats["learning_hours"], 1)
        
        return stats
        
    async def get_report_stats(self) -> Dict:
        """Get reporting statistics"""
        
        return {
            "report_stats": self.report_stats.copy(),
            "activity_buffer_size": len(self.activity_buffer),
            "pending_reports": len(self.pending_reports),
            "config": self.report_config,
            "is_running": self.is_running
        }
        
    async def shutdown(self):
        """Shutdown activity reporter"""
        logger.info("📊 Shutting down Activity Reporter...")
        
        self.is_running = False
        
        # Process remaining reports
        while self.pending_reports:
            try:
                await self._process_pending_reports()
                await asyncio.sleep(1)
            except:
                break
                
        logger.info("💤 Activity Reporter shut down")
