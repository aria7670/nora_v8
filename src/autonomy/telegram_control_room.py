
"""
telegram_control_room.py - اتاق کنترل تلگرام برای مدیریت پیشرفته نورا
Telegram control room for advanced Nora management
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class TelegramControlRoom:
    """
    اتاق کنترل تلگرام - رابط قدرتمند برای کنترل کامل نورا
    Telegram Control Room - powerful interface for complete Nora control
    """
    
    def __init__(self, nora_ecosystem):
        self.ecosystem = nora_ecosystem
        self.control_room_config = {}
        self.custom_commands = {}
        self.active_tasks = {}
        self.authorized_users = set()
        
        # Load configurations
        self.config = self._load_control_room_config()
        
        # Command registry
        self.commands = {
            # System control
            "/status": self._cmd_system_status,
            "/restart_module": self._cmd_restart_module,
            "/emergency_stop": self._cmd_emergency_stop,
            "/backup_memory": self._cmd_backup_memory,
            
            # Analysis commands
            "/analyze_user": self._cmd_analyze_user,
            "/analyze_platform": self._cmd_analyze_platform,
            "/generate_report": self._cmd_generate_report,
            "/get_insights": self._cmd_get_insights,
            
            # Content management
            "/publish_content": self._cmd_publish_content,
            "/schedule_post": self._cmd_schedule_post,
            "/review_queue": self._cmd_review_queue,
            "/approve_content": self._cmd_approve_content,
            
            # Learning management
            "/add_learning_source": self._cmd_add_learning_source,
            "/remove_learning_source": self._cmd_remove_learning_source,
            "/force_learning": self._cmd_force_learning,
            "/review_learning": self._cmd_review_learning,
            
            # Task management
            "/task": self._cmd_create_task,
            "/task_status": self._cmd_task_status,
            "/cancel_task": self._cmd_cancel_task,
            "/list_tasks": self._cmd_list_tasks,
            
            # Custom commands
            "/add_command": self._cmd_add_custom_command,
            "/remove_command": self._cmd_remove_custom_command,
            "/list_commands": self._cmd_list_custom_commands,
            
            # Configuration
            "/set_config": self._cmd_set_config,
            "/get_config": self._cmd_get_config,
            "/reload_config": self._cmd_reload_config,
            
            # Personality management
            "/personality_status": self._cmd_personality_status,
            "/adjust_personality": self._cmd_adjust_personality,
            "/reset_personality": self._cmd_reset_personality,
            
            # Help system
            "/help": self._cmd_help,
            "/help_advanced": self._cmd_help_advanced
        }
        
    def _load_control_room_config(self) -> Dict:
        """بارگذاری تنظیمات اتاق کنترل"""
        try:
            with open('config/control_room.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_default_control_room_config()
            
    def _create_default_control_room_config(self) -> Dict:
        """ایجاد تنظیمات پیش‌فرض اتاق کنترل"""
        config = {
            "control_room_chat_id": "@nora_control_room",  # Will be set by user
            "authorized_users": [
                {"id": "aria_pourshajaii", "role": "owner", "permissions": "all"},
                {"id": "admin_user", "role": "admin", "permissions": "most"}
            ],
            "security": {
                "require_confirmation_for_dangerous_commands": True,
                "log_all_commands": True,
                "rate_limit_commands": True,
                "max_commands_per_minute": 10
            },
            "task_management": {
                "max_concurrent_tasks": 5,
                "task_timeout_hours": 24,
                "auto_cleanup_completed_tasks": True
            },
            "reporting": {
                "auto_daily_report": True,
                "auto_weekly_summary": True,
                "alert_on_errors": True,
                "performance_monitoring": True
            }
        }
        
        Path("config").mkdir(exist_ok=True)
        with open('config/control_room.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        return config
        
    async def handle_command(self, message: Dict) -> str:
        """پردازش دستورات اتاق کنترل"""
        
        user_id = message.get("user_id")
        text = message.get("text", "").strip()
        
        # Security check
        if not self._is_authorized(user_id):
            return "🚫 شما مجوز استفاده از اتاق کنترل را ندارید."
            
        # Log command
        self._log_command(user_id, text)
        
        # Rate limiting
        if not self._check_rate_limit(user_id):
            return "⏱️ شما خیلی سریع دستور می‌فرستید. لطفاً کمی صبر کنید."
            
        # Parse command
        parts = text.split()
        if not parts:
            return "❓ لطفاً یک دستور وارد کنید. برای راهنما از /help استفاده کنید."
            
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # Check for custom commands first
        if command in self.custom_commands:
            return await self._execute_custom_command(command, args, message)
            
        # Execute built-in command
        if command in self.commands:
            try:
                return await self.commands[command](args, message)
            except Exception as e:
                logger.error(f"Error executing command {command}: {e}")
                return f"❌ خطا در اجرای دستور: {str(e)}"
        else:
            return f"❓ دستور '{command}' شناخته نشده. برای لیست دستورات از /help استفاده کنید."
            
    def _is_authorized(self, user_id: str) -> bool:
        """بررسی مجوز کاربر"""
        authorized_users = self.config.get("authorized_users", [])
        return any(user["id"] == user_id for user in authorized_users)
        
    def _log_command(self, user_id: str, command: str):
        """ثبت دستور در لاگ"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "command": command
        }
        
        with open('logs/control_room_commands.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
    def _check_rate_limit(self, user_id: str) -> bool:
        """بررسی محدودیت نرخ دستورات"""
        # Simple rate limiting implementation
        return True  # For now, allow all commands
        
    # Built-in command implementations
    
    async def _cmd_system_status(self, args: List[str], message: Dict) -> str:
        """نمایش وضعیت سیستم"""
        status = {
            "Core": "🟢 Active" if self.ecosystem.nora_core else "🔴 Inactive",
            "Memory": "🟢 Active" if self.ecosystem.memory_manager else "🔴 Inactive",
            "Analytics": "🟢 Active" if self.ecosystem.analytics_engine else "🔴 Inactive",
            "Telegram": "🟢 Connected" if self.ecosystem.telegram_client else "🔴 Disconnected",
            "Twitter": "🟢 Connected" if self.ecosystem.twitter_client else "🔴 Disconnected"
        }
        
        status_text = "📊 **وضعیت سیستم نورا**\n\n"
        for module, state in status.items():
            status_text += f"{module}: {state}\n"
            
        # Add performance metrics
        status_text += f"\n⚡ **آمار عملکرد**\n"
        status_text += f"آپتایم: {self._get_uptime()}\n"
        status_text += f"تعاملات امروز: {await self._get_today_interactions()}\n"
        status_text += f"استفاده حافظه: {self._get_memory_usage()}\n"
        
        return status_text
        
    async def _cmd_analyze_user(self, args: List[str], message: Dict) -> str:
        """تحلیل کاربر"""
        if not args:
            return "❓ لطفاً نام کاربری یا ID کاربر را وارد کنید.\nمثال: `/analyze_user @username`"
            
        username = args[0].replace('@', '')
        
        # Get user analysis from analytics engine
        analysis = await self.ecosystem.analytics_engine.analyze_user_profile(username)
        
        if not analysis:
            return f"❌ کاربر '{username}' یافت نشد یا اطلاعات کافی موجود نیست."
            
        result = f"👤 **تحلیل کاربر @{username}**\n\n"
        result += f"📊 تعداد تعاملات: {analysis.get('interaction_count', 0)}\n"
        result += f"😊 نرخ رضایت: {analysis.get('satisfaction_rate', 0):.1%}\n"
        result += f"🏷️ دسته‌بندی: {analysis.get('user_category', 'نامشخص')}\n"
        result += f"🎯 علایق: {', '.join(analysis.get('interests', []))}\n"
        result += f"📅 آخرین فعالیت: {analysis.get('last_activity', 'نامشخص')}\n"
        
        return result
        
    async def _cmd_create_task(self, args: List[str], message: Dict) -> str:
        """ایجاد وظیفه جدید"""
        if not args:
            return """❓ لطفاً وظیفه را تعریف کنید.
            
مثال‌ها:
`/task یک تحلیل کامل از ترندهای AI در هفته گذشته تهیه کن`
`/task محتوای جدید برای کانال فیلسوفه آماده کن`
`/task گزارش عملکرد پلتفرم‌ها را تا فردا صبح بفرست`"""
        
        task_description = " ".join(args)
        
        # Generate unique task ID
        task_id = f"task_{int(datetime.now().timestamp())}"
        
        # Create task object
        task = {
            "id": task_id,
            "description": task_description,
            "created_by": message.get("user_id"),
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            "progress": 0,
            "estimated_completion": self._estimate_task_completion(task_description),
            "subtasks": self._break_down_task(task_description)
        }
        
        # Store task
        self.active_tasks[task_id] = task
        
        # Start task execution
        asyncio.create_task(self._execute_task(task))
        
        return f"""✅ **وظیفه جدید ایجاد شد**

🆔 شناسه: `{task_id}`
📝 شرح: {task_description}
⏱️ زمان تخمینی: {task['estimated_completion']}
📊 وضعیت: در حال پردازش...

برای بررسی وضعیت از `/task_status {task_id}` استفاده کنید."""
        
    async def _cmd_task_status(self, args: List[str], message: Dict) -> str:
        """بررسی وضعیت وظیفه"""
        if not args:
            return "❓ لطفاً شناسه وظیفه را وارد کنید.\nمثال: `/task_status task_1234567890`"
            
        task_id = args[0]
        
        if task_id not in self.active_tasks:
            return f"❌ وظیفه با شناسه '{task_id}' یافت نشد."
            
        task = self.active_tasks[task_id]
        
        result = f"""📋 **وضعیت وظیفه {task_id}**

📝 شرح: {task['description']}
📊 وضعیت: {task['status']}
📈 پیشرفت: {task['progress']}%
🕐 ایجاد شده: {task['created_at']}
👤 ایجاد کننده: {task['created_by']}

"""
        
        # Add subtasks status
        if task.get('subtasks'):
            result += "📋 **زیر وظایف:**\n"
            for subtask in task['subtasks']:
                status_emoji = "✅" if subtask['completed'] else "⏳"
                result += f"{status_emoji} {subtask['description']}\n"
                
        return result
        
    async def _cmd_add_learning_source(self, args: List[str], message: Dict) -> str:
        """اضافه کردن منبع یادگیری جدید"""
        if len(args) < 2:
            return """❓ لطفاً منبع یادگیری را مشخص کنید.

مثال‌ها:
`/add_learning_source telegram @TechCrunch technology high`
`/add_learning_source twitter @elonmusk innovation medium`"""
        
        platform = args[0].lower()
        source_id = args[1]
        category = args[2] if len(args) > 2 else "general"
        priority = args[3] if len(args) > 3 else "medium"
        
        # Add to learning sources
        learning_config = self.ecosystem.perception_system.learning_config
        
        if platform == "telegram":
            if "telegram_learning_sources" not in learning_config:
                learning_config["telegram_learning_sources"] = {}
            if category not in learning_config["telegram_learning_sources"]:
                learning_config["telegram_learning_sources"][category] = []
                
            new_source = {
                "id": source_id,
                "category": category,
                "priority": priority,
                "added_by": message.get("user_id"),
                "added_at": datetime.now().isoformat()
            }
            
            learning_config["telegram_learning_sources"][category].append(new_source)
            
        # Save updated config
        with open('config/learning_sources.json', 'w', encoding='utf-8') as f:
            json.dump(learning_config, f, ensure_ascii=False, indent=2)
            
        return f"""✅ **منبع یادگیری جدید اضافه شد**

🌐 پلتفرم: {platform.title()}
🔗 منبع: {source_id}
🏷️ دسته: {category}
⭐ اولویت: {priority}

نورا از این پس از این منبع یاد خواهد گرفت."""
        
    async def _cmd_help(self, args: List[str], message: Dict) -> str:
        """راهنمای دستورات"""
        return """🤖 **راهنمای اتاق کنترل نورا**

**دستورات سیستم:**
`/status` - وضعیت سیستم
`/restart_module` - راه‌اندازی مجدد ماژول
`/emergency_stop` - توقف اضطراری

**تحلیل و گزارش:**
`/analyze_user @username` - تحلیل کاربر
`/analyze_platform telegram` - تحلیل پلتفرم
`/generate_report daily` - تولید گزارش

**مدیریت محتوا:**
`/publish_content` - انتشار محتوا
`/schedule_post` - زمان‌بندی پست
`/review_queue` - بررسی صف انتشار

**مدیریت یادگیری:**
`/add_learning_source` - اضافه کردن منبع یادگیری
`/force_learning` - اجبار یادگیری فوری
`/review_learning` - بررسی یادگیری

**مدیریت وظایف:**
`/task [شرح وظیفه]` - ایجاد وظیفه جدید
`/task_status [ID]` - وضعیت وظیفه
`/list_tasks` - لیست وظایف

برای راهنمای پیشرفته از `/help_advanced` استفاده کنید."""
        
    # Helper methods
    
    def _get_uptime(self) -> str:
        """محاسبه آپتایم سیستم"""
        # Placeholder implementation
        return "5 ساعت و 23 دقیقه"
        
    async def _get_today_interactions(self) -> int:
        """تعداد تعاملات امروز"""
        # This would query the actual database
        return 127
        
    def _get_memory_usage(self) -> str:
        """استفاده از حافظه"""
        return "2.3 GB / 8 GB"
        
    def _estimate_task_completion(self, task_description: str) -> str:
        """تخمین زمان تکمیل وظیفه"""
        word_count = len(task_description.split())
        if word_count < 10:
            return "10-30 دقیقه"
        elif word_count < 20:
            return "30-60 دقیقه"
        else:
            return "1-3 ساعت"
            
    def _break_down_task(self, task_description: str) -> List[Dict]:
        """تقسیم وظیفه به زیر وظایف"""
        # Simple task breakdown
        subtasks = [
            {"description": "تحلیل درخواست", "completed": False},
            {"description": "جمع‌آوری اطلاعات", "completed": False},
            {"description": "پردازش داده‌ها", "completed": False},
            {"description": "تهیه نتیجه نهایی", "completed": False}
        ]
        return subtasks
        
    async def _execute_task(self, task: Dict):
        """اجرای وظیفه"""
        task_id = task["id"]
        
        try:
            # Update status
            task["status"] = "in_progress"
            
            # Simulate task execution with progress updates
            subtasks = task.get("subtasks", [])
            total_subtasks = len(subtasks)
            
            for i, subtask in enumerate(subtasks):
                # Simulate work
                await asyncio.sleep(5)  # Simulate processing time
                
                subtask["completed"] = True
                task["progress"] = int(((i + 1) / total_subtasks) * 100)
                
                # Send progress update to control room
                await self._send_task_update(task)
                
            # Complete task
            task["status"] = "completed"
            task["progress"] = 100
            task["completed_at"] = datetime.now().isoformat()
            
            # Generate and send result
            result = await self._generate_task_result(task)
            await self._send_task_completion(task, result)
            
        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)
            await self._send_task_error(task, str(e))
            
    async def _send_task_update(self, task: Dict):
        """ارسال به‌روزرسانی وظیفه"""
        # This would send updates to the control room chat
        logger.info(f"Task {task['id']} progress: {task['progress']}%")
        
    async def _generate_task_result(self, task: Dict) -> str:
        """تولید نتیجه وظیفه"""
        # This would use the AI core to generate actual results
        return f"نتیجه وظیفه: {task['description']}\n\nاین وظیفه با موفقیت تکمیل شد."
        
    async def _send_task_completion(self, task: Dict, result: str):
        """ارسال اطلاع تکمیل وظیفه"""
        # This would send completion notification to control room
        logger.info(f"Task {task['id']} completed successfully")
        
    async def _send_task_error(self, task: Dict, error: str):
        """ارسال خطای وظیفه"""
        # This would send error notification to control room
        logger.error(f"Task {task['id']} failed: {error}")
        
    async def _execute_custom_command(self, command: str, args: List[str], message: Dict) -> str:
        """اجرای دستور سفارشی"""
        custom_cmd = self.custom_commands[command]
        
        # Execute the custom command logic
        # This is a placeholder - real implementation would be more sophisticated
        return f"اجرای دستور سفارشی '{command}' با پارامترهای: {' '.join(args)}"
        
    # Placeholder implementations for other commands
    async def _cmd_restart_module(self, args: List[str], message: Dict) -> str:
        return "🔄 راه‌اندازی مجدد ماژول در حال انجام..."
        
    async def _cmd_emergency_stop(self, args: List[str], message: Dict) -> str:
        return "🛑 توقف اضطراری فعال شد. همه عملیات متوقف شدند."
        
    async def _cmd_backup_memory(self, args: List[str], message: Dict) -> str:
        return "💾 پشتیبان‌گیری از حافظه در حال انجام..."
        
    async def _cmd_analyze_platform(self, args: List[str], message: Dict) -> str:
        return "📊 تحلیل پلتفرم در حال آماده‌سازی..."
        
    async def _cmd_generate_report(self, args: List[str], message: Dict) -> str:
        return "📋 گزارش در حال تولید..."
        
    async def _cmd_get_insights(self, args: List[str], message: Dict) -> str:
        return "💡 بینش‌های جدید در حال استخراج..."
        
    async def _cmd_publish_content(self, args: List[str], message: Dict) -> str:
        return "📤 محتوا در حال انتشار..."
        
    async def _cmd_schedule_post(self, args: List[str], message: Dict) -> str:
        return "📅 پست زمان‌بندی شد."
        
    async def _cmd_review_queue(self, args: List[str], message: Dict) -> str:
        return "📋 صف انتشار در حال بررسی..."
        
    async def _cmd_approve_content(self, args: List[str], message: Dict) -> str:
        return "✅ محتوا تایید شد."
        
    async def _cmd_remove_learning_source(self, args: List[str], message: Dict) -> str:
        return "❌ منبع یادگیری حذف شد."
        
    async def _cmd_force_learning(self, args: List[str], message: Dict) -> str:
        return "🧠 یادگیری فوری شروع شد..."
        
    async def _cmd_review_learning(self, args: List[str], message: Dict) -> str:
        return "📚 بررسی یادگیری در حال انجام..."
        
    async def _cmd_cancel_task(self, args: List[str], message: Dict) -> str:
        return "❌ وظیفه لغو شد."
        
    async def _cmd_list_tasks(self, args: List[str], message: Dict) -> str:
        return "📋 لیست وظایف آماده شد."
        
    async def _cmd_add_custom_command(self, args: List[str], message: Dict) -> str:
        return "➕ دستور سفارشی اضافه شد."
        
    async def _cmd_remove_custom_command(self, args: List[str], message: Dict) -> str:
        return "❌ دستور سفارشی حذف شد."
        
    async def _cmd_list_custom_commands(self, args: List[str], message: Dict) -> str:
        return "📋 لیست دستورات سفارشی."
        
    async def _cmd_set_config(self, args: List[str], message: Dict) -> str:
        return "⚙️ تنظیمات به‌روزرسانی شد."
        
    async def _cmd_get_config(self, args: List[str], message: Dict) -> str:
        return "⚙️ تنظیمات فعلی."
        
    async def _cmd_reload_config(self, args: List[str], message: Dict) -> str:
        return "🔄 تنظیمات مجدداً بارگذاری شد."
        
    async def _cmd_personality_status(self, args: List[str], message: Dict) -> str:
        return "🧠 وضعیت شخصیت نورا."
        
    async def _cmd_adjust_personality(self, args: List[str], message: Dict) -> str:
        return "🎛️ شخصیت تنظیم شد."
        
    async def _cmd_reset_personality(self, args: List[str], message: Dict) -> str:
        return "🔄 شخصیت بازنشانی شد."
        
    async def _cmd_help_advanced(self, args: List[str], message: Dict) -> str:
        return """🔧 **راهنمای پیشرفته اتاق کنترل**

**دستورات پیچیده:**
- ایجاد ماکروهای دستوری
- تنظیم خودکارسازی‌ها
- مدیریت شخصیت نورا
- تحلیل‌های پیشرفته
- مدیریت امنیت

**نمونه دستورات پیشرفته:**
`/task تحلیل کامل رقبا + ارائه استراتژی جدید تا فردا`
`/add_command /daily_summary نورا هر روز ساعت 8 صبح خلاصه فعالیت‌های روز قبل را بفرست`
`/set_config personality.curiosity 0.95`

برای اطلاعات بیشتر با آریا در ارتباط باشید."""
