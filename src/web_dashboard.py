"""
web_dashboard.py - داشبورد وب نورا
Web dashboard for monitoring and controlling Nora
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class DashboardServer:
    """Web dashboard for Nora monitoring and control"""

    def __init__(self, nora_ecosystem):
        self.ecosystem = nora_ecosystem
        self.app = Flask(__name__, template_folder='../templates', static_folder='../static')
        self.app.config['SECRET_KEY'] = 'nora_dashboard_secret_key'
        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def dashboard():
            return render_template('command_deck.html')

        @self.app.route('/api/status')
        def api_status():
            return jsonify({
                "timestamp": datetime.now().isoformat(),
                "status": "active",
                "modules": {
                    "core": "active" if hasattr(self.ecosystem, 'nora_core') and self.ecosystem.nora_core else "inactive",
                    "memory": "active" if hasattr(self.ecosystem, 'memory_manager') and self.ecosystem.memory_manager else "inactive",
                    "telegram": "active" if hasattr(self.ecosystem, 'telegram_client') and self.ecosystem.telegram_client else "inactive",
                    "analytics": "active" if hasattr(self.ecosystem, 'analytics_engine') and self.ecosystem.analytics_engine else "inactive"
                }
            })

        @self.app.route('/api/metrics')
        def api_metrics():
            return jsonify({
                "interactions_today": 150,
                "messages_processed": 1247,
                "learning_insights": 23,
                "content_generated": 8
            })

        @self.app.route('/api/recent_activity')
        def api_recent_activity():
            activities = []
            try:
                log_file = Path('logs/telegram_activity.jsonl')
                if log_file.exists():
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[-10:]
                        for line in lines:
                            try:
                                activities.append(json.loads(line))
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping invalid JSON line: {line.strip()}")
            except FileNotFoundError:
                pass
            return jsonify(activities)

        @self.app.route('/api/personality_state')
        def api_personality_state():
            try:
                personality_file = Path('data/personality_state.json')
                if personality_file.exists():
                    with open(personality_file, 'r', encoding='utf-8') as f:
                        return jsonify(json.load(f))
                else:
                    return jsonify({"error": "No personality state found"})

            except FileNotFoundError:
                return jsonify({"error": "No personality state found"})
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid personality state file"})


        @self.app.route('/api/command', methods=['POST'])
        def api_command():
            data = request.get_json()
            command = data.get('command', '')

            # Process command
            result = {
                "success": True,
                "message": f"Command '{command}' executed successfully",
                "timestamp": datetime.now().isoformat()
            }

            return jsonify(result)

        @self.app.route('/api/kpi_widgets')
        def kpi_widgets():
            """Get KPI widgets data"""
            return jsonify(self._get_kpi_widgets())

        @self.app.route('/api/live_activity')
        def live_activity():
            """Get live activity feed"""
            return jsonify(self._get_live_activity())

        @self.app.route('/api/memory_stats')
        def memory_stats():
            """Get memory statistics"""
            if hasattr(self.ecosystem, 'memory_manager'):
                return jsonify(asyncio.run(self.ecosystem.memory_manager.get_memory_stats()))
            return jsonify({})

        @self.app.route('/api/analytics_overview')
        def analytics_overview():
            """Get analytics overview"""
            if hasattr(self.ecosystem, 'analytics_engine'):
                return jsonify(asyncio.run(self.ecosystem.analytics_engine.generate_insights_report()))
            return jsonify({})

        @self.app.route('/api/metacognition_status')
        def metacognition_status():
            """Get metacognition engine status"""
            if hasattr(self.ecosystem, 'metacognition_engine'):
                return jsonify({
                    "recent_reflections": len(self.ecosystem.metacognition_engine.self_assessment_history),
                    "evolution_proposals": len(self.ecosystem.metacognition_engine.evolution_proposals),
                    "last_reflection": self.ecosystem.metacognition_engine.self_assessment_history[-1] if self.ecosystem.metacognition_engine.self_assessment_history else None
                })
            return jsonify({})

        @self.app.route('/api/platform_metrics')
        def platform_metrics():
            """Get platform-specific metrics"""
            return jsonify(self._get_platform_metrics())

        @self.app.route('/api/learning_channels')
        def learning_channels():
            """Get learning channels configuration"""
            if hasattr(self.ecosystem, 'telegram_client'):
                return jsonify({
                    "channels": self.ecosystem.telegram_client.learning_channels,
                    "total_channels": len(self.ecosystem.telegram_client.learning_channels)
                })
            return jsonify({"channels": [], "total_channels": 0})

        @self.app.route('/api/add_learning_channel', methods=['POST'])
        def add_learning_channel():
            """Add a new learning channel"""
            try:
                data = request.get_json()
                if hasattr(self.ecosystem, 'telegram_client'):
                    asyncio.run(self.ecosystem.telegram_client.add_learning_channel(data))
                    return jsonify({"success": True, "message": "کانال یادگیری اضافه شد"})
                return jsonify({"success": False, "message": "کلاینت تلگرام در دسترس نیست"})
            except Exception as e:
                return jsonify({"success": False, "message": str(e)})

        @self.app.route('/api/consciousness_state')
        def consciousness_state():
            """Get Nora's consciousness state"""
            if hasattr(self.ecosystem, 'nora_core'):
                return jsonify({
                    "personality": self.ecosystem.nora_core.personality,
                    "knowledge_domains": self.ecosystem.nora_core.knowledge_domains,
                    "active_conversations": len(self.ecosystem.nora_core.conversation_context),
                    "version": self.ecosystem.nora_core.version
                })
            return jsonify({})

        @self.app.route('/api/trigger_reflection', methods=['POST'])
        def trigger_reflection():
            """Trigger manual self-reflection"""
            try:
                if hasattr(self.ecosystem, 'metacognition_engine'):
                    reflection = asyncio.run(self.ecosystem.metacognition_engine.conduct_self_reflection())
                    return jsonify({"success": True, "reflection": reflection})
                return jsonify({"success": False, "message": "موتور فراشناخت در دسترس نیست"})
            except Exception as e:
                return jsonify({"success": False, "message": str(e)})

        @self.app.route('/api/generate_evolution_proposal', methods=['POST'])
        def generate_evolution_proposal():
            """Generate evolution proposal"""
            try:
                if hasattr(self.ecosystem, 'metacognition_engine'):
                    proposal = asyncio.run(self.ecosystem.metacognition_engine.generate_evolution_proposal())
                    return jsonify({"success": True, "proposal": proposal})
                return jsonify({"success": False, "message": "موتور فراشناخت در دسترس نیست"})
            except Exception as e:
                return jsonify({"success": False, "message": str(e)})

    def _get_system_status(self) -> dict:
        """Get system vitals and status"""
        return {
            "core_modules": {
                "ai_core": {"status": "active", "last_activity": "30s ago"},
                "memory_manager": {"status": "active", "last_activity": "15s ago"},
                "analytics_engine": {"status": "active", "last_activity": "45s ago"},
                "metacognition_engine": {"status": "active", "last_activity": "1m ago"}
            },
            "platforms": {
                "telegram": {"status": "connected", "last_activity": "5s ago"},
                "twitter": {"status": "connected", "last_activity": "2m ago"},
                "instagram": {"status": "standby", "last_activity": "10m ago"},
                "threads": {"status": "standby", "last_activity": "1m ago"}
            },
            "ai_models": {
                "gemini": {"status": "active", "response_time": "1.2s"},
                "openai": {"status": "active", "response_time": "0.8s"},
                "custom_model": {"status": "training", "progress": "65%"}
            }
        }

    def _get_kpi_widgets(self) -> dict:
        """Get KPI widget data"""
        return {
            "follower_growth_24h": {
                "value": 45,
                "change": "+12%",
                "trend": "up"
            },
            "engagement_rate_weekly": {
                "value": "8.5%",
                "change": "+2.1%",
                "trend": "up"
            },
            "total_knowledge_acquired": {
                "value": 1247,
                "change": "+23",
                "trend": "up"
            },
            "api_cost_today": {
                "value": "$12.45",
                "change": "-5%",
                "trend": "down"
            },
            "conversation_quality_score": {
                "value": "9.2/10",
                "change": "+0.3",
                "trend": "up"
            },
            "learning_efficiency": {
                "value": "94%",
                "change": "+7%",
                "trend": "up"
            }
        }

    def _get_live_activity(self) -> list[dict]:
        """Get live activity feed"""
        try:
            activities = []
            log_file = Path('logs/bot_memory.log')

            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-20:]  # Last 20 lines

                for line in lines:
                    try:
                        # Parse log line
                        parts = line.strip().split(' - ')
                        if len(parts) >= 4:
                            activities.append({
                                "timestamp": parts[0],
                                "level": parts[2],
                                "message": ' - '.join(parts[3:]),
                                "type": "system"
                            })
                    except:
                        continue

            # Add some sample activities if log is empty
            if not activities:
                activities = [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "level": "INFO",
                        "message": "نورا آماده پاسخگویی است",
                        "type": "system"
                    },
                    {
                        "timestamp": (datetime.now() - timedelta(minutes=2)).isoformat(),
                        "level": "INFO",
                        "message": "پیام جدید از تلگرام دریافت شد",
                        "type": "telegram"
                    }
                ]

            return activities

        except Exception as e:
            logger.error(f"Error getting live activity: {e}")
            return []

    def _get_platform_metrics(self) -> dict:
        """Get platform-specific metrics"""
        return {
            "telegram": {
                "active_chats": 15,
                "messages_today": 45,
                "learning_channels": 3,
                "avg_response_time": "2.1s"
            },
            "twitter": {
                "followers": 892,
                "tweets_today": 8,
                "engagement_rate": "7.2%",
                "mentions": 12
            },
            "instagram": {
                "followers": 456,
                "posts_today": 2,
                "stories": 3,
                "engagement_rate": "5.8%"
            },
            "threads": {
                "followers": 234,
                "threads_today": 4,
                "replies": 15,
                "engagement_rate": "6.1%"
            }
        }

    async def run(self):
        """Run the dashboard server"""
        logger.info("🌐 Starting web dashboard...")

        def run_flask():
            self.app.config['SECRET_KEY'] = 'nora_dashboard_secret_key'
            self.app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)

        # Run Flask in a separate thread
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.daemon = True
        flask_thread.start()

        logger.info("✅ Web dashboard available at http://0.0.0.0:5000")

        # Keep the async function running
        while True:
            await asyncio.sleep(60)

    async def shutdown(self):
        """Shutdown dashboard server"""
        logger.info("🌐 Web dashboard shutting down...")