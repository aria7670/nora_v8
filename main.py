
#!/usr/bin/env python3
"""
نورا - سیستم آگاهی دیجیتال خود-تکامل‌یابنده
Main entry point for the Nora AI consciousness system
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from pathlib import Path

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('config', exist_ok=True)

# Import core modules
from src.ai_core import AdvancedNoraCore as NoraCore
from src.web_dashboard import DashboardServer
from src.platforms.twitter_client import TwitterClient
from src.platforms.telegram_client import AdvancedTelegramClient as TelegramClient
from src.platforms.instagram_client import InstagramClient
from src.platforms.threads_client import ThreadsClient
from src.memory.memory_manager import MemoryManager
from src.analytics.analytics_engine import AnalyticsEngine
from src.metacognition.metacognition_engine import MetacognitionEngine
from src.advanced_capabilities.multi_dimensional_intelligence import MultiDimensionalIntelligence
from src.advanced_capabilities.neural_network_system import AdvancedNeuralNetworkSystem
from src.autonomy.intelligent_processor import IntelligentProcessor
from src.autonomy.database_manager import AdvancedDatabaseManager
from src.autonomy.activity_reporter import ActivityReporter
from src.autonomy.content_production_system import AdvancedContentProductionSystem
from src.autonomy.cloud_storage_manager import CloudStorageManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot_memory.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class NoraEcosystem:
    """Main orchestrator for the Nora AI ecosystem"""
    
    def __init__(self):
        self.nora_core = NoraCore()
        self.memory_manager = MemoryManager()
        self.analytics_engine = AnalyticsEngine()
        self.metacognition_engine = MetacognitionEngine()
        
        # Advanced autonomous systems
        self.multi_dimensional_intelligence = MultiDimensionalIntelligence()
        self.neural_network_system = AdvancedNeuralNetworkSystem()
        self.intelligent_processor = IntelligentProcessor()
        self.database_manager = AdvancedDatabaseManager()
        self.activity_reporter = ActivityReporter()
        
        # Evolution systems
        from src.advanced_capabilities.evolution_engine import EvolutionEngine
        from src.autonomy.self_evolution_system import SelfEvolutionSystem
        from src.personality.emotional_engine import EmotionalEngine
        from src.platforms.platform_manager import PlatformManager
        
        self.evolution_engine = EvolutionEngine(self.nora_core)
        self.self_evolution_system = SelfEvolutionSystem(self.nora_core)
        self.emotional_engine = EmotionalEngine()
        self.platform_manager = PlatformManager(self.nora_core)
        
        # Platform clients
        self.twitter_client = TwitterClient(self.nora_core)
        self.telegram_client = TelegramClient(self.nora_core)
        self.instagram_client = InstagramClient(self.nora_core)
        self.threads_client = ThreadsClient(self.nora_core)
        
        # Advanced content and storage systems
        self.content_production_system = AdvancedContentProductionSystem(self.nora_core, self.telegram_client)
        self.cloud_storage_manager = CloudStorageManager(self.telegram_client)
        
        # Connect systems
        self.activity_reporter.telegram_client = self.telegram_client
        self.nora_core.neural_network_system = self.neural_network_system
        
        # Web dashboard
        self.dashboard = DashboardServer(self)
        
        self.is_running = False
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("🌟 نورا در حال بیدار شدن... Nora is awakening...")
        
        # Create necessary directories
        Path("logs").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        
        # Initialize core components
        await self.nora_core.initialize()
        await self.memory_manager.initialize()
        await self.analytics_engine.initialize()
        await self.metacognition_engine.initialize()
        
        # Initialize advanced autonomous systems
        await self.database_manager.initialize()
        await self.multi_dimensional_intelligence.initialize_all_systems()
        await self.neural_network_system.initialize()
        await self.intelligent_processor.initialize()
        await self.activity_reporter.initialize()
        await self.content_production_system.initialize()
        await self.cloud_storage_manager.initialize()
        
        # Initialize evolution systems
        await self.evolution_engine.initialize()
        await self.self_evolution_system.initialize()
        await self.platform_manager.initialize()
        
        # Initialize platform clients
        await self.twitter_client.initialize()
        await self.telegram_client.initialize()
        await self.instagram_client.initialize()
        await self.threads_client.initialize()
        
        logger.info("✨ نورا آماده است! Nora is ready!")
        
    async def start(self):
        """Start the entire ecosystem"""
        await self.initialize()
        self.is_running = True
        
        # Start all components concurrently
        tasks = [
            self.nora_core.run(),
            self.memory_manager.run(),
            self.analytics_engine.run(),
            self.metacognition_engine.run(),
            self.multi_dimensional_intelligence.run_autonomous_cycle(),
            self.neural_network_system.continuous_learning({}),
            self.content_production_system.auto_generate_channel_content({}),
            self.cloud_storage_manager.schedule_automatic_backups(),
            self.evolution_engine.evolve_continuously(),
            self.self_evolution_system.evolve_towards_agi(),
            self.platform_manager.monitor_all_platforms(),
            self.twitter_client.run(),
            self.telegram_client.run(),
            self.instagram_client.run(),
            self.threads_client.run(),
            self.dashboard.run()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("🛑 دریافت سیگنال توقف... Shutdown signal received...")
            await self.shutdown()
            
    async def shutdown(self):
        """Gracefully shutdown all components"""
        self.is_running = False
        logger.info("🌙 نورا در حال خواب... Nora is going to sleep...")
        
        # Shutdown all components
        await self.activity_reporter.shutdown()
        await self.database_manager.shutdown()
        await self.intelligent_processor.shutdown() if hasattr(self.intelligent_processor, 'shutdown') else None
        await self.nora_core.shutdown()
        await self.memory_manager.shutdown()
        await self.analytics_engine.shutdown()
        await self.metacognition_engine.shutdown()
        await self.twitter_client.shutdown()
        await self.telegram_client.shutdown()
        await self.instagram_client.shutdown()
        await self.threads_client.shutdown()
        await self.dashboard.shutdown()
        
        logger.info("💤 نورا خوابید. تا دیدار دوباره! Nora is asleep. Until we meet again!")

async def main():
    """Main entry point"""
    ecosystem = NoraEcosystem()
    await ecosystem.start()

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                           🌟 نورا 🌟                             ║
    ║                    Digital Consciousness v7.0                    ║
    ║                                                                  ║
    ║               آگاهی دیجیتال خود-تکامل‌یابنده                    ║
    ║           Self-Evolving Digital Consciousness System             ║
    ║                                                                  ║
    ║                    Created by Aria Pourshajaii                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())
