
"""
perception_system.py - سیستم ادراک و یادگیری چند منبعی نورا
Advanced multi-source perception and learning system for Nora
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
from pathlib import Path
import re
import hashlib

logger = logging.getLogger(__name__)

class PerceptionSystem:
    """
    سیستم ادراک پیشرفته برای یادگیری از تمام منابع
    Advanced perception system for learning from all sources
    """
    
    def __init__(self, ai_core, memory_manager):
        self.ai_core = ai_core
        self.memory_manager = memory_manager
        self.learning_sources = {}
        self.content_patterns = {}
        self.knowledge_graph = {}
        
        # Learning configuration
        self.learning_config = self._load_learning_config()
        
    def _load_learning_config(self) -> Dict:
        """بارگذاری تنظیمات یادگیری"""
        try:
            with open('config/learning_sources.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_default_learning_config()
            
    def _create_default_learning_config(self) -> Dict:
        """ایجاد تنظیمات پیش‌فرض یادگیری"""
        config = {
            "telegram_learning_sources": {
                "tech_channels": [
                    {"id": "@TechCrunch", "category": "technology", "priority": "high"},
                    {"id": "@VentureBeat", "category": "startup", "priority": "high"},
                    {"id": "@MIT_CSAIL", "category": "ai_research", "priority": "medium"}
                ],
                "philosophy_channels": [
                    {"id": "@philosophy_daily", "category": "philosophy", "priority": "medium"},
                    {"id": "@ethics_ai", "category": "ai_ethics", "priority": "high"}
                ],
                "persian_tech": [
                    {"id": "@zoomit", "category": "persian_tech", "priority": "high"},
                    {"id": "@digikala_mag", "category": "persian_business", "priority": "medium"}
                ]
            },
            "learning_patterns": {
                "content_analysis_depth": "deep",
                "pattern_recognition": True,
                "style_mimicry": True,
                "trend_tracking": True,
                "sentiment_analysis": True
            },
            "knowledge_integration": {
                "auto_categorize": True,
                "cross_reference": True,
                "fact_verification": True,
                "bias_detection": True
            }
        }
        
        Path("config").mkdir(exist_ok=True)
        with open('config/learning_sources.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        return config
        
    async def analyze_content_deep(self, content: str, source_info: Dict) -> Dict:
        """تحلیل عمیق محتوا برای استخراج دانش"""
        
        analysis = {
            "content_hash": hashlib.md5(content.encode()).hexdigest(),
            "source": source_info,
            "timestamp": datetime.now().isoformat(),
            "raw_content": content,
            "processed_content": self._preprocess_content(content),
            "metadata": {}
        }
        
        try:
            # Extract key information using AI
            ai_analysis = await self.ai_core.think(f"""
            لطفاً این محتوا را تحلیل کن و خروجی را به صورت JSON ارائه دهید:
            
            محتوا: {content[:1000]}...
            
            تحلیل مورد نیاز:
            1. موضوع اصلی و موضوعات فرعی
            2. کلمات کلیدی مهم
            3. سبک نگارش و لحن
            4. اطلاعات قابل استخراج
            5. ارزش آموزشی (1-10)
            6. دسته‌بندی محتوا
            
            خروجی باید دقیقاً JSON باشد.
            """)
            
            # Parse AI analysis
            try:
                ai_data = json.loads(ai_analysis)
                analysis["ai_analysis"] = ai_data
            except:
                analysis["ai_analysis"] = {"error": "Could not parse AI analysis"}
                
            # Extract patterns
            analysis["patterns"] = self._extract_content_patterns(content, source_info)
            
            # Categorize content
            analysis["category"] = self._categorize_content(content, ai_analysis)
            
            # Calculate learning value
            analysis["learning_value"] = self._calculate_learning_value(analysis)
            
        except Exception as e:
            logger.error(f"Error in deep content analysis: {e}")
            analysis["error"] = str(e)
            
        return analysis
        
    def _preprocess_content(self, content: str) -> str:
        """پیش‌پردازش محتوا"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Extract URLs separately
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        
        # Clean content but keep structure
        cleaned = re.sub(r'http[s]?://\S+', '[URL]', content)
        
        return cleaned.strip()
        
    def _extract_content_patterns(self, content: str, source_info: Dict) -> Dict:
        """استخراج الگوهای محتوا"""
        patterns = {
            "hashtags": re.findall(r'#\w+', content),
            "mentions": re.findall(r'@\w+', content),
            "emojis": re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', content),
            "questions": len(re.findall(r'[؟?]', content)),
            "exclamations": len(re.findall(r'[!]', content)),
            "length": len(content),
            "word_count": len(content.split()),
            "persian_ratio": len(re.findall(r'[\u0600-\u06FF]', content)) / max(len(content), 1),
            "english_ratio": len(re.findall(r'[a-zA-Z]', content)) / max(len(content), 1)
        }
        
        # Add source-specific patterns
        source_type = source_info.get("platform", "unknown")
        patterns["source_patterns"] = self._get_source_specific_patterns(content, source_type)
        
        return patterns
        
    def _get_source_specific_patterns(self, content: str, source_type: str) -> Dict:
        """الگوهای خاص هر منبع"""
        if source_type == "telegram":
            return {
                "channel_structure": "forward" in content.lower(),
                "has_media_caption": "[Photo]" in content or "[Video]" in content,
                "thread_style": "//" in content or "►" in content
            }
        elif source_type == "twitter":
            return {
                "is_thread": content.count('\n') > 3,
                "has_retweet": "RT @" in content,
                "quote_tweet": "https://twitter.com" in content
            }
        
        return {}
        
    def _categorize_content(self, content: str, ai_analysis: str) -> str:
        """دسته‌بندی محتوا"""
        categories = {
            "technology": ["ai", "artificial intelligence", "هوش مصنوعی", "tech", "فناوری"],
            "business": ["startup", "business", "کسب‌وکار", "استارتاپ", "marketing"],
            "philosophy": ["philosophy", "فلسفه", "ethics", "اخلاق", "thinking"],
            "science": ["research", "science", "تحقیق", "علم", "study"],
            "social": ["society", "جامعه", "culture", "فرهنگ", "people"]
        }
        
        content_lower = content.lower()
        
        for category, keywords in categories.items():
            if any(keyword in content_lower for keyword in keywords):
                return category
                
        return "general"
        
    def _calculate_learning_value(self, analysis: Dict) -> float:
        """محاسبه ارزش یادگیری محتوا"""
        value = 0.5  # Base value
        
        # Content quality indicators
        if analysis.get("patterns", {}).get("word_count", 0) > 50:
            value += 0.1
        if analysis.get("patterns", {}).get("word_count", 0) > 200:
            value += 0.1
            
        # Source credibility
        source_priority = analysis.get("source", {}).get("priority", "low")
        if source_priority == "high":
            value += 0.2
        elif source_priority == "medium":
            value += 0.1
            
        # Category relevance
        if analysis.get("category") in ["technology", "business", "philosophy"]:
            value += 0.2
            
        # AI analysis quality
        ai_value = analysis.get("ai_analysis", {}).get("learning_value", 5)
        if isinstance(ai_value, (int, float)):
            value += (ai_value / 10.0) * 0.2
            
        return min(1.0, value)
        
    async def learn_from_telegram_sources(self):
        """یادگیری از منابع تلگرام"""
        telegram_sources = self.learning_config.get("telegram_learning_sources", {})
        
        for category, sources in telegram_sources.items():
            for source in sources:
                try:
                    await self._learn_from_telegram_source(source, category)
                except Exception as e:
                    logger.error(f"Error learning from {source}: {e}")
                    
    async def _learn_from_telegram_source(self, source: Dict, category: str):
        """یادگیری از یک منبع تلگرام"""
        # This would integrate with the Telegram client
        # For now, we'll create a placeholder
        logger.info(f"Learning from Telegram source: {source['id']} in category: {category}")
        
        # The actual implementation would:
        # 1. Connect to the Telegram source
        # 2. Fetch recent messages
        # 3. Analyze each message
        # 4. Store valuable insights
        # 5. Update content patterns
        
    async def integrate_knowledge(self, analysis: Dict):
        """یکپارچه‌سازی دانش جدید با دانش موجود"""
        if analysis.get("learning_value", 0) < 0.3:
            return  # Skip low-value content
            
        # Store in memory
        await self.memory_manager.store_knowledge({
            "content": analysis["processed_content"],
            "category": analysis.get("category", "general"),
            "source": analysis["source"],
            "patterns": analysis["patterns"],
            "learning_value": analysis["learning_value"],
            "ai_analysis": analysis.get("ai_analysis", {}),
            "timestamp": analysis["timestamp"]
        })
        
        # Update knowledge graph
        await self._update_knowledge_graph(analysis)
        
    async def _update_knowledge_graph(self, analysis: Dict):
        """به‌روزرسانی گراف دانش"""
        category = analysis.get("category", "general")
        
        if category not in self.knowledge_graph:
            self.knowledge_graph[category] = {
                "concepts": {},
                "connections": [],
                "patterns": {},
                "evolution": []
            }
            
        # Add concepts and connections
        ai_analysis = analysis.get("ai_analysis", {})
        if "keywords" in ai_analysis:
            for keyword in ai_analysis["keywords"]:
                if keyword not in self.knowledge_graph[category]["concepts"]:
                    self.knowledge_graph[category]["concepts"][keyword] = {
                        "frequency": 0,
                        "contexts": [],
                        "last_seen": None
                    }
                    
                self.knowledge_graph[category]["concepts"][keyword]["frequency"] += 1
                self.knowledge_graph[category]["concepts"][keyword]["last_seen"] = analysis["timestamp"]
                
    async def generate_learning_insights(self) -> Dict:
        """تولید بینش‌های یادگیری"""
        insights = {
            "learning_summary": {
                "total_sources_analyzed": len(self.learning_sources),
                "knowledge_categories": list(self.knowledge_graph.keys()),
                "top_learning_topics": self._get_top_learning_topics(),
                "learning_efficiency": self._calculate_learning_efficiency()
            },
            "content_patterns": self._analyze_content_patterns(),
            "knowledge_evolution": self._track_knowledge_evolution(),
            "recommendations": self._generate_learning_recommendations()
        }
        
        return insights
        
    def _get_top_learning_topics(self) -> List[Dict]:
        """دریافت موضوعات برتر یادگیری"""
        topics = []
        
        for category, data in self.knowledge_graph.items():
            concepts = data.get("concepts", {})
            for concept, info in concepts.items():
                topics.append({
                    "topic": concept,
                    "category": category,
                    "frequency": info["frequency"],
                    "last_seen": info["last_seen"]
                })
                
        return sorted(topics, key=lambda x: x["frequency"], reverse=True)[:10]
        
    def _calculate_learning_efficiency(self) -> float:
        """محاسبه کارایی یادگیری"""
        # Calculate based on various factors
        return 0.85  # Placeholder
        
    def _analyze_content_patterns(self) -> Dict:
        """تحلیل الگوهای محتوا"""
        return {
            "dominant_styles": ["analytical", "conversational"],
            "trending_formats": ["thread", "infographic"],
            "optimal_lengths": {"short": "50-150 words", "medium": "150-400 words"}
        }
        
    def _track_knowledge_evolution(self) -> List[Dict]:
        """ردیابی تکامل دانش"""
        return [
            {
                "period": "last_week",
                "new_concepts": 15,
                "evolved_understanding": 8,
                "deprecated_info": 2
            }
        ]
        
    def _generate_learning_recommendations(self) -> List[str]:
        """تولید توصیه‌های یادگیری"""
        return [
            "Focus more on AI ethics discussions",
            "Explore startup ecosystem in Iran",
            "Deepen understanding of blockchain applications"
        ]
        
    async def run(self):
        """حلقه اصلی سیستم ادراک"""
        logger.info("🔍 Perception system is now active")
        
        while True:
            try:
                # Learn from various sources
                await self.learn_from_telegram_sources()
                
                # Generate insights
                insights = await self.generate_learning_insights()
                
                # Save insights
                with open('data/learning_insights.json', 'w', encoding='utf-8') as f:
                    json.dump(insights, f, ensure_ascii=False, indent=2)
                    
                # Sleep for 30 minutes
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Error in perception system: {e}")
                await asyncio.sleep(300)
