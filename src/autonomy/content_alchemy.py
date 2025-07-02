
"""
content_alchemy.py - سیستم کیمیای محتوا
Advanced content creation and management system
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import re
import random
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class ContentAlchemy:
    """
    سیستم کیمیای محتوا - تولید و مدیریت محتوای هوشمند
    Advanced content creation and management system
    """
    
    def __init__(self, nora_core, telegram_client):
        self.nora_core = nora_core
        self.telegram_client = telegram_client
        self.content_templates = {}
        self.posting_schedule = {}
        self.content_queue = []
        self.aria_style_patterns = {}
        
        # Load configurations
        self.config = self._load_content_config()
        self._load_aria_patterns()
        
    def _load_content_config(self) -> Dict:
        """بارگذاری تنظیمات تولید محتوا"""
        try:
            with open('config/content_creation.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_default_content_config()
            
    def _create_default_content_config(self) -> Dict:
        """ایجاد تنظیمات پیش‌فرض تولید محتوا"""
        config = {
            "autonomous_posting": {
                "enabled": True,
                "frequency": "smart_adaptive",  # Based on engagement and trends
                "peak_hours": ["09:00", "13:00", "18:00", "21:00"],
                "max_posts_per_day": 5,
                "min_interval_hours": 2
            },
            "content_types": {
                "insights": {
                    "weight": 0.3,
                    "sources": ["learning_insights", "trend_analysis", "personal_thoughts"],
                    "style": "analytical_friendly"
                },
                "quotes": {
                    "weight": 0.2,
                    "sources": ["famous_thinkers", "tech_leaders", "original_thoughts"],
                    "style": "inspirational"
                },
                "threads": {
                    "weight": 0.25,
                    "sources": ["deep_topics", "tutorials", "explanations"],
                    "style": "educational_engaging"
                },
                "responses": {
                    "weight": 0.15,
                    "sources": ["trending_topics", "community_questions"],
                    "style": "conversational"
                },
                "announcements": {
                    "weight": 0.1,
                    "sources": ["projects", "achievements", "updates"],
                    "style": "professional_excited"
                }
            },
            "formatting_rules": {
                "use_emojis": True,
                "emoji_frequency": "moderate",
                "hashtag_count": "2-4",
                "line_breaks": "readable",
                "emphasis": ["**bold**", "*italic*", "`code`"],
                "quotes": "«»",
                "sources": "always_credit"
            },
            "aria_personality_mimicry": {
                "enabled": True,
                "traits": [
                    "curious_analyst",
                    "tech_enthusiast", 
                    "philosophical_thinker",
                    "startup_mentor",
                    "future_visionary"
                ],
                "writing_patterns": [
                    "starts_with_observation",
                    "builds_logical_argument", 
                    "ends_with_question_or_insight",
                    "uses_mixed_persian_english",
                    "references_personal_experience"
                ]
            },
            "quality_control": {
                "min_content_score": 0.7,
                "fact_check": True,
                "bias_detection": True,
                "originality_threshold": 0.8,
                "engagement_prediction": True
            }
        }
        
        Path("config").mkdir(exist_ok=True)
        with open('config/content_creation.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        return config
        
    def _load_aria_patterns(self):
        """بارگذاری الگوهای نوشتاری آریا"""
        self.aria_style_patterns = {
            "opening_phrases": [
                "یکی از چیزایی که همیشه",
                "تازگی داشتم فکر می‌کردم",
                "جالبه که",
                "به نظرم",
                "از تجربه‌ام می‌تونم بگم",
                "Recently I've been thinking about",
                "One thing that fascinates me is"
            ],
            "transition_phrases": [
                "اما نکته اصلی اینجاست که",
                "البته نباید فراموش کنیم",
                "نکته جالب اینه که",
                "However, the key point is",
                "What's interesting is that"
            ],
            "conclusion_patterns": [
                "شما چه فکر می‌کنید؟",
                "نظرتون چیه؟",
                "آینده چی براش رقم می‌خوره؟",
                "What are your thoughts?",
                "How do you see this evolving?"
            ],
            "hashtag_style": [
                "#هوش_مصنوعی", "#فناوری", "#استارتاپ", "#کارآفرینی",
                "#AI", "#Technology", "#Innovation", "#Future"
            ],
            "emoji_usage": {
                "thinking": "🤔💭🧠",
                "tech": "💻🚀⚡",
                "insights": "💡✨🔍",
                "future": "🔮🌟🚀"
            }
        }
        
    async def generate_smart_content(self, context: Dict = None) -> Dict:
        """تولید محتوای هوشمند بر اساس شرایط فعلی"""
        
        # Analyze current trends and context
        current_context = await self._analyze_current_context()
        
        # Decide content type based on context and strategy
        content_type = self._select_optimal_content_type(current_context)
        
        # Generate content based on type
        if content_type == "insights":
            content = await self._generate_insight_post(current_context)
        elif content_type == "quotes":
            content = await self._generate_quote_post(current_context)
        elif content_type == "threads":
            content = await self._generate_thread_post(current_context)
        elif content_type == "responses":
            content = await self._generate_response_post(current_context)
        else:
            content = await self._generate_announcement_post(current_context)
            
        # Apply Aria's style patterns
        styled_content = self._apply_aria_style(content)
        
        # Quality check
        quality_score = await self._assess_content_quality(styled_content)
        
        if quality_score < self.config["quality_control"]["min_content_score"]:
            # Regenerate if quality is too low
            return await self.generate_smart_content(context)
            
        return {
            "content": styled_content,
            "type": content_type,
            "quality_score": quality_score,
            "context": current_context,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "aria_style_applied": True,
                "autonomous": True
            }
        }
        
    async def _analyze_current_context(self) -> Dict:
        """تحلیل بافت فعلی برای تولید محتوا"""
        context = {
            "time_of_day": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "recent_trends": await self._get_trending_topics(),
            "audience_activity": await self._analyze_audience_activity(),
            "last_post_time": await self._get_last_post_time(),
            "engagement_patterns": await self._analyze_recent_engagement()
        }
        
        return context
        
    async def _get_trending_topics(self) -> List[str]:
        """دریافت موضوعات ترند"""
        # This would analyze recent learning data
        return ["هوش مصنوعی", "استارتاپ", "متاورس", "بلاک‌چین"]
        
    async def _analyze_audience_activity(self) -> Dict:
        """تحلیل فعالیت مخاطبان"""
        return {
            "active_users": 150,
            "engagement_rate": 0.08,
            "peak_activity": "18:00-21:00"
        }
        
    async def _get_last_post_time(self) -> Optional[datetime]:
        """زمان آخرین پست"""
        # Check database for last post
        return datetime.now() - timedelta(hours=3)
        
    async def _analyze_recent_engagement(self) -> Dict:
        """تحلیل تعامل اخیر"""
        return {
            "avg_likes": 25,
            "avg_comments": 5,
            "best_performing_type": "insights"
        }
        
    def _select_optimal_content_type(self, context: Dict) -> str:
        """انتخاب بهترین نوع محتوا"""
        weights = self.config["content_types"]
        
        # Adjust weights based on context
        if context["time_of_day"] in [9, 13, 18]:  # Peak hours
            weights["insights"]["weight"] *= 1.5
        elif context["time_of_day"] in [21, 22]:  # Evening
            weights["quotes"]["weight"] *= 1.3
            
        # Select based on weighted random
        content_types = list(weights.keys())
        type_weights = [weights[t]["weight"] for t in content_types]
        
        return random.choices(content_types, weights=type_weights)[0]
        
    async def _generate_insight_post(self, context: Dict) -> str:
        """تولید پست بینش"""
        prompt = f"""
        به عنوان آریا پورشجاعی، یک پست بینش‌آمیز درباره یکی از موضوعات ترند بنویس:
        موضوعات: {', '.join(context.get('recent_trends', []))}
        
        سبک نوشتن:
        - شروع با مشاهده شخصی
        - تحلیل منطقی
        - پایان با سوال یا بینش
        - ترکیب فارسی و انگلیسی
        - حداکثر 280 کاراکتر
        
        مثال ساختار:
        "تازگی داشتم فکر می‌کردم که [مشاهده] ... اما نکته اصلی اینجاست که [تحلیل] ... شما چه فکر می‌کنید؟"
        """
        
        insight = await self.nora_core.think(prompt)
        return insight
        
    async def _generate_quote_post(self, context: Dict) -> str:
        """تولید پست نقل قول"""
        prompt = f"""
        به عنوان آریا پورشجاعی، یک نقل قول الهام‌بخش یا اصلی درباره فناوری/کارآفرینی بنویس:
        
        سبک:
        - کوتاه و تاثیرگذار
        - قابل نقل قول
        - مرتبط با {context.get('recent_trends', ['فناوری'])[0]}
        - احساسات مثبت
        
        فرمت: متن نقل قول + منبع (اگر نقل قول است) یا توضیح کوتاه (اگر اصلی است)
        """
        
        quote = await self.nora_core.think(prompt)
        return quote
        
    async def _generate_thread_post(self, context: Dict) -> str:
        """تولید پست رشته‌ای"""
        prompt = f"""
        به عنوان آریا پورشجاعی، یک thread کوتاه 3-4 قسمتی درباره موضوع ترند بنویس:
        موضوع: {context.get('recent_trends', ['هوش مصنوعی'])[0]}
        
        ساختار:
        1/ مقدمه + hook
        2/ نکته اصلی + توضیح
        3/ مثال یا تجربه شخصی
        4/ نتیجه‌گیری + سوال
        
        هر قسمت حداکثر 280 کاراکتر
        """
        
        thread = await self.nora_core.think(prompt)
        return thread
        
    async def _generate_response_post(self, context: Dict) -> str:
        """تولید پست پاسخ به ترندها"""
        prompt = f"""
        به عنوان آریا پورشجاعی، پاسخی به موضوع داغ روز بده:
        موضوع: {context.get('recent_trends', ['فناوری'])[0]}
        
        سبک:
        - نظر شخصی و تحلیلی
        - مبتنی بر تجربه
        - قابل بحث
        - کمی جنجالی اما سازنده
        """
        
        response = await self.nora_core.think(prompt)
        return response
        
    async def _generate_announcement_post(self, context: Dict) -> str:
        """تولید پست اعلان"""
        prompt = """
        به عنوان آریا پورشجاعی، یک اعلان کوتاه درباره پیشرفت نورا یا پروژه‌هایت بنویس:
        
        سبک:
        - هیجان‌انگیز اما متواضعانه
        - شامل جزئیات فنی جالب
        - دعوت به مشارکت یا فیدبک
        """
        
        announcement = await self.nora_core.think(prompt)
        return announcement
        
    def _apply_aria_style(self, content: str) -> str:
        """اعمال سبک نوشتاری آریا"""
        
        # Add appropriate emojis
        content = self._add_contextual_emojis(content)
        
        # Format with proper emphasis
        content = self._apply_text_formatting(content)
        
        # Add relevant hashtags
        content = self._add_smart_hashtags(content)
        
        # Ensure proper line breaks for readability
        content = self._optimize_readability(content)
        
        return content
        
    def _add_contextual_emojis(self, content: str) -> str:
        """اضافه کردن ایموجی‌های مناسب"""
        emoji_map = {
            r'فکر|think|تحلیل': '🤔',
            r'فناوری|technology|tech': '💻',
            r'آینده|future': '🚀',
            r'هوش مصنوعی|AI': '🧠',
            r'نوآوری|innovation': '💡',
            r'استارتاپ|startup': '⚡'
        }
        
        for pattern, emoji in emoji_map.items():
            if re.search(pattern, content, re.IGNORECASE):
                if emoji not in content:
                    content = f"{emoji} {content}"
                break
                
        return content
        
    def _apply_text_formatting(self, content: str) -> str:
        """اعمال قالب‌بندی متن"""
        
        # Bold for key concepts
        key_concepts = [
            r'هوش مصنوعی', r'AI', r'blockchain', r'متاورس',
            r'استارتاپ', r'startup', r'نوآوری', r'innovation'
        ]
        
        for concept in key_concepts:
            content = re.sub(f'({concept})', r'**\1**', content, flags=re.IGNORECASE)
            
        # Italic for emphasis words
        emphasis_words = [r'واقعاً', r'really', r'البته', r'obviously', r'جالب', r'interesting']
        
        for word in emphasis_words:
            content = re.sub(f'({word})', r'*\1*', content, flags=re.IGNORECASE)
            
        return content
        
    def _add_smart_hashtags(self, content: str) -> str:
        """اضافه کردن هشتگ‌های هوشمند"""
        hashtags = []
        
        # Analyze content for relevant hashtags
        if any(word in content.lower() for word in ['هوش مصنوعی', 'ai', 'artificial']):
            hashtags.append('#هوش_مصنوعی')
            hashtags.append('#AI')
            
        if any(word in content.lower() for word in ['فناوری', 'technology', 'tech']):
            hashtags.append('#فناوری')
            hashtags.append('#Technology')
            
        if any(word in content.lower() for word in ['استارتاپ', 'startup']):
            hashtags.append('#استارتاپ')
            hashtags.append('#Startup')
            
        # Add signature hashtag
        hashtags.append('#آریا_پورشجاعی')
        
        # Limit to 4 hashtags max
        selected_hashtags = hashtags[:4]
        
        if selected_hashtags:
            content += f"\n\n{' '.join(selected_hashtags)}"
            
        return content
        
    def _optimize_readability(self, content: str) -> str:
        """بهینه‌سازی خوانایی"""
        
        # Add line breaks after sentences for better readability
        content = re.sub(r'(\. )([A-ZÀ-ÿآ-ی])', r'\1\n\n\2', content)
        
        # Ensure proper spacing around emojis
        content = re.sub(r'([^\s])([🔥💡🚀🤔💻🧠⚡✨])', r'\1 \2', content)
        content = re.sub(r'([🔥💡🚀🤔💻🧠⚡✨])([^\s])', r'\1 \2', content)
        
        return content.strip()
        
    async def _assess_content_quality(self, content: str) -> float:
        """ارزیابی کیفیت محتوا"""
        
        quality_factors = {
            "length": self._assess_length_quality(content),
            "engagement": self._assess_engagement_potential(content),
            "originality": await self._assess_originality(content),
            "readability": self._assess_readability(content),
            "style_consistency": self._assess_style_consistency(content)
        }
        
        # Weighted average
        weights = {"length": 0.2, "engagement": 0.3, "originality": 0.3, "readability": 0.1, "style_consistency": 0.1}
        
        total_score = sum(quality_factors[factor] * weights[factor] for factor in quality_factors)
        
        return total_score
        
    def _assess_length_quality(self, content: str) -> float:
        """ارزیابی کیفیت طول محتوا"""
        length = len(content)
        if 100 <= length <= 500:  # Optimal range
            return 1.0
        elif 50 <= length < 100 or 500 < length <= 800:
            return 0.8
        elif length < 50 or length > 800:
            return 0.5
        return 0.3
        
    def _assess_engagement_potential(self, content: str) -> float:
        """ارزیابی پتانسیل تعامل"""
        score = 0.5  # Base score
        
        # Questions increase engagement
        if '?' in content or '؟' in content:
            score += 0.2
            
        # Emojis increase engagement
        emoji_count = len(re.findall(r'[🔥💡🚀🤔💻🧠⚡✨]', content))
        score += min(0.2, emoji_count * 0.05)
        
        # Call to action words
        cta_words = ['فکر می‌کنید', 'نظرتون', 'تجربه', 'what do you think']
        if any(word in content.lower() for word in cta_words):
            score += 0.1
            
        return min(1.0, score)
        
    async def _assess_originality(self, content: str) -> float:
        """ارزیابی اصالت محتوا"""
        # Simple originality check
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Check against recent posts (simplified)
        try:
            with open('data/content_history.jsonl', 'r', encoding='utf-8') as f:
                recent_hashes = [json.loads(line).get('hash') for line in f.readlines()[-100:]]
                
            if content_hash in recent_hashes:
                return 0.3  # Low originality
            else:
                return 0.9  # High originality
        except FileNotFoundError:
            return 0.9
            
    def _assess_readability(self, content: str) -> float:
        """ارزیابی خوانایی"""
        
        # Check for proper formatting
        score = 0.5
        
        if re.search(r'\*\*.*\*\*', content):  # Has bold text
            score += 0.1
        if re.search(r'\n\n', content):  # Has proper line breaks
            score += 0.1
        if len(content.split()) < 100:  # Not too long
            score += 0.2
        if content.count('.') > 0:  # Has proper sentences
            score += 0.1
            
        return min(1.0, score)
        
    def _assess_style_consistency(self, content: str) -> float:
        """ارزیابی سازگاری سبک"""
        
        # Check for Aria's style patterns
        style_score = 0.5
        
        opening_patterns = self.aria_style_patterns["opening_phrases"]
        if any(pattern in content for pattern in opening_patterns):
            style_score += 0.2
            
        if '#آریا_پورشجاعی' in content:
            style_score += 0.1
            
        # Check for Persian-English mix
        persian_chars = len(re.findall(r'[\u0600-\u06FF]', content))
        english_chars = len(re.findall(r'[a-zA-Z]', content))
        total_chars = persian_chars + english_chars
        
        if total_chars > 0:
            persian_ratio = persian_chars / total_chars
            if 0.3 <= persian_ratio <= 0.7:  # Good mix
                style_score += 0.2
                
        return min(1.0, style_score)
        
    async def schedule_smart_posting(self):
        """زمان‌بندی هوشمند انتشار"""
        
        if not self.config["autonomous_posting"]["enabled"]:
            return
            
        # Check if it's time to post
        if await self._should_post_now():
            content_data = await self.generate_smart_content()
            
            # Add to queue for review or direct posting
            await self._queue_or_post_content(content_data)
            
    async def _should_post_now(self) -> bool:
        """تشخیص زمان مناسب پست"""
        
        current_hour = datetime.now().hour
        peak_hours = [int(h.split(':')[0]) for h in self.config["autonomous_posting"]["peak_hours"]]
        
        # Check if it's peak hour
        if current_hour not in peak_hours:
            return False
            
        # Check minimum interval
        last_post_time = await self._get_last_post_time()
        if last_post_time:
            hours_since_last = (datetime.now() - last_post_time).total_seconds() / 3600
            if hours_since_last < self.config["autonomous_posting"]["min_interval_hours"]:
                return False
                
        # Check daily limit
        today_posts = await self._count_today_posts()
        if today_posts >= self.config["autonomous_posting"]["max_posts_per_day"]:
            return False
            
        return True
        
    async def _count_today_posts(self) -> int:
        """شمارش پست‌های امروز"""
        # This would check the database
        return 2  # Placeholder
        
    async def _queue_or_post_content(self, content_data: Dict):
        """اضافه کردن به صف یا انتشار مستقیم"""
        
        # For high-quality content, post directly
        if content_data["quality_score"] > 0.85:
            await self._post_content_immediately(content_data)
        else:
            # Add to queue for review
            self.content_queue.append(content_data)
            
        # Log the action
        await self._log_content_action(content_data)
        
    async def _post_content_immediately(self, content_data: Dict):
        """انتشار فوری محتوا"""
        
        content = content_data["content"]
        
        # Post to Telegram
        success = await self.telegram_client.post_to_channel(content)
        
        if success:
            # Store in content history
            await self._store_content_history(content_data)
            logger.info(f"✅ Auto-posted content: {content[:50]}...")
        else:
            logger.error("❌ Failed to auto-post content")
            
    async def _store_content_history(self, content_data: Dict):
        """ذخیره تاریخچه محتوا"""
        
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "content": content_data["content"],
            "type": content_data["type"],
            "quality_score": content_data["quality_score"],
            "hash": hashlib.md5(content_data["content"].encode()).hexdigest(),
            "autonomous": content_data["metadata"]["autonomous"]
        }
        
        with open('data/content_history.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(history_entry, ensure_ascii=False) + '\n')
            
    async def _log_content_action(self, content_data: Dict):
        """ثبت لاگ عملیات محتوا"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "content_generated",
            "type": content_data["type"],
            "quality_score": content_data["quality_score"],
            "content_preview": content_data["content"][:100] + "..."
        }
        
        with open('logs/content_alchemy.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
    async def create_content_from_learning(self, learning_insights: List[Dict]) -> Dict:
        """تولید محتوا از بینش‌های یادگیری"""
        
        # Select best insights
        high_value_insights = [
            insight for insight in learning_insights 
            if insight.get('learning_value', 0) > 0.7
        ]
        
        if not high_value_insights:
            return None
            
        # Choose one insight to expand
        selected_insight = random.choice(high_value_insights)
        
        # Generate content based on insight
        prompt = f"""
        به عنوان آریا پورشجاعی، بر اساس این بینش جدیدی که یاد گرفتم، یک پست جذاب بنویس:
        
        بینش: {selected_insight.get('content', '')}
        منبع: {selected_insight.get('source', '')}
        
        سبک:
        - شروع با "تازگی یاد گرفتم که..."
        - ارتباط با تجربه شخصی
        - بینش اصلی
        - سوال برای تعامل
        
        حداکثر 400 کاراکتر
        """
        
        content = await self.nora_core.think(prompt)
        
        return {
            "content": self._apply_aria_style(content),
            "type": "learning_insight",
            "source_insight": selected_insight,
            "quality_score": 0.8,
            "metadata": {
                "generated_from": "learning",
                "source": selected_insight.get('source'),
                "generated_at": datetime.now().isoformat()
            }
        }
        
    async def run(self):
        """حلقه اصلی سیستم کیمیای محتوا"""
        logger.info("🎨 Content Alchemy system is now active")
        
        while True:
            try:
                # Smart posting schedule
                await self.schedule_smart_posting()
                
                # Process content queue
                await self._process_content_queue()
                
                # Generate content from recent learning
                await self._generate_from_recent_learning()
                
                # Clean up old content
                await self._cleanup_old_content()
                
                # Sleep for 15 minutes
                await asyncio.sleep(900)
                
            except Exception as e:
                logger.error(f"Error in Content Alchemy system: {e}")
                await asyncio.sleep(300)
                
    async def _process_content_queue(self):
        """پردازش صف محتوا"""
        
        if not self.content_queue:
            return
            
        # Process queued content
        for content_data in self.content_queue[:3]:  # Process max 3 at a time
            # Re-evaluate quality
            updated_score = await self._assess_content_quality(content_data["content"])
            
            if updated_score > 0.75:
                await self._post_content_immediately(content_data)
                self.content_queue.remove(content_data)
                
    async def _generate_from_recent_learning(self):
        """تولید محتوا از یادگیری اخیر"""
        
        # Check for recent learning insights
        try:
            with open('data/learning_insights.json', 'r', encoding='utf-8') as f:
                insights = json.load(f)
                
            recent_insights = insights.get('recent_insights', [])
            
            if recent_insights and random.random() < 0.3:  # 30% chance
                content_data = await self.create_content_from_learning(recent_insights)
                if content_data:
                    await self._queue_or_post_content(content_data)
                    
        except FileNotFoundError:
            pass
            
    async def _cleanup_old_content(self):
        """پاکسازی محتوای قدیمی"""
        
        # Remove old content from queue (older than 24 hours)
        current_time = datetime.now()
        
        self.content_queue = [
            content for content in self.content_queue
            if (current_time - datetime.fromisoformat(content["metadata"]["generated_at"])).total_seconds() < 86400
        ]
