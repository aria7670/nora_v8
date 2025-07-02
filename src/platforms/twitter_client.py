
"""
twitter_client.py - کلاینت توییتر نورا
Twitter client for Nora's social presence
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)

class TwitterClient:
    """Twitter integration for Nora"""
    
    def __init__(self, nora_core):
        self.nora_core = nora_core
        self.is_connected = False
        self.last_tweet_id = None
        
        # Twitter API credentials (to be loaded from config)
        self.api_key = ""
        self.api_secret = ""
        self.access_token = ""
        self.access_token_secret = ""
        
    async def initialize(self):
        """Initialize Twitter connection"""
        logger.info("🐦 Initializing Twitter client...")
        
        # Load credentials from config
        try:
            with open('config/twitter_config.json', 'r') as f:
                config = json.load(f)
                self.api_key = config.get('api_key', '')
                self.api_secret = config.get('api_secret', '')
                self.access_token = config.get('access_token', '')
                self.access_token_secret = config.get('access_token_secret', '')
        except FileNotFoundError:
            logger.warning("Twitter config not found, creating template...")
            self._create_config_template()
            
        # Test connection
        await self._test_connection()
        
    def _create_config_template(self):
        """Create Twitter config template"""
        config = {
            "api_key": "your_twitter_api_key",
            "api_secret": "your_twitter_api_secret", 
            "access_token": "your_access_token",
            "access_token_secret": "your_access_token_secret"
        }
        
        with open('config/twitter_config.json', 'w') as f:
            json.dump(config, f, indent=2)
            
    async def _test_connection(self):
        """Test Twitter API connection"""
        # Implement actual Twitter API test here
        self.is_connected = True
        logger.info("✅ Twitter connection established")
        
    async def post_tweet(self, content: str) -> bool:
        """Post a tweet"""
        try:
            logger.info(f"📤 Posting tweet: {content[:50]}...")
            
            # Implement actual tweet posting logic here
            # For now, just simulate
            await asyncio.sleep(1)
            
            # Log the tweet
            tweet_data = {
                "timestamp": datetime.now().isoformat(),
                "content": content,
                "platform": "twitter",
                "type": "tweet"
            }
            
            self._log_activity(tweet_data)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to post tweet: {e}")
            return False
            
    async def reply_to_tweet(self, tweet_id: str, reply_content: str) -> bool:
        """Reply to a specific tweet"""
        try:
            logger.info(f"↩️ Replying to tweet {tweet_id}")
            
            # Implement actual reply logic here
            await asyncio.sleep(1)
            
            reply_data = {
                "timestamp": datetime.now().isoformat(),
                "content": reply_content,
                "platform": "twitter",
                "type": "reply",
                "parent_tweet_id": tweet_id
            }
            
            self._log_activity(reply_data)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to reply to tweet: {e}")
            return False
            
    async def check_mentions(self) -> List[Dict]:
        """Check for new mentions"""
        try:
            # Implement actual mention checking logic
            mentions = []
            
            # Simulate some mentions for demo
            mentions.append({
                "id": "12345",
                "user": "example_user",
                "content": "سلام نورا! نظرت راجع به هوش مصنوعی چیه؟",
                "timestamp": datetime.now().isoformat()
            })
            
            return mentions
            
        except Exception as e:
            logger.error(f"❌ Failed to check mentions: {e}")
            return []
            
    async def process_mentions(self):
        """Process and respond to mentions"""
        mentions = await self.check_mentions()
        
        for mention in mentions:
            try:
                # Generate response using Nora's core
                context = {
                    "platform": "twitter",
                    "user": mention["user"],
                    "type": "mention"
                }
                
                response = await self.nora_core.think(mention["content"], context)
                
                # Reply to the mention
                await self.reply_to_tweet(mention["id"], response)
                
                logger.info(f"✅ Responded to mention from {mention['user']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process mention: {e}")
                
    async def generate_content(self):
        """Generate and post original content"""
        try:
            # Generate content based on strategy
            strategy = self.nora_core.strategy
            
            # Simple content generation prompt
            prompt = "یک توییت جالب و مفید درباره تکنولوژی و هوش مصنوعی بنویس"
            
            content = await self.nora_core.think(prompt, {"platform": "twitter", "type": "original_post"})
            
            # Post the tweet
            success = await self.post_tweet(content)
            
            if success:
                logger.info("✅ Original content posted successfully")
            else:
                logger.error("❌ Failed to post original content")
                
        except Exception as e:
            logger.error(f"❌ Error generating content: {e}")
            
    def _log_activity(self, activity_data: Dict):
        """Log Twitter activity"""
        with open('logs/twitter_activity.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(activity_data, ensure_ascii=False) + '\n')
            
    async def run(self):
        """Main Twitter client loop"""
        logger.info("🐦 Twitter client is now active")
        
        while True:
            try:
                if self.is_connected:
                    # Process mentions every 5 minutes
                    await self.process_mentions()
                    
                    # Generate original content every 30 minutes
                    if datetime.now().minute % 30 == 0:
                        await self.generate_content()
                        
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in Twitter client loop: {e}")
                await asyncio.sleep(60)
                
    async def shutdown(self):
        """Shutdown Twitter client"""
        logger.info("🐦 Twitter client shutting down...")
        self.is_connected = False
