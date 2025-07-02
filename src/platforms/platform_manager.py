
"""
platform_manager.py - مدیر یکپارچه تمام پلتفرم‌ها
Unified platform manager for all social media platforms
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)

class PlatformManager:
    """مدیر یکپارچه برای کنترل تمام پلتفرم‌ها"""
    
    def __init__(self, nora_core):
        self.nora_core = nora_core
        self.platforms = {}
        self.unified_analytics = {}
        self.cross_platform_strategies = {}
        self.content_sync_manager = ContentSyncManager()
        
    async def initialize(self):
        """Initialize platform manager"""
        logger.info("🌐 Initializing Platform Manager...")
        
        # Initialize all platform clients
        from .telegram_client import AdvancedTelegramClient
        from .twitter_client import TwitterClient
        from .instagram_client import InstagramClient
        from .threads_client import ThreadsClient
        
        self.platforms = {
            'telegram': AdvancedTelegramClient(self.nora_core),
            'twitter': TwitterClient(self.nora_core),
            'instagram': InstagramClient(self.nora_core),
            'threads': ThreadsClient(self.nora_core)
        }
        
        # Initialize each platform
        for name, client in self.platforms.items():
            try:
                await client.initialize()
                logger.info(f"✅ {name.title()} initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {name}: {e}")
                
        # Setup cross-platform features
        await self._setup_cross_platform_features()
        
        logger.info("✅ Platform Manager initialized")
        
    async def _setup_cross_platform_features(self):
        """Setup cross-platform features"""
        
        # Content synchronization
        self.content_sync_rules = {
            'auto_cross_post': True,
            'platform_specific_formatting': True,
            'timing_optimization': True,
            'audience_targeting': True
        }
        
        # Unified analytics
        self.analytics_aggregator = AnalyticsAggregator(self.platforms)
        
        # Cross-platform campaigns
        self.campaign_manager = CrossPlatformCampaignManager(self.platforms)
        
    async def post_to_all_platforms(self, content: str, platforms: List[str] = None):
        """Post content to specified platforms (or all)"""
        target_platforms = platforms or list(self.platforms.keys())
        results = {}
        
        for platform_name in target_platforms:
            if platform_name in self.platforms:
                try:
                    # Format content for specific platform
                    formatted_content = await self._format_for_platform(content, platform_name)
                    
                    # Post to platform
                    client = self.platforms[platform_name]
                    
                    if platform_name == 'telegram':
                        success = await client.post_to_channel(formatted_content)
                    elif platform_name == 'twitter':
                        success = await client.post_tweet(formatted_content)
                    elif platform_name == 'instagram':
                        success = await client.post_story(formatted_content)
                    elif platform_name == 'threads':
                        success = await client.post_thread(formatted_content)
                    else:
                        success = False
                        
                    results[platform_name] = {'success': success, 'content': formatted_content}
                    
                except Exception as e:
                    results[platform_name] = {'success': False, 'error': str(e)}
                    
        return results
        
    async def _format_for_platform(self, content: str, platform: str) -> str:
        """Format content specifically for each platform"""
        
        if platform == 'telegram':
            # Rich formatting for Telegram
            formatted = f"📢 **{content}**\n\n"
            formatted += "━━━━━━━━━━━━━━━━━━━━━━\n"
            formatted += "🤖 *تولید شده توسط نورا*\n"
            formatted += f"📅 {datetime.now().strftime('%Y/%m/%d - %H:%M')}\n"
            formatted += "#نورا_AI #هوش_مصنوعی"
            
        elif platform == 'twitter':
            # Twitter character limit and hashtags
            max_length = 280 - 50  # Reserve space for hashtags
            if len(content) > max_length:
                content = content[:max_length-3] + "..."
            formatted = f"{content}\n\n#AI #نورا #TechPersian"
            
        elif platform == 'instagram':
            # Instagram story format
            formatted = f"💭 {content}\n\n📱 @nora_ai_official\n#AI #Persian #Tech"
            
        elif platform == 'threads':
            # Threads conversational format
            formatted = f"🧵 {content}\n\nWhat do you think? Let's discuss! 💬"
            
        else:
            formatted = content
            
        return formatted
        
    async def get_unified_analytics(self) -> Dict:
        """Get unified analytics across all platforms"""
        
        analytics = {
            'total_reach': 0,
            'total_engagement': 0,
            'platform_breakdown': {},
            'top_content': [],
            'audience_insights': {}
        }
        
        for platform_name, client in self.platforms.items():
            try:
                # Get platform-specific analytics
                if hasattr(client, 'get_analytics'):
                    platform_analytics = await client.get_analytics()
                    
                    analytics['platform_breakdown'][platform_name] = platform_analytics
                    analytics['total_reach'] += platform_analytics.get('reach', 0)
                    analytics['total_engagement'] += platform_analytics.get('engagement', 0)
                    
            except Exception as e:
                logger.error(f"Error getting {platform_name} analytics: {e}")
                
        return analytics
        
    async def run_cross_platform_campaign(self, campaign_config: Dict):
        """Run coordinated campaign across platforms"""
        
        campaign_id = f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Schedule content for each platform
        scheduled_posts = []
        
        for platform_name in campaign_config.get('platforms', []):
            if platform_name in self.platforms:
                
                platform_content = campaign_config['content'].get(platform_name)
                if not platform_content:
                    # Generate platform-specific content
                    platform_content = await self._generate_platform_content(
                        campaign_config['theme'], 
                        platform_name
                    )
                
                # Schedule post
                schedule_time = campaign_config.get('schedule_time', datetime.now())
                
                scheduled_posts.append({
                    'platform': platform_name,
                    'content': platform_content,
                    'schedule_time': schedule_time,
                    'campaign_id': campaign_id
                })
                
        # Execute campaign
        results = await self._execute_campaign(scheduled_posts)
        
        return {
            'campaign_id': campaign_id,
            'scheduled_posts': len(scheduled_posts),
            'execution_results': results
        }
        
    async def monitor_all_platforms(self):
        """Monitor all platforms for mentions, comments, messages"""
        
        monitoring_results = {}
        
        for platform_name, client in self.platforms.items():
            try:
                if platform_name == 'telegram':
                    updates = await client.get_updates()
                    monitoring_results[platform_name] = {
                        'new_messages': len(updates),
                        'updates': updates
                    }
                    
                elif platform_name == 'twitter':
                    mentions = await client.check_mentions()
                    monitoring_results[platform_name] = {
                        'new_mentions': len(mentions),
                        'mentions': mentions
                    }
                    
                elif platform_name == 'instagram':
                    comments = await client.check_comments()
                    monitoring_results[platform_name] = {
                        'new_comments': len(comments),
                        'comments': comments
                    }
                    
                # Process and respond to interactions
                await self._process_platform_interactions(platform_name, monitoring_results[platform_name])
                
            except Exception as e:
                logger.error(f"Error monitoring {platform_name}: {e}")
                monitoring_results[platform_name] = {'error': str(e)}
                
        return monitoring_results
        
    async def optimize_posting_schedule(self):
        """Optimize posting schedule based on audience activity"""
        
        # Analyze audience activity patterns
        activity_patterns = await self._analyze_audience_activity()
        
        # Generate optimized schedule
        optimized_schedule = {}
        
        for platform_name in self.platforms.keys():
            platform_pattern = activity_patterns.get(platform_name, {})
            
            # Find peak activity hours
            peak_hours = platform_pattern.get('peak_hours', [12, 18, 21])
            
            # Generate posting schedule
            optimized_schedule[platform_name] = {
                'daily_posts': 3,
                'optimal_hours': peak_hours,
                'time_zone': 'Asia/Tehran',
                'content_types': {
                    'morning': 'news_insights',
                    'afternoon': 'educational_content',
                    'evening': 'engaging_discussions'
                }
            }
            
        return optimized_schedule
        
    async def sync_content_across_platforms(self, source_platform: str, target_platforms: List[str]):
        """Sync content from one platform to others"""
        
        if source_platform not in self.platforms:
            return {'error': 'Source platform not available'}
            
        # Get recent content from source platform
        source_content = await self._get_recent_content(source_platform)
        
        sync_results = {}
        
        for target_platform in target_platforms:
            if target_platform in self.platforms and target_platform != source_platform:
                try:
                    # Adapt content for target platform
                    adapted_content = await self._adapt_content_for_platform(
                        source_content, 
                        source_platform, 
                        target_platform
                    )
                    
                    # Post adapted content
                    result = await self.post_to_all_platforms(adapted_content, [target_platform])
                    sync_results[target_platform] = result[target_platform]
                    
                except Exception as e:
                    sync_results[target_platform] = {'error': str(e)}
                    
        return sync_results


class ContentSyncManager:
    """مدیریت همگام‌سازی محتوا بین پلتفرم‌ها"""
    
    def __init__(self):
        self.sync_rules = {}
        self.content_queue = []
        
    async def setup_sync_rules(self, rules: Dict):
        """Setup content synchronization rules"""
        self.sync_rules = rules
        
    async def queue_content_for_sync(self, content: str, source_platform: str, target_platforms: List[str]):
        """Queue content for synchronization"""
        sync_item = {
            'id': f"sync_{datetime.now().timestamp()}",
            'content': content,
            'source_platform': source_platform,
            'target_platforms': target_platforms,
            'created_at': datetime.now().isoformat(),
            'status': 'queued'
        }
        
        self.content_queue.append(sync_item)
        
    async def process_sync_queue(self):
        """Process queued content synchronization"""
        processed_items = []
        
        for item in self.content_queue:
            if item['status'] == 'queued':
                try:
                    # Process synchronization
                    result = await self._sync_content_item(item)
                    item['status'] = 'completed' if result['success'] else 'failed'
                    item['result'] = result
                    
                except Exception as e:
                    item['status'] = 'failed'
                    item['error'] = str(e)
                    
                processed_items.append(item)
                
        # Remove processed items from queue
        self.content_queue = [item for item in self.content_queue if item['status'] == 'queued']
        
        return processed_items


class AnalyticsAggregator:
    """تجمیع‌کننده آنالیتیک از تمام پلتفرم‌ها"""
    
    def __init__(self, platforms: Dict):
        self.platforms = platforms
        self.aggregated_data = {}
        
    async def collect_all_analytics(self):
        """Collect analytics from all platforms"""
        
        all_analytics = {}
        
        for platform_name, client in self.platforms.items():
            try:
                if hasattr(client, 'get_analytics'):
                    platform_analytics = await client.get_analytics()
                    all_analytics[platform_name] = platform_analytics
                    
            except Exception as e:
                logger.error(f"Error collecting {platform_name} analytics: {e}")
                
        return all_analytics
        
    async def generate_unified_report(self):
        """Generate unified analytics report"""
        
        analytics = await self.collect_all_analytics()
        
        # Aggregate metrics
        total_reach = sum(data.get('reach', 0) for data in analytics.values())
        total_engagement = sum(data.get('engagement', 0) for data in analytics.values())
        
        # Calculate averages
        avg_engagement_rate = total_engagement / total_reach if total_reach > 0 else 0
        
        report = {
            'summary': {
                'total_reach': total_reach,
                'total_engagement': total_engagement,
                'avg_engagement_rate': avg_engagement_rate,
                'active_platforms': len(analytics)
            },
            'platform_breakdown': analytics,
            'insights': await self._generate_insights(analytics),
            'recommendations': await self._generate_recommendations(analytics)
        }
        
        return report


class CrossPlatformCampaignManager:
    """مدیر کمپین‌های میان‌پلتفرمی"""
    
    def __init__(self, platforms: Dict):
        self.platforms = platforms
        self.active_campaigns = {}
        
    async def create_campaign(self, campaign_config: Dict):
        """Create new cross-platform campaign"""
        
        campaign_id = f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        campaign = {
            'id': campaign_id,
            'name': campaign_config.get('name', 'Untitled Campaign'),
            'platforms': campaign_config.get('platforms', []),
            'content_strategy': campaign_config.get('content_strategy', {}),
            'schedule': campaign_config.get('schedule', {}),
            'goals': campaign_config.get('goals', {}),
            'status': 'created',
            'created_at': datetime.now().isoformat()
        }
        
        self.active_campaigns[campaign_id] = campaign
        
        return campaign_id
        
    async def execute_campaign(self, campaign_id: str):
        """Execute campaign across platforms"""
        
        if campaign_id not in self.active_campaigns:
            return {'error': 'Campaign not found'}
            
        campaign = self.active_campaigns[campaign_id]
        campaign['status'] = 'executing'
        
        execution_results = {}
        
        for platform_name in campaign['platforms']:
            if platform_name in self.platforms:
                try:
                    # Execute platform-specific campaign tasks
                    result = await self._execute_platform_campaign(platform_name, campaign)
                    execution_results[platform_name] = result
                    
                except Exception as e:
                    execution_results[platform_name] = {'error': str(e)}
                    
        campaign['status'] = 'completed'
        campaign['results'] = execution_results
        
        return execution_results
