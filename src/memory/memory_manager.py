
"""
memory_manager.py - سیستم مدیریت حافظه نورا
Advanced memory management system for Nora
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import sqlite3
import pickle

logger = logging.getLogger(__name__)

class MemoryManager:
    """Advanced memory management for Nora's experiences and knowledge"""
    
    def __init__(self):
        self.db_path = "data/nora_memory.db"
        self.working_memory = {}
        self.long_term_memory = {}
        
    async def initialize(self):
        """Initialize memory systems"""
        logger.info("🧠 Initializing memory systems...")
        
        # Create database
        await self._create_database()
        
        # Load working memory
        await self._load_working_memory()
        
        logger.info("✅ Memory systems initialized")
        
    async def _create_database(self):
        """Create SQLite database for persistent memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                platform TEXT,
                user_id TEXT,
                user_message TEXT,
                nora_response TEXT,
                context TEXT,
                sentiment REAL,
                importance REAL
            )
        ''')
        
        # Knowledge base table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                content TEXT,
                source TEXT,
                confidence REAL,
                timestamp TEXT,
                last_accessed TEXT
            )
        ''')
        
        # User profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                platform TEXT,
                name TEXT,
                preferences TEXT,
                interaction_count INTEGER,
                sentiment_history TEXT,
                last_interaction TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    async def _load_working_memory(self):
        """Load working memory from cache"""
        try:
            with open('data/working_memory.json', 'r', encoding='utf-8') as f:
                self.working_memory = json.load(f)
        except FileNotFoundError:
            self.working_memory = {
                "current_context": {},
                "active_conversations": {},
                "recent_learnings": [],
                "pending_actions": []
            }
            
    async def store_conversation(self, conversation_data: Dict):
        """Store conversation in memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (timestamp, platform, user_id, user_message, nora_response, context, sentiment, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            conversation_data.get('timestamp', datetime.now().isoformat()),
            conversation_data.get('platform', ''),
            conversation_data.get('user_id', ''),
            conversation_data.get('user_message', ''),
            conversation_data.get('nora_response', ''),
            json.dumps(conversation_data.get('context', {})),
            conversation_data.get('sentiment', 0.0),
            conversation_data.get('importance', 0.5)
        ))
        
        conn.commit()
        conn.close()
        
        # Update working memory
        user_id = conversation_data.get('user_id')
        if user_id:
            if user_id not in self.working_memory['active_conversations']:
                self.working_memory['active_conversations'][user_id] = []
            
            self.working_memory['active_conversations'][user_id].append({
                'timestamp': conversation_data.get('timestamp'),
                'message': conversation_data.get('user_message'),
                'response': conversation_data.get('nora_response')
            })
            
        await self._save_working_memory()
        
    async def retrieve_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve conversation history for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM conversations 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        conversations = []
        for row in rows:
            conversations.append({
                'id': row[0],
                'timestamp': row[1],
                'platform': row[2],
                'user_id': row[3],
                'user_message': row[4],
                'nora_response': row[5],
                'context': json.loads(row[6]) if row[6] else {},
                'sentiment': row[7],
                'importance': row[8]
            })
            
        return conversations
        
    async def store_knowledge(self, knowledge_data: Dict):
        """Store new knowledge"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO knowledge 
            (topic, content, source, confidence, timestamp, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            knowledge_data.get('topic', ''),
            knowledge_data.get('content', ''),
            knowledge_data.get('source', ''),
            knowledge_data.get('confidence', 0.5),
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        # Add to recent learnings
        self.working_memory['recent_learnings'].append({
            'topic': knowledge_data.get('topic'),
            'timestamp': datetime.now().isoformat(),
            'source': knowledge_data.get('source')
        })
        
        # Keep only recent 100 learnings in working memory
        if len(self.working_memory['recent_learnings']) > 100:
            self.working_memory['recent_learnings'] = self.working_memory['recent_learnings'][-100:]
            
        await self._save_working_memory()
        
    async def search_knowledge(self, query: str, limit: int = 5) -> List[Dict]:
        """Search knowledge base"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple text search (can be improved with vector search)
        cursor.execute('''
            SELECT * FROM knowledge 
            WHERE topic LIKE ? OR content LIKE ?
            ORDER BY confidence DESC, timestamp DESC
            LIMIT ?
        ''', (f'%{query}%', f'%{query}%', limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        knowledge_items = []
        for row in rows:
            knowledge_items.append({
                'id': row[0],
                'topic': row[1],
                'content': row[2],
                'source': row[3],
                'confidence': row[4],
                'timestamp': row[5],
                'last_accessed': row[6]
            })
            
        return knowledge_items
        
    async def update_user_profile(self, user_data: Dict):
        """Update or create user profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute('SELECT user_id FROM user_profiles WHERE user_id = ?', (user_data['user_id'],))
        exists = cursor.fetchone()
        
        if exists:
            cursor.execute('''
                UPDATE user_profiles 
                SET name = ?, preferences = ?, interaction_count = interaction_count + 1, 
                    sentiment_history = ?, last_interaction = ?
                WHERE user_id = ?
            ''', (
                user_data.get('name', ''),
                json.dumps(user_data.get('preferences', {})),
                json.dumps(user_data.get('sentiment_history', [])),
                datetime.now().isoformat(),
                user_data['user_id']
            ))
        else:
            cursor.execute('''
                INSERT INTO user_profiles 
                (user_id, platform, name, preferences, interaction_count, sentiment_history, last_interaction)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_data['user_id'],
                user_data.get('platform', ''),
                user_data.get('name', ''),
                json.dumps(user_data.get('preferences', {})),
                1,
                json.dumps(user_data.get('sentiment_history', [])),
                datetime.now().isoformat()
            ))
            
        conn.commit()
        conn.close()
        
    async def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count conversations
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_conversations = cursor.fetchone()[0]
        
        # Count knowledge items
        cursor.execute('SELECT COUNT(*) FROM knowledge')
        total_knowledge = cursor.fetchone()[0]
        
        # Count users
        cursor.execute('SELECT COUNT(*) FROM user_profiles')
        total_users = cursor.fetchone()[0]
        
        # Recent activity
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        cursor.execute('SELECT COUNT(*) FROM conversations WHERE timestamp > ?', (yesterday,))
        recent_conversations = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_conversations": total_conversations,
            "total_knowledge_items": total_knowledge,
            "total_users": total_users,
            "recent_conversations_24h": recent_conversations,
            "working_memory_size": len(self.working_memory),
            "active_conversations": len(self.working_memory.get('active_conversations', {}))
        }
        
    async def _save_working_memory(self):
        """Save working memory to cache"""
        with open('data/working_memory.json', 'w', encoding='utf-8') as f:
            json.dump(self.working_memory, f, ensure_ascii=False, indent=2)
            
    async def cleanup_old_memories(self, days_threshold: int = 30):
        """Clean up old memories to save space"""
        cutoff_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete old, low-importance conversations
        cursor.execute('''
            DELETE FROM conversations 
            WHERE timestamp < ? AND importance < 0.3
        ''', (cutoff_date,))
        
        # Delete old, low-confidence knowledge
        cursor.execute('''
            DELETE FROM knowledge 
            WHERE timestamp < ? AND confidence < 0.3 AND last_accessed < ?
        ''', (cutoff_date, cutoff_date))
        
        conn.commit()
        conn.close()
        
        logger.info(f"🧹 Cleaned up old memories older than {days_threshold} days")
        
    async def run(self):
        """Main memory management loop"""
        logger.info("🧠 Memory manager is now active")
        
        while True:
            try:
                # Periodic memory maintenance
                await self.cleanup_old_memories()
                
                # Save working memory periodically
                await self._save_working_memory()
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in memory manager loop: {e}")
                await asyncio.sleep(300)
                
    async def shutdown(self):
        """Shutdown memory manager"""
        logger.info("🧠 Memory manager shutting down...")
        await self._save_working_memory()
