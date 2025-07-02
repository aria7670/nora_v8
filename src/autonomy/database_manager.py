
"""
database_manager.py - مدیریت پیشرفته پایگاه داده
Advanced database management system for Nora
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
import uuid
import threading
import os
from pathlib import Path
import shutil
import hashlib
import pickle
import gzip

logger = logging.getLogger(__name__)

class AdvancedDatabaseManager:
    """
    مدیر پیشرفته پایگاه داده با قابلیت‌های خودکار
    Advanced database manager with autonomous capabilities
    """
    
    def __init__(self):
        # Database connections
        self.databases = {}
        self.connection_pool = {}
        
        # Database configurations
        self.db_configs = {
            "intelligence": {
                "path": "data/intelligence.db",
                "tables": ["intelligence_metrics", "cognitive_states", "learning_progress"],
                "backup_frequency": "daily",
                "retention_days": 365
            },
            "decisions": {
                "path": "data/decisions.db", 
                "tables": ["autonomous_decisions", "decision_outcomes", "decision_patterns"],
                "backup_frequency": "hourly",
                "retention_days": 730
            },
            "learning": {
                "path": "data/learning.db",
                "tables": ["learning_sessions", "knowledge_items", "skill_development"],
                "backup_frequency": "daily",
                "retention_days": -1  # Permanent
            },
            "activities": {
                "path": "data/activities.db",
                "tables": ["activities_log", "performance_metrics", "user_interactions"],
                "backup_frequency": "daily",
                "retention_days": 180
            },
            "conversations": {
                "path": "data/conversations.db",
                "tables": ["messages", "conversation_context", "user_profiles"],
                "backup_frequency": "daily",
                "retention_days": 365
            },
            "knowledge_graphs": {
                "path": "data/knowledge_graphs.db",
                "tables": ["concepts", "relationships", "semantic_networks"],
                "backup_frequency": "daily",
                "retention_days": -1  # Permanent
            },
            "performance": {
                "path": "data/performance.db",
                "tables": ["system_metrics", "optimization_logs", "error_tracking"],
                "backup_frequency": "hourly",
                "retention_days": 90
            },
            "security": {
                "path": "data/security.db",
                "tables": ["access_logs", "security_events", "threat_analysis"],
                "backup_frequency": "hourly",
                "retention_days": 365
            }
        }
        
        # Background tasks
        self.backup_scheduler = None
        self.cleanup_scheduler = None
        self.optimization_scheduler = None
        
        # Statistics
        self.operation_stats = {
            "queries_executed": 0,
            "data_inserted": 0,
            "data_updated": 0,
            "data_deleted": 0,
            "backups_created": 0,
            "optimizations_performed": 0
        }
        
    async def initialize(self):
        """Initialize database manager"""
        logger.info("🗄️ Initializing Advanced Database Manager...")
        
        # Create data directory
        Path("data").mkdir(exist_ok=True)
        Path("data/backups").mkdir(exist_ok=True)
        
        # Initialize all databases
        await self._initialize_databases()
        
        # Start background tasks
        await self._start_background_tasks()
        
        logger.info("✅ Advanced Database Manager initialized")
        
    async def _initialize_databases(self):
        """Initialize all databases"""
        
        for db_name, config in self.db_configs.items():
            try:
                # Create database connection
                conn = sqlite3.connect(
                    config["path"],
                    check_same_thread=False,
                    timeout=30.0
                )
                
                # Enable WAL mode for better performance
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=MEMORY")
                
                self.databases[db_name] = conn
                
                # Create tables
                await self._create_database_tables(db_name, conn)
                
                logger.info(f"✅ Database {db_name} initialized")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize database {db_name}: {e}")
                
    async def _create_database_tables(self, db_name: str, conn: sqlite3.Connection):
        """Create tables for a specific database"""
        
        cursor = conn.cursor()
        
        if db_name == "intelligence":
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS intelligence_metrics (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    cognitive_score REAL,
                    emotional_score REAL,
                    social_score REAL,
                    creative_score REAL,
                    strategic_score REAL,
                    autonomous_score REAL,
                    overall_score REAL,
                    context TEXT,
                    notes TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cognitive_states (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    state_type TEXT,
                    state_data TEXT,
                    confidence REAL,
                    duration INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_progress (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    domain TEXT,
                    skill TEXT,
                    progress_score REAL,
                    learning_rate REAL,
                    mastery_level TEXT
                )
            ''')
            
        elif db_name == "decisions":
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS autonomous_decisions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    decision_type TEXT,
                    context TEXT,
                    options_considered TEXT,
                    chosen_option TEXT,
                    reasoning TEXT,
                    confidence_score REAL,
                    expected_outcome TEXT,
                    actual_outcome TEXT DEFAULT NULL,
                    success_rating REAL DEFAULT NULL,
                    learning_extracted TEXT DEFAULT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decision_outcomes (
                    id TEXT PRIMARY KEY,
                    decision_id TEXT,
                    timestamp TEXT NOT NULL,
                    outcome_type TEXT,
                    outcome_data TEXT,
                    success_metrics TEXT,
                    lessons_learned TEXT,
                    FOREIGN KEY (decision_id) REFERENCES autonomous_decisions (id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decision_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    pattern_data TEXT,
                    frequency INTEGER,
                    success_rate REAL,
                    confidence REAL,
                    last_updated TEXT
                )
            ''')
            
        elif db_name == "learning":
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    session_type TEXT,
                    source TEXT,
                    content TEXT,
                    concepts_learned TEXT,
                    retention_score REAL,
                    application_success REAL,
                    learning_time INTEGER,
                    effectiveness_score REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    domain TEXT,
                    topic TEXT,
                    content TEXT,
                    source TEXT,
                    confidence REAL,
                    relevance REAL,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 0,
                    usefulness_score REAL DEFAULT 0.5
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS skill_development (
                    id TEXT PRIMARY KEY,
                    skill_name TEXT,
                    skill_category TEXT,
                    current_level REAL,
                    target_level REAL,
                    progress_rate REAL,
                    last_practiced TEXT,
                    practice_count INTEGER DEFAULT 0,
                    mastery_indicators TEXT
                )
            ''')
            
        elif db_name == "activities":
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activities_log (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    activity_type TEXT,
                    description TEXT,
                    platform TEXT,
                    user_involved TEXT,
                    success BOOLEAN,
                    performance_metrics TEXT,
                    user_feedback TEXT,
                    duration INTEGER,
                    resource_usage TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    context TEXT,
                    benchmark_value REAL,
                    improvement_needed BOOLEAN,
                    optimization_suggestions TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    interaction_type TEXT,
                    platform TEXT,
                    content TEXT,
                    sentiment_score REAL,
                    satisfaction_score REAL,
                    response_quality REAL,
                    engagement_level REAL
                )
            ''')
            
        # Add more table definitions for other databases...
        
        # Create indexes for better performance
        await self._create_indexes(db_name, cursor)
        
        conn.commit()
        
    async def _create_indexes(self, db_name: str, cursor: sqlite3.Cursor):
        """Create indexes for better query performance"""
        
        try:
            if db_name == "intelligence":
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_intelligence_timestamp ON intelligence_metrics(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_states_timestamp ON cognitive_states(timestamp)")
                
            elif db_name == "decisions":
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON autonomous_decisions(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_type ON autonomous_decisions(decision_type)")
                
            elif db_name == "learning":
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_learning_timestamp ON learning_sessions(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge_items(domain)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge_items(topic)")
                
            elif db_name == "activities":
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_activities_timestamp ON activities_log(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_activities_type ON activities_log(activity_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_activities_platform ON activities_log(platform)")
                
        except Exception as e:
            logger.error(f"Error creating indexes for {db_name}: {e}")
            
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        
        # Backup scheduler
        self.backup_scheduler = threading.Thread(
            target=self._backup_scheduler_loop,
            daemon=True
        )
        self.backup_scheduler.start()
        
        # Cleanup scheduler
        self.cleanup_scheduler = threading.Thread(
            target=self._cleanup_scheduler_loop,
            daemon=True
        )
        self.cleanup_scheduler.start()
        
        # Optimization scheduler
        self.optimization_scheduler = threading.Thread(
            target=self._optimization_scheduler_loop,
            daemon=True
        )
        self.optimization_scheduler.start()
        
    def _backup_scheduler_loop(self):
        """Background backup scheduler"""
        while True:
            try:
                # Check each database for backup needs
                for db_name, config in self.db_configs.items():
                    if self._needs_backup(db_name, config):
                        asyncio.run(self._create_backup(db_name))
                        
                # Sleep for 1 hour
                threading.Event().wait(3600)
                
            except Exception as e:
                logger.error(f"Backup scheduler error: {e}")
                threading.Event().wait(600)  # Wait 10 minutes on error
                
    def _cleanup_scheduler_loop(self):
        """Background cleanup scheduler"""
        while True:
            try:
                # Run cleanup for each database
                for db_name, config in self.db_configs.items():
                    if config["retention_days"] > 0:
                        asyncio.run(self._cleanup_old_data(db_name, config["retention_days"]))
                        
                # Sleep for 24 hours
                threading.Event().wait(86400)
                
            except Exception as e:
                logger.error(f"Cleanup scheduler error: {e}")
                threading.Event().wait(3600)  # Wait 1 hour on error
                
    def _optimization_scheduler_loop(self):
        """Background optimization scheduler"""
        while True:
            try:
                # Optimize each database
                for db_name in self.databases.keys():
                    asyncio.run(self._optimize_database(db_name))
                    
                # Sleep for 6 hours
                threading.Event().wait(21600)
                
            except Exception as e:
                logger.error(f"Optimization scheduler error: {e}")
                threading.Event().wait(3600)  # Wait 1 hour on error
                
    async def insert_data(self, db_name: str, table: str, data: Dict) -> str:
        """Insert data into database"""
        
        if db_name not in self.databases:
            raise ValueError(f"Database {db_name} not found")
            
        try:
            conn = self.databases[db_name]
            cursor = conn.cursor()
            
            # Generate ID if not provided
            if "id" not in data:
                data["id"] = str(uuid.uuid4())
                
            # Add timestamp if not provided
            if "timestamp" not in data:
                data["timestamp"] = datetime.now().isoformat()
                
            # Prepare insert statement
            columns = list(data.keys())
            placeholders = ", ".join(["?" for _ in columns])
            values = list(data.values())
            
            # Convert complex data to JSON strings
            for i, value in enumerate(values):
                if isinstance(value, (dict, list)):
                    values[i] = json.dumps(value, ensure_ascii=False)
                    
            query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            
            cursor.execute(query, values)
            conn.commit()
            
            self.operation_stats["data_inserted"] += 1
            
            logger.debug(f"📝 Inserted data into {db_name}.{table}: {data['id']}")
            return data["id"]
            
        except Exception as e:
            logger.error(f"Error inserting data into {db_name}.{table}: {e}")
            raise
            
    async def query_data(self, db_name: str, query: str, params: tuple = ()) -> List[Dict]:
        """Query data from database"""
        
        if db_name not in self.databases:
            raise ValueError(f"Database {db_name} not found")
            
        try:
            conn = self.databases[db_name]
            cursor = conn.cursor()
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Convert to list of dictionaries
            data = []
            for row in results:
                row_dict = dict(zip(columns, row))
                
                # Try to parse JSON strings back to objects
                for key, value in row_dict.items():
                    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        try:
                            row_dict[key] = json.loads(value)
                        except:
                            pass  # Keep as string if not valid JSON
                            
                data.append(row_dict)
                
            self.operation_stats["queries_executed"] += 1
            
            logger.debug(f"🔍 Queried {len(data)} records from {db_name}")
            return data
            
        except Exception as e:
            logger.error(f"Error querying {db_name}: {e}")
            raise
            
    async def update_data(self, db_name: str, table: str, data: Dict, where_clause: str, params: tuple = ()) -> int:
        """Update data in database"""
        
        if db_name not in self.databases:
            raise ValueError(f"Database {db_name} not found")
            
        try:
            conn = self.databases[db_name]
            cursor = conn.cursor()
            
            # Prepare update statement  
            set_clauses = []
            values = []
            
            for key, value in data.items():
                set_clauses.append(f"{key} = ?")
                if isinstance(value, (dict, list)):
                    values.append(json.dumps(value, ensure_ascii=False))
                else:
                    values.append(value)
                    
            values.extend(params)
            
            query = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE {where_clause}"
            
            cursor.execute(query, values)
            conn.commit()
            
            rows_affected = cursor.rowcount
            self.operation_stats["data_updated"] += rows_affected
            
            logger.debug(f"✏️ Updated {rows_affected} records in {db_name}.{table}")
            return rows_affected
            
        except Exception as e:
            logger.error(f"Error updating data in {db_name}.{table}: {e}")
            raise
            
    async def delete_data(self, db_name: str, table: str, where_clause: str, params: tuple = ()) -> int:
        """Delete data from database"""
        
        if db_name not in self.databases:
            raise ValueError(f"Database {db_name} not found")
            
        try:
            conn = self.databases[db_name]
            cursor = conn.cursor()
            
            query = f"DELETE FROM {table} WHERE {where_clause}"
            
            cursor.execute(query, params)
            conn.commit()
            
            rows_affected = cursor.rowcount
            self.operation_stats["data_deleted"] += rows_affected
            
            logger.debug(f"🗑️ Deleted {rows_affected} records from {db_name}.{table}")
            return rows_affected
            
        except Exception as e:
            logger.error(f"Error deleting data from {db_name}.{table}: {e}")
            raise
            
    async def _create_backup(self, db_name: str):
        """Create database backup"""
        
        try:
            config = self.db_configs[db_name]
            source_path = config["path"]
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{db_name}_backup_{timestamp}.db"
            backup_path = f"data/backups/{backup_filename}"
            
            # Copy database file
            shutil.copy2(source_path, backup_path)
            
            # Compress backup
            compressed_path = f"{backup_path}.gz"
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
            # Remove uncompressed backup
            os.remove(backup_path)
            
            self.operation_stats["backups_created"] += 1
            
            logger.info(f"💾 Created backup for {db_name}: {compressed_path}")
            
            # Clean up old backups (keep only last 10)
            await self._cleanup_old_backups(db_name)
            
        except Exception as e:
            logger.error(f"Error creating backup for {db_name}: {e}")
            
    async def _cleanup_old_backups(self, db_name: str):
        """Clean up old backup files"""
        
        try:
            backup_dir = Path("data/backups")
            backup_pattern = f"{db_name}_backup_*.db.gz"
            
            # Get all backup files for this database
            backup_files = list(backup_dir.glob(backup_pattern))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only the 10 most recent backups
            for old_backup in backup_files[10:]:
                old_backup.unlink()
                logger.debug(f"🗑️ Removed old backup: {old_backup}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old backups for {db_name}: {e}")
            
    async def _cleanup_old_data(self, db_name: str, retention_days: int):
        """Clean up old data based on retention policy"""
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=retention_days)).isoformat()
            
            config = self.db_configs[db_name]
            
            for table in config["tables"]:
                # Delete old records
                deleted_count = await self.delete_data(
                    db_name, table, "timestamp < ?", (cutoff_date,)
                )
                
                if deleted_count > 0:
                    logger.info(f"🧹 Cleaned up {deleted_count} old records from {db_name}.{table}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old data in {db_name}: {e}")
            
    async def _optimize_database(self, db_name: str):
        """Optimize database performance"""
        
        try:
            conn = self.databases[db_name]
            cursor = conn.cursor()
            
            # Run VACUUM to reclaim space and defragment
            cursor.execute("VACUUM")
            
            # Update statistics
            cursor.execute("ANALYZE")
            
            conn.commit()
            
            self.operation_stats["optimizations_performed"] += 1
            
            logger.debug(f"⚡ Optimized database {db_name}")
            
        except Exception as e:
            logger.error(f"Error optimizing database {db_name}: {e}")
            
    def _needs_backup(self, db_name: str, config: Dict) -> bool:
        """Check if database needs backup"""
        
        try:
            backup_dir = Path("data/backups")
            backup_pattern = f"{db_name}_backup_*.db.gz"
            
            # Get the most recent backup
            backup_files = list(backup_dir.glob(backup_pattern))
            
            if not backup_files:
                return True  # No backups exist
                
            # Get the newest backup
            newest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
            backup_age = datetime.now().timestamp() - newest_backup.stat().st_mtime
            
            # Check if backup is needed based on frequency
            frequency = config["backup_frequency"]
            
            if frequency == "hourly" and backup_age > 3600:
                return True
            elif frequency == "daily" and backup_age > 86400:
                return True
            elif frequency == "weekly" and backup_age > 604800:
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking backup needs for {db_name}: {e}")
            return True  # Backup on error to be safe
            
    async def get_database_stats(self) -> Dict:
        """Get database statistics"""
        
        stats = {
            "operation_stats": self.operation_stats.copy(),
            "database_stats": {}
        }
        
        for db_name, conn in self.databases.items():
            try:
                cursor = conn.cursor()
                
                # Get database size
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                
                db_size = page_count * page_size
                
                # Get table counts
                config = self.db_configs[db_name]
                table_counts = {}
                
                for table in config["tables"]:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        table_counts[table] = count
                    except:
                        table_counts[table] = 0
                        
                stats["database_stats"][db_name] = {
                    "size_bytes": db_size,
                    "size_mb": round(db_size / (1024 * 1024), 2),
                    "table_counts": table_counts,
                    "total_records": sum(table_counts.values())
                }
                
            except Exception as e:
                logger.error(f"Error getting stats for {db_name}: {e}")
                stats["database_stats"][db_name] = {"error": str(e)}
                
        return stats
        
    async def shutdown(self):
        """Shutdown database manager"""
        logger.info("🗄️ Shutting down Database Manager...")
        
        # Close all database connections
        for db_name, conn in self.databases.items():
            try:
                conn.close()
                logger.debug(f"✅ Closed database {db_name}")
            except Exception as e:
                logger.error(f"Error closing database {db_name}: {e}")
                
        logger.info("💤 Database Manager shut down")
