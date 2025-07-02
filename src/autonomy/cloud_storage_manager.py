
"""
cloud_storage_manager.py - سیستم مدیریت ذخیره‌سازی ابری
Cloud Storage Management System using Telegram channels
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid
import hashlib
import zipfile
import gzip
import sqlite3
import pickle
import base64
from pathlib import Path
import aiofiles
import aiohttp
from collections import defaultdict

logger = logging.getLogger(__name__)

class CloudStorageManager:
    """
    مدیر ذخیره‌سازی ابری با استفاده از کانال تلگرام
    Cloud Storage Manager using Telegram channel as storage backend
    """
    
    def __init__(self, telegram_client):
        self.telegram_client = telegram_client
        self.storage_channel_id = None
        self.backup_channel_id = None
        
        # File management
        self.file_index = {}
        self.backup_index = {}
        self.storage_stats = {}
        
        # Backup configuration
        self.backup_config = {
            'databases': {
                'intelligence.db': {'frequency': 'hourly', 'retention': 30},
                'decisions.db': {'frequency': 'daily', 'retention': 90},
                'learning.db': {'frequency': 'daily', 'retention': 365},
                'activities.db': {'frequency': 'daily', 'retention': 180},
                'performance.db': {'frequency': 'weekly', 'retention': 52},
                'knowledge_graphs.db': {'frequency': 'daily', 'retention': 365}
            },
            'configs': {
                'frequency': 'daily',
                'retention': 365
            },
            'logs': {
                'frequency': 'weekly',
                'retention': 12
            }
        }
        
        # Encryption settings
        self.encryption_enabled = True
        self.encryption_key = self._generate_encryption_key()
        
        # Compression settings
        self.compression_enabled = True
        self.compression_level = 6
        
    async def initialize(self):
        """Initialize cloud storage system"""
        logger.info("☁️ Initializing Cloud Storage Manager...")
        
        # Setup storage channels
        await self._setup_storage_channels()
        
        # Load file indexes
        await self._load_file_indexes()
        
        # Setup backup schedules
        await self._setup_backup_schedules()
        
        # Start maintenance tasks
        await self._start_maintenance_tasks()
        
        logger.info("✅ Cloud Storage Manager initialized")
        
    async def _setup_storage_channels(self):
        """Setup dedicated storage channels"""
        
        try:
            # Main storage channel
            storage_channel = await self._create_or_find_channel(
                "فضای_ابری_نورا",
                "📁 فضای ذخیره‌سازی ابری نورا - فایل‌های مهم و داده‌ها"
            )
            self.storage_channel_id = storage_channel['id']
            
            # Backup channel
            backup_channel = await self._create_or_find_channel(
                "پشتیبان_گیری_نورا", 
                "💾 کانال پشتیبان‌گیری خودکار نورا - بکاپ دیتابیس‌ها و تنظیمات"
            )
            self.backup_channel_id = backup_channel['id']
            
            logger.info("Storage channels setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup storage channels: {e}")
            
    async def _create_or_find_channel(self, channel_name: str, description: str) -> Dict:
        """Create channel if doesn't exist or find existing one"""
        
        try:
            # Try to find existing channel first
            existing_channel = await self._find_channel_by_name(channel_name)
            
            if existing_channel:
                return existing_channel
                
            # Create new channel
            new_channel = await self.telegram_client.create_channel(
                title=channel_name,
                description=description,
                private=True
            )
            
            return new_channel
            
        except Exception as e:
            logger.error(f"Channel creation/finding failed: {e}")
            raise
            
    async def store_file(self, file_path: str, category: str = "general", 
                        metadata: Dict = None) -> Dict:
        """Store file in cloud storage"""
        
        file_id = str(uuid.uuid4())
        
        try:
            # Read file
            async with aiofiles.open(file_path, 'rb') as f:
                file_data = await f.read()
                
            # Prepare metadata
            file_metadata = {
                'file_id': file_id,
                'original_path': file_path,
                'filename': Path(file_path).name,
                'size': len(file_data),
                'category': category,
                'uploaded_at': datetime.now().isoformat(),
                'hash': hashlib.sha256(file_data).hexdigest(),
                'custom_metadata': metadata or {}
            }
            
            # Compress if enabled
            if self.compression_enabled:
                compressed_data = await self._compress_data(file_data)
                file_metadata['compressed'] = True
                file_metadata['original_size'] = len(file_data)
                file_metadata['compressed_size'] = len(compressed_data)
                upload_data = compressed_data
            else:
                upload_data = file_data
                
            # Encrypt if enabled
            if self.encryption_enabled:
                encrypted_data = await self._encrypt_data(upload_data)
                file_metadata['encrypted'] = True
                upload_data = encrypted_data
                
            # Upload to Telegram
            message = await self.telegram_client.send_document(
                chat_id=self.storage_channel_id,
                document=upload_data,
                filename=f"{file_id}_{Path(file_path).name}",
                caption=f"📁 {category}\n🔍 {file_id}"
            )
            
            # Update file index
            file_metadata['message_id'] = message.message_id
            file_metadata['telegram_file_id'] = message.document.file_id
            
            self.file_index[file_id] = file_metadata
            await self._save_file_index()
            
            # Update storage stats
            await self._update_storage_stats(file_metadata)
            
            logger.info(f"File stored successfully: {file_path} -> {file_id}")
            
            return {
                'file_id': file_id,
                'metadata': file_metadata,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"File storage failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
            
    async def retrieve_file(self, file_id: str, output_path: str = None) -> Dict:
        """Retrieve file from cloud storage"""
        
        try:
            # Get file metadata
            file_metadata = self.file_index.get(file_id)
            
            if not file_metadata:
                return {'error': 'File not found', 'success': False}
                
            # Download from Telegram
            file_data = await self.telegram_client.download_file(
                file_metadata['telegram_file_id']
            )
            
            # Decrypt if needed
            if file_metadata.get('encrypted'):
                file_data = await self._decrypt_data(file_data)
                
            # Decompress if needed
            if file_metadata.get('compressed'):
                file_data = await self._decompress_data(file_data)
                
            # Save to file if output path provided
            if output_path:
                async with aiofiles.open(output_path, 'wb') as f:
                    await f.write(file_data)
                    
            return {
                'file_id': file_id,
                'file_data': file_data,
                'metadata': file_metadata,
                'output_path': output_path,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"File retrieval failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
            
    async def backup_database(self, db_path: str, backup_name: str = None) -> Dict:
        """Backup database to cloud storage"""
        
        backup_id = str(uuid.uuid4())
        backup_name = backup_name or Path(db_path).stem
        
        try:
            # Create database backup
            backup_data = await self._create_database_backup(db_path)
            
            # Prepare backup metadata
            backup_metadata = {
                'backup_id': backup_id,
                'backup_name': backup_name,
                'database_path': db_path,
                'backup_time': datetime.now().isoformat(),
                'size': len(backup_data),
                'hash': hashlib.sha256(backup_data).hexdigest(),
                'type': 'database_backup'
            }
            
            # Compress backup
            compressed_backup = await self._compress_data(backup_data)
            backup_metadata['compressed_size'] = len(compressed_backup)
            
            # Encrypt backup
            if self.encryption_enabled:
                encrypted_backup = await self._encrypt_data(compressed_backup)
                upload_data = encrypted_backup
                backup_metadata['encrypted'] = True
            else:
                upload_data = compressed_backup
                
            # Upload to backup channel
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backup_{backup_name}_{timestamp}.db.bak"
            
            message = await self.telegram_client.send_document(
                chat_id=self.backup_channel_id,
                document=upload_data,
                filename=filename,
                caption=f"💾 Database Backup\n📊 {backup_name}\n🕐 {timestamp}\n📏 {len(backup_data)} bytes"
            )
            
            # Update backup index
            backup_metadata['message_id'] = message.message_id
            backup_metadata['telegram_file_id'] = message.document.file_id
            backup_metadata['filename'] = filename
            
            if backup_name not in self.backup_index:
                self.backup_index[backup_name] = []
                
            self.backup_index[backup_name].append(backup_metadata)
            await self._save_backup_index()
            
            # Cleanup old backups
            await self._cleanup_old_backups(backup_name)
            
            logger.info(f"Database backup completed: {db_path} -> {backup_id}")
            
            return {
                'backup_id': backup_id,
                'metadata': backup_metadata,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
            
    async def restore_database(self, backup_id: str, restore_path: str) -> Dict:
        """Restore database from backup"""
        
        try:
            # Find backup metadata
            backup_metadata = None
            for backups in self.backup_index.values():
                for backup in backups:
                    if backup['backup_id'] == backup_id:
                        backup_metadata = backup
                        break
                if backup_metadata:
                    break
                    
            if not backup_metadata:
                return {'error': 'Backup not found', 'success': False}
                
            # Download backup
            backup_data = await self.telegram_client.download_file(
                backup_metadata['telegram_file_id']
            )
            
            # Decrypt if needed
            if backup_metadata.get('encrypted'):
                backup_data = await self._decrypt_data(backup_data)
                
            # Decompress backup
            decompressed_data = await self._decompress_data(backup_data)
            
            # Restore database
            await self._restore_database_from_data(decompressed_data, restore_path)
            
            logger.info(f"Database restored: {backup_id} -> {restore_path}")
            
            return {
                'backup_id': backup_id,
                'restore_path': restore_path,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Database restoration failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
            
    async def schedule_automatic_backups(self):
        """Schedule automatic backups"""
        
        try:
            for db_name, config in self.backup_config['databases'].items():
                db_path = f"data/{db_name}"
                
                if Path(db_path).exists():
                    # Schedule backup based on frequency
                    if config['frequency'] == 'hourly':
                        asyncio.create_task(self._hourly_backup(db_path, db_name))
                    elif config['frequency'] == 'daily':
                        asyncio.create_task(self._daily_backup(db_path, db_name))
                    elif config['frequency'] == 'weekly':
                        asyncio.create_task(self._weekly_backup(db_path, db_name))
                        
            logger.info("Automatic backup schedules created")
            
        except Exception as e:
            logger.error(f"Backup scheduling failed: {e}")
            
    async def _hourly_backup(self, db_path: str, db_name: str):
        """Perform hourly backup"""
        while True:
            try:
                await asyncio.sleep(3600)  # 1 hour
                await self.backup_database(db_path, db_name)
            except Exception as e:
                logger.error(f"Hourly backup failed for {db_name}: {e}")
                
    async def _daily_backup(self, db_path: str, db_name: str):
        """Perform daily backup"""
        while True:
            try:
                await asyncio.sleep(86400)  # 24 hours
                await self.backup_database(db_path, db_name)
            except Exception as e:
                logger.error(f"Daily backup failed for {db_name}: {e}")
                
    async def _weekly_backup(self, db_path: str, db_name: str):
        """Perform weekly backup"""
        while True:
            try:
                await asyncio.sleep(604800)  # 7 days
                await self.backup_database(db_path, db_name)
            except Exception as e:
                logger.error(f"Weekly backup failed for {db_name}: {e}")
                
    async def get_storage_statistics(self) -> Dict:
        """Get storage statistics"""
        
        total_files = len(self.file_index)
        total_size = sum(file['size'] for file in self.file_index.values())
        
        # Calculate category statistics
        category_stats = defaultdict(lambda: {'count': 0, 'size': 0})
        for file_metadata in self.file_index.values():
            category = file_metadata['category']
            category_stats[category]['count'] += 1
            category_stats[category]['size'] += file_metadata['size']
            
        # Calculate backup statistics
        backup_stats = {}
        for backup_name, backups in self.backup_index.items():
            backup_stats[backup_name] = {
                'count': len(backups),
                'latest': max(backup['backup_time'] for backup in backups) if backups else None,
                'total_size': sum(backup['size'] for backup in backups)
            }
            
        return {
            'total_files': total_files,
            'total_size': total_size,
            'category_statistics': dict(category_stats),
            'backup_statistics': backup_stats,
            'storage_channels': {
                'main_storage': self.storage_channel_id,
                'backup_storage': self.backup_channel_id
            }
        }
        
    async def cleanup_old_files(self, days_old: int = 90):
        """Cleanup old files based on age"""
        
        cleanup_results = {
            'files_deleted': 0,
            'space_freed': 0,
            'errors': []
        }
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        files_to_delete = []
        for file_id, file_metadata in self.file_index.items():
            upload_date = datetime.fromisoformat(file_metadata['uploaded_at'])
            if upload_date < cutoff_date:
                files_to_delete.append(file_id)
                
        for file_id in files_to_delete:
            try:
                await self.delete_file(file_id)
                cleanup_results['files_deleted'] += 1
                cleanup_results['space_freed'] += self.file_index[file_id]['size']
            except Exception as e:
                cleanup_results['errors'].append(f"Failed to delete {file_id}: {e}")
                
        return cleanup_results
        
    async def _create_database_backup(self, db_path: str) -> bytes:
        """Create database backup data"""
        
        try:
            # Read database file
            async with aiofiles.open(db_path, 'rb') as f:
                db_data = await f.read()
                
            # Create backup with metadata
            backup_info = {
                'original_path': db_path,
                'backup_time': datetime.now().isoformat(),
                'size': len(db_data),
                'hash': hashlib.sha256(db_data).hexdigest()
            }
            
            # Combine metadata and data
            backup_package = {
                'info': backup_info,
                'data': base64.b64encode(db_data).decode('utf-8')
            }
            
            return json.dumps(backup_package).encode('utf-8')
            
        except Exception as e:
            logger.error(f"Database backup creation failed: {e}")
            raise
