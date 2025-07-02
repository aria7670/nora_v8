
"""
file_manager.py - مدیر فایل هوشمند برای سازماندهی بهتر
Smart file manager for better organization and maintenance
"""

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib
import zipfile

logger = logging.getLogger(__name__)

class SmartFileManager:
    """مدیر فایل هوشمند برای سازماندهی و نگهداری"""
    
    def __init__(self):
        self.project_structure = {
            'src/': {
                'description': 'کد اصلی پروژه',
                'subdirs': {
                    'ai_core.py': 'هسته هوش مصنوعی',
                    'web_dashboard.py': 'داشبورد وب',
                    'advanced_capabilities/': 'قابلیت‌های پیشرفته',
                    'autonomy/': 'سیستم‌های خودمختار',
                    'platforms/': 'کلاینت‌های پلتفرم',
                    'personality/': 'سیستم شخصیت',
                    'learning/': 'سیستم‌های یادگیری',
                    'memory/': 'مدیریت حافظه',
                    'analytics/': 'موتور تحلیل',
                    'utils/': 'ابزارهای کمکی'
                }
            },
            'config/': {
                'description': 'فایل‌های تنظیمات',
                'subdirs': {
                    'platform_configs/': 'تنظیمات پلتفرم‌ها',
                    'security/': 'تنظیمات امنیتی'
                }
            },
            'data/': {
                'description': 'پایگاه داده‌ها',
                'subdirs': {}
            },
            'logs/': {
                'description': 'فایل‌های لاگ',
                'subdirs': {}
            },
            'docs/': {
                'description': 'مستندات',
                'subdirs': {
                    'technical/': 'مستندات فنی',
                    'user_guide/': 'راهنمای کاربر',
                    'api/': 'مستندات API'
                }
            },
            'templates/': {
                'description': 'قالب‌های HTML',
                'subdirs': {}
            },
            'static/': {
                'description': 'فایل‌های استاتیک',
                'subdirs': {
                    'css/': 'فایل‌های CSS',
                    'js/': 'فایل‌های JavaScript',
                    'images/': 'تصاویر'
                }
            }
        }
        
    async def organize_project_structure(self):
        """سازماندهی ساختار پروژه"""
        logger.info("📁 Organizing project structure...")
        
        for main_dir, info in self.project_structure.items():
            # Create main directory
            Path(main_dir).mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            for subdir in info.get('subdirs', {}):
                if subdir.endswith('/'):
                    Path(main_dir + subdir).mkdir(parents=True, exist_ok=True)
                    
        # Create __init__.py files
        await self._create_init_files()
        
        logger.info("✅ Project structure organized")
        
    async def _create_init_files(self):
        """ایجاد فایل‌های __init__.py"""
        
        init_locations = [
            'src/',
            'src/advanced_capabilities/',
            'src/autonomy/',
            'src/platforms/',
            'src/personality/',
            'src/learning/',
            'src/memory/',
            'src/analytics/',
            'src/metacognition/',
            'src/utils/'
        ]
        
        for location in init_locations:
            init_file = Path(location) / '__init__.py'
            if not init_file.exists():
                init_file.write_text('# -*- coding: utf-8 -*-\n"""Auto-generated __init__.py"""\n')
                
    async def cleanup_project(self):
        """پاک‌سازی پروژه از فایل‌های غیرضروری"""
        logger.info("🧹 Cleaning up project...")
        
        # Files to remove
        cleanup_patterns = [
            '*.pyc',
            '__pycache__/',
            '*.tmp',
            '*.log.old',
            '.DS_Store',
            'Thumbs.db'
        ]
        
        # Clean up directories
        for pattern in cleanup_patterns:
            await self._remove_pattern(pattern)
            
        logger.info("✅ Project cleanup completed")
        
    async def _remove_pattern(self, pattern: str):
        """حذف فایل‌ها بر اساس الگو"""
        import glob
        
        for file_path in glob.glob(pattern, recursive=True):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(f"Could not remove {file_path}: {e}")
                
    async def create_backup(self, backup_name: Optional[str] = None) -> str:
        """ایجاد پشتیبان از پروژه"""
        
        if not backup_name:
            backup_name = f"nora_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        backup_path = f"backups/{backup_name}.zip"
        Path("backups").mkdir(exist_ok=True)
        
        logger.info(f"💾 Creating backup: {backup_path}")
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
            for root, dirs, files in os.walk('.'):
                # Skip certain directories
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'backups', 'venv']]
                
                for file in files:
                    if not file.endswith('.pyc'):
                        file_path = os.path.join(root, file)
                        backup_zip.write(file_path, file_path)
                        
        logger.info(f"✅ Backup created: {backup_path}")
        return backup_path
        
    async def generate_project_map(self) -> Dict:
        """تولید نقشه پروژه"""
        
        project_map = {
            'structure': {},
            'file_count': 0,
            'total_size': 0,
            'last_updated': datetime.now().isoformat()
        }
        
        for root, dirs, files in os.walk('.'):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            relative_root = os.path.relpath(root, '.')
            if relative_root == '.':
                relative_root = 'root'
                
            project_map['structure'][relative_root] = {
                'files': [],
                'size': 0
            }
            
            for file in files:
                if not file.startswith('.') and not file.endswith('.pyc'):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    
                    project_map['structure'][relative_root]['files'].append({
                        'name': file,
                        'size': file_size,
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                    })
                    
                    project_map['structure'][relative_root]['size'] += file_size
                    project_map['file_count'] += 1
                    project_map['total_size'] += file_size
                    
        # Save project map
        with open('docs/project_map.json', 'w', encoding='utf-8') as f:
            json.dump(project_map, f, ensure_ascii=False, indent=2)
            
        return project_map
        
    async def validate_file_integrity(self) -> Dict:
        """بررسی یکپارچگی فایل‌ها"""
        
        integrity_report = {
            'status': 'checking',
            'files_checked': 0,
            'errors': [],
            'warnings': [],
            'timestamp': datetime.now().isoformat()
        }
        
        critical_files = [
            'main.py',
            'src/ai_core.py',
            'src/web_dashboard.py'
        ]
        
        for file_path in critical_files:
            if not os.path.exists(file_path):
                integrity_report['errors'].append(f"Critical file missing: {file_path}")
            else:
                integrity_report['files_checked'] += 1
                
        # Check Python syntax
        for root, dirs, files in os.walk('src'):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            compile(f.read(), file_path, 'exec')
                        integrity_report['files_checked'] += 1
                    except SyntaxError as e:
                        integrity_report['errors'].append(f"Syntax error in {file_path}: {e}")
                    except Exception as e:
                        integrity_report['warnings'].append(f"Could not check {file_path}: {e}")
                        
        integrity_report['status'] = 'completed'
        
        # Save integrity report
        with open('logs/integrity_report.json', 'w', encoding='utf-8') as f:
            json.dump(integrity_report, f, ensure_ascii=False, indent=2)
            
        return integrity_report
