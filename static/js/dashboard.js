
// Nora Dashboard JavaScript

class NoraDashboard {
    constructor() {
        this.refreshInterval = 30000; // 30 seconds
        this.init();
    }

    init() {
        this.loadInitialData();
        this.setupEventListeners();
        this.startPeriodicRefresh();
    }

    async loadInitialData() {
        await Promise.all([
            this.updateSystemStatus(),
            this.updateKPIWidgets(),
            this.updatePlatformMetrics(),
            this.updateLearningChannels(),
            this.updateLiveActivity(),
            this.updateConsciousnessState(),
            this.updateMemoryStats(),
            this.updateMetacognitionStatus()
        ]);
    }

    setupEventListeners() {
        // Auto-refresh on window focus
        window.addEventListener('focus', () => {
            this.loadInitialData();
        });

        // Handle modal close on outside click
        window.addEventListener('click', (event) => {
            const modal = document.getElementById('addChannelModal');
            if (event.target === modal) {
                this.closeModal();
            }
        });
    }

    startPeriodicRefresh() {
        setInterval(() => {
            this.loadInitialData();
        }, this.refreshInterval);
    }

    async apiCall(endpoint) {
        try {
            const response = await fetch(`/api/${endpoint}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Error calling ${endpoint}:`, error);
            return null;
        }
    }

    async updateSystemStatus() {
        const data = await this.apiCall('system_status');
        if (!data) return;

        const vitalsContainer = document.getElementById('systemVitals');
        vitalsContainer.innerHTML = '';

        // Core Modules
        Object.entries(data.core_modules || {}).forEach(([module, status]) => {
            vitalsContainer.appendChild(this.createVitalCard(
                module.replace('_', ' ').toUpperCase(),
                status.status,
                status.last_activity,
                'fas fa-cog'
            ));
        });

        // Platforms
        Object.entries(data.platforms || {}).forEach(([platform, status]) => {
            vitalsContainer.appendChild(this.createVitalCard(
                platform.toUpperCase(),
                status.status,
                status.last_activity,
                this.getPlatformIcon(platform)
            ));
        });

        // AI Models
        Object.entries(data.ai_models || {}).forEach(([model, status]) => {
            vitalsContainer.appendChild(this.createVitalCard(
                model.toUpperCase(),
                status.status,
                status.response_time || status.progress,
                'fas fa-brain'
            ));
        });
    }

    createVitalCard(title, status, detail, icon) {
        const card = document.createElement('div');
        card.className = 'vital-card';
        
        const statusClass = this.getStatusClass(status);
        
        card.innerHTML = `
            <h4><i class="${icon}"></i> ${title}</h4>
            <div class="vital-status ${statusClass}">
                <i class="fas fa-circle"></i>
                <span>${this.translateStatus(status)}</span>
            </div>
            <div class="vital-detail">${detail}</div>
        `;
        
        return card;
    }

    getStatusClass(status) {
        switch (status) {
            case 'active':
            case 'connected':
                return 'online';
            case 'training':
            case 'standby':
                return 'training';
            default:
                return 'offline';
        }
    }

    translateStatus(status) {
        const translations = {
            'active': 'فعال',
            'connected': 'متصل',
            'training': 'در حال آموزش',
            'standby': 'آماده‌باش',
            'offline': 'آفلاین'
        };
        return translations[status] || status;
    }

    getPlatformIcon(platform) {
        const icons = {
            'telegram': 'fab fa-telegram-plane',
            'twitter': 'fab fa-twitter',
            'instagram': 'fab fa-instagram',
            'threads': 'fas fa-comments'
        };
        return icons[platform] || 'fas fa-globe';
    }

    async updateKPIWidgets() {
        const data = await this.apiCall('kpi_widgets');
        if (!data) return;

        const kpiContainer = document.getElementById('kpiWidgets');
        kpiContainer.innerHTML = '';

        Object.entries(data).forEach(([key, widget]) => {
            kpiContainer.appendChild(this.createKPIWidget(
                this.translateKPI(key),
                widget.value,
                widget.change,
                widget.trend
            ));
        });
    }

    createKPIWidget(title, value, change, trend) {
        const widget = document.createElement('div');
        widget.className = 'kpi-widget';
        
        const trendClass = trend === 'up' ? 'up' : trend === 'down' ? 'down' : '';
        const trendIcon = trend === 'up' ? 'fa-arrow-up' : trend === 'down' ? 'fa-arrow-down' : 'fa-minus';
        
        widget.innerHTML = `
            <h4>${title}</h4>
            <div class="kpi-value">${value}</div>
            <div class="kpi-change ${trendClass}">
                <i class="fas ${trendIcon}"></i> ${change}
            </div>
        `;
        
        return widget;
    }

    translateKPI(key) {
        const translations = {
            'follower_growth_24h': 'رشد فالوور ۲۴ ساعت',
            'engagement_rate_weekly': 'نرخ تعامل هفتگی',
            'total_knowledge_acquired': 'کل دانش کسب‌شده',
            'api_cost_today': 'هزینه API امروز',
            'conversation_quality_score': 'کیفیت مکالمات',
            'learning_efficiency': 'کارایی یادگیری'
        };
        return translations[key] || key;
    }

    async updatePlatformMetrics() {
        const data = await this.apiCall('platform_metrics');
        if (!data) return;

        const platformsContainer = document.getElementById('platformMetrics');
        platformsContainer.innerHTML = '';

        Object.entries(data).forEach(([platform, metrics]) => {
            platformsContainer.appendChild(this.createPlatformCard(platform, metrics));
        });
    }

    createPlatformCard(platform, metrics) {
        const card = document.createElement('div');
        card.className = `platform-card ${platform}`;
        
        card.innerHTML = `
            <h4>
                <i class="${this.getPlatformIcon(platform)}"></i>
                ${platform.toUpperCase()}
            </h4>
            <div class="platform-metrics">
                ${Object.entries(metrics).map(([key, value]) => `
                    <div class="metric-item">
                        <div class="metric-value">${value}</div>
                        <div class="metric-label">${this.translateMetric(key)}</div>
                    </div>
                `).join('')}
            </div>
        `;
        
        return card;
    }

    translateMetric(key) {
        const translations = {
            'active_chats': 'چت فعال',
            'messages_today': 'پیام امروز',
            'learning_channels': 'کانال یادگیری',
            'avg_response_time': 'زمان پاسخ',
            'followers': 'فالوور',
            'tweets_today': 'توییت امروز',
            'engagement_rate': 'نرخ تعامل',
            'mentions': 'منشن',
            'posts_today': 'پست امروز',
            'stories': 'استوری',
            'threads_today': 'ترد امروز',
            'replies': 'پاسخ'
        };
        return translations[key] || key;
    }

    async updateLearningChannels() {
        const data = await this.apiCall('learning_channels');
        if (!data) return;

        const channelsContainer = document.getElementById('learningChannels');
        channelsContainer.innerHTML = '';

        if (data.channels && data.channels.length > 0) {
            data.channels.forEach(channel => {
                channelsContainer.appendChild(this.createChannelCard(channel));
            });
        } else {
            channelsContainer.innerHTML = '<p>هیچ کانال یادگیری تعریف نشده است.</p>';
        }
    }

    createChannelCard(channel) {
        const card = document.createElement('div');
        card.className = 'channel-card';
        
        const status = channel.active ? 'active' : 'inactive';
        const statusText = channel.active ? 'فعال' : 'غیرفعال';
        
        card.innerHTML = `
            <div class="channel-header">
                <div class="channel-name">${channel.name}</div>
                <div class="channel-status ${status}">${statusText}</div>
            </div>
            <div class="channel-id">${channel.id}</div>
            <div class="channel-weight">وزن یادگیری: ${channel.learning_weight || 0.5}</div>
            <div class="channel-topics">
                ${(channel.topics || []).map(topic => 
                    `<span class="topic-tag">${topic}</span>`
                ).join('')}
            </div>
        `;
        
        return card;
    }

    async updateLiveActivity() {
        const data = await this.apiCall('live_activity');
        if (!data) return;

        const activityContainer = document.getElementById('liveActivity');
        activityContainer.innerHTML = '';

        if (Array.isArray(data) && data.length > 0) {
            data.forEach(activity => {
                activityContainer.appendChild(this.createActivityItem(activity));
            });
        } else {
            activityContainer.innerHTML = '<p>فعالیت جدیدی ثبت نشده است.</p>';
        }
    }

    createActivityItem(activity) {
        const item = document.createElement('div');
        item.className = `activity-item ${activity.type || 'system'}`;
        
        const time = new Date(activity.timestamp).toLocaleTimeString('fa-IR');
        
        item.innerHTML = `
            <div class="activity-header">
                <span class="activity-level ${activity.level || 'INFO'}">${activity.level || 'INFO'}</span>
                <span class="activity-time">${time}</span>
            </div>
            <div class="activity-message">${activity.message}</div>
        `;
        
        return item;
    }

    async updateConsciousnessState() {
        const data = await this.apiCall('consciousness_state');
        if (!data) return;

        const container = document.getElementById('consciousnessState');
        container.innerHTML = '';

        if (data.personality) {
            const personalityDiv = document.createElement('div');
            personalityDiv.innerHTML = '<h4>ویژگی‌های شخصیتی:</h4>';
            
            Object.entries(data.personality).forEach(([trait, value]) => {
                const traitDiv = document.createElement('div');
                traitDiv.className = 'personality-trait';
                traitDiv.innerHTML = `
                    <span>${this.translateTrait(trait)}</span>
                    <div class="trait-bar">
                        <div class="trait-fill" style="width: ${value * 100}%"></div>
                    </div>
                `;
                personalityDiv.appendChild(traitDiv);
            });
            
            container.appendChild(personalityDiv);
        }

        if (data.version) {
            container.innerHTML += `<p><strong>نسخه:</strong> ${data.version}</p>`;
        }

        if (data.active_conversations !== undefined) {
            container.innerHTML += `<p><strong>مکالمات فعال:</strong> ${data.active_conversations}</p>`;
        }
    }

    translateTrait(trait) {
        const translations = {
            'curiosity': 'کنجکاوی',
            'honesty': 'صداقت',
            'analytical': 'تحلیلگری',
            'empathy': 'همدلی',
            'creativity': 'خلاقیت',
            'loyalty_to_aria': 'وفاداری به آریا'
        };
        return translations[trait] || trait;
    }

    async updateMemoryStats() {
        const data = await this.apiCall('memory_stats');
        if (!data) return;

        const container = document.getElementById('memoryStats');
        container.innerHTML = '';

        Object.entries(data).forEach(([key, value]) => {
            const statDiv = document.createElement('div');
            statDiv.className = 'memory-stat';
            statDiv.innerHTML = `
                <span>${this.translateMemoryStat(key)}</span>
                <span class="stat-value">${value}</span>
            `;
            container.appendChild(statDiv);
        });
    }

    translateMemoryStat(key) {
        const translations = {
            'total_conversations': 'کل مکالمات',
            'total_knowledge_items': 'کل اقلام دانش',
            'total_users': 'کل کاربران',
            'recent_conversations_24h': 'مکالمات ۲۴ ساعت',
            'working_memory_size': 'حجم حافظه کاری',
            'active_conversations': 'مکالمات فعال'
        };
        return translations[key] || key;
    }

    async updateMetacognitionStatus() {
        const data = await this.apiCall('metacognition_status');
        if (!data) return;

        const container = document.getElementById('metacognitionStatus');
        container.innerHTML = '';

        if (data.recent_reflections !== undefined) {
            container.innerHTML += `<p><strong>خودبازبینی‌های اخیر:</strong> ${data.recent_reflections}</p>`;
        }

        if (data.evolution_proposals !== undefined) {
            container.innerHTML += `<p><strong>پیشنهادات تکامل:</strong> ${data.evolution_proposals}</p>`;
        }

        if (data.last_reflection) {
            const lastReflection = new Date(data.last_reflection.timestamp).toLocaleDateString('fa-IR');
            container.innerHTML += `<p><strong>آخرین خودبازبینی:</strong> ${lastReflection}</p>`;
        }
    }

    // Modal Functions
    showAddChannelModal() {
        document.getElementById('addChannelModal').style.display = 'block';
    }

    closeModal() {
        document.getElementById('addChannelModal').style.display = 'none';
        document.getElementById('addChannelForm').reset();
    }

    async addLearningChannel() {
        const form = document.getElementById('addChannelForm');
        const formData = new FormData(form);
        
        const channelData = {
            id: document.getElementById('channelId').value,
            name: document.getElementById('channelName').value,
            topics: document.getElementById('channelTopics').value.split(',').map(t => t.trim()),
            learning_weight: parseFloat(document.getElementById('learningWeight').value),
            active: document.getElementById('channelActive').checked
        };

        try {
            const response = await fetch('/api/add_learning_channel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(channelData)
            });

            const result = await response.json();
            
            if (result.success) {
                this.closeModal();
                await this.updateLearningChannels();
                this.showNotification('کانال یادگیری با موفقیت اضافه شد!', 'success');
            } else {
                this.showNotification(result.message, 'error');
            }
        } catch (error) {
            console.error('Error adding channel:', error);
            this.showNotification('خطا در افزودن کانال', 'error');
        }
    }

    async refreshLearningChannels() {
        await this.updateLearningChannels();
        this.showNotification('کانال‌های یادگیری بروزرسانی شدند', 'info');
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // Style the notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '15px 20px',
            borderRadius: '8px',
            color: 'white',
            fontWeight: 'bold',
            zIndex: '10000',
            maxWidth: '300px'
        });

        // Set background color based on type
        const colors = {
            success: '#48bb78',
            error: '#f56565',
            info: '#4299e1',
            warning: '#ed8936'
        };
        notification.style.backgroundColor = colors[type] || colors.info;

        // Add to page
        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }
}

// Global Functions (called from HTML)
let dashboard;

window.onload = function() {
    dashboard = new NoraDashboard();
};

function showAddChannelModal() {
    dashboard.showAddChannelModal();
}

function closeModal() {
    dashboard.closeModal();
}

function addLearningChannel() {
    dashboard.addLearningChannel();
}

function refreshLearningChannels() {
    dashboard.refreshLearningChannels();
}

async function triggerReflection() {
    try {
        const response = await fetch('/api/trigger_reflection', {
            method: 'POST'
        });
        const result = await response.json();
        
        if (result.success) {
            dashboard.showNotification('خودبازبینی با موفقیت انجام شد!', 'success');
            await dashboard.updateMetacognitionStatus();
        } else {
            dashboard.showNotification(result.message, 'error');
        }
    } catch (error) {
        console.error('Error triggering reflection:', error);
        dashboard.showNotification('خطا در انجام خودبازبینی', 'error');
    }
}

async function generateEvolution() {
    try {
        const response = await fetch('/api/generate_evolution_proposal', {
            method: 'POST'
        });
        const result = await response.json();
        
        if (result.success) {
            dashboard.showNotification('پیشنهاد تکامل تولید شد!', 'success');
            await dashboard.updateMetacognitionStatus();
        } else {
            dashboard.showNotification(result.message, 'error');
        }
    } catch (error) {
        console.error('Error generating evolution:', error);
        dashboard.showNotification('خطا در تولید پیشنهاد تکامل', 'error');
    }
}
