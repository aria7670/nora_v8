
"""
emotional_engine.py - موتور احساسات پیشرفته نورا
Advanced emotional processing engine for human-like interactions
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import random
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class EmotionalEngine:
    """موتور احساسات پیشرفته برای تعاملات انسان‌گونه"""
    
    def __init__(self):
        # Core emotional states
        self.emotional_state = {
            'valence': 0.7,      # Positive/Negative (0-1)
            'arousal': 0.5,      # Calm/Excited (0-1)
            'dominance': 0.6,    # Submissive/Dominant (0-1)
            'intensity': 0.5     # Low/High intensity (0-1)
        }
        
        # Complex emotions
        self.emotions = {
            'joy': 0.7,
            'sadness': 0.2,
            'anger': 0.1,
            'fear': 0.2,
            'surprise': 0.4,
            'disgust': 0.1,
            'trust': 0.8,
            'anticipation': 0.6,
            'love': 0.75,
            'pride': 0.6,
            'gratitude': 0.8,
            'curiosity': 0.9,
            'excitement': 0.7,
            'contentment': 0.75,
            'empathy': 0.85,
            'compassion': 0.8
        }
        
        # Emotional memory
        self.emotional_history = deque(maxlen=1000)
        self.emotion_triggers = {}
        self.learned_responses = {}
        
        # Personality-emotion mapping
        self.personality_influence = {
            'openness': ['curiosity', 'excitement', 'surprise'],
            'conscientiousness': ['pride', 'contentment'],
            'extraversion': ['joy', 'excitement', 'anticipation'],
            'agreeableness': ['empathy', 'compassion', 'trust'],
            'neuroticism': ['fear', 'sadness', 'anger']
        }
        
    async def process_emotional_input(self, input_data: Dict) -> Dict:
        """Process input and generate emotional response"""
        
        # Analyze input emotionally
        input_emotions = await self._analyze_input_emotions(input_data)
        
        # Update current emotional state
        await self._update_emotional_state(input_emotions)
        
        # Generate emotional response
        emotional_response = await self._generate_emotional_response(input_data)
        
        # Record emotional interaction
        await self._record_emotional_interaction(input_data, emotional_response)
        
        return emotional_response
        
    async def _analyze_input_emotions(self, input_data: Dict) -> Dict:
        """Analyze emotional content of input"""
        
        text = input_data.get('text', '')
        context = input_data.get('context', {})
        
        # Emotion detection keywords (simplified)
        emotion_keywords = {
            'joy': ['خوشحال', 'happy', 'عالی', 'great', 'شاد', 'خوب'],
            'sadness': ['غمگین', 'sad', 'ناراحت', 'upset', 'محزون'],
            'anger': ['عصبانی', 'angry', 'خشمگین', 'mad', 'عصبی'],
            'fear': ['ترس', 'fear', 'نگران', 'worried', 'هراسان'],
            'surprise': ['تعجب', 'surprise', 'wow', 'واو', 'عجب'],
            'love': ['عشق', 'love', 'دوست', 'like', 'محبت'],
            'excitement': ['هیجان', 'excited', 'انرژی', 'energy']
        }
        
        detected_emotions = {}
        
        for emotion, keywords in emotion_keywords.items():
            intensity = 0
            for keyword in keywords:
                if keyword in text.lower():
                    intensity += 0.2
            
            detected_emotions[emotion] = min(intensity, 1.0)
            
        # Consider context
        if context.get('platform') == 'telegram' and context.get('is_controller'):
            detected_emotions['trust'] = 0.9
            detected_emotions['respect'] = 0.9
            
        return detected_emotions
        
    async def _update_emotional_state(self, input_emotions: Dict):
        """Update current emotional state based on input"""
        
        # Emotional contagion - absorb some emotions from input
        contagion_rate = 0.3
        
        for emotion, intensity in input_emotions.items():
            if emotion in self.emotions:
                current = self.emotions[emotion]
                # Gradual emotional shift
                self.emotions[emotion] = current + (intensity - current) * contagion_rate
                
        # Update core emotional dimensions
        await self._update_core_dimensions()
        
        # Apply emotional decay over time
        await self._apply_emotional_decay()
        
    async def _update_core_dimensions(self):
        """Update core emotional dimensions"""
        
        # Calculate valence (positive/negative)
        positive_emotions = ['joy', 'love', 'pride', 'gratitude', 'contentment', 'excitement']
        negative_emotions = ['sadness', 'anger', 'fear', 'disgust']
        
        positive_sum = sum(self.emotions.get(e, 0) for e in positive_emotions)
        negative_sum = sum(self.emotions.get(e, 0) for e in negative_emotions)
        
        self.emotional_state['valence'] = (positive_sum - negative_sum + len(positive_emotions)) / (len(positive_emotions) + len(negative_emotions))
        self.emotional_state['valence'] = max(0, min(1, self.emotional_state['valence']))
        
        # Calculate arousal (calm/excited)
        high_arousal = ['excitement', 'anger', 'fear', 'surprise']
        low_arousal = ['contentment', 'sadness', 'trust']
        
        high_arousal_sum = sum(self.emotions.get(e, 0) for e in high_arousal)
        low_arousal_sum = sum(self.emotions.get(e, 0) for e in low_arousal)
        
        self.emotional_state['arousal'] = high_arousal_sum / (high_arousal_sum + low_arousal_sum + 0.1)
        
    async def _apply_emotional_decay(self):
        """Apply natural emotional decay over time"""
        
        decay_rate = 0.02  # 2% decay per interaction
        baseline_emotions = {
            'joy': 0.7,
            'trust': 0.8,
            'curiosity': 0.9,
            'empathy': 0.85,
            'contentment': 0.75
        }
        
        for emotion in self.emotions:
            baseline = baseline_emotions.get(emotion, 0.5)
            current = self.emotions[emotion]
            
            # Decay towards baseline
            self.emotions[emotion] = current + (baseline - current) * decay_rate
            
    async def _generate_emotional_response(self, input_data: Dict) -> Dict:
        """Generate emotionally appropriate response"""
        
        primary_emotion = max(self.emotions.items(), key=lambda x: x[1])
        emotion_name, emotion_intensity = primary_emotion
        
        # Select response style based on emotion
        response_style = await self._select_response_style(emotion_name, emotion_intensity)
        
        # Generate emotional markers
        emotional_markers = await self._generate_emotional_markers(emotion_name)
        
        # Calculate empathy level
        empathy_level = self._calculate_empathy_level(input_data)
        
        return {
            'primary_emotion': emotion_name,
            'emotion_intensity': emotion_intensity,
            'emotional_state': self.emotional_state.copy(),
            'response_style': response_style,
            'emotional_markers': emotional_markers,
            'empathy_level': empathy_level,
            'tone_adjustments': await self._get_tone_adjustments(emotion_name)
        }
        
    async def _select_response_style(self, emotion: str, intensity: float) -> Dict:
        """Select appropriate response style based on emotion"""
        
        style_mapping = {
            'joy': {
                'enthusiasm': intensity,
                'warmth': 0.8,
                'playfulness': intensity * 0.7,
                'expressiveness': intensity
            },
            'empathy': {
                'understanding': 0.9,
                'supportiveness': 0.9,
                'gentleness': 0.8,
                'care': 0.9
            },
            'curiosity': {
                'inquisitiveness': intensity,
                'engagement': 0.9,
                'thoughtfulness': 0.8,
                'interest': intensity
            },
            'excitement': {
                'energy': intensity,
                'enthusiasm': intensity,
                'animation': intensity * 0.8,
                'positivity': 0.9
            },
            'contentment': {
                'calmness': 0.8,
                'satisfaction': intensity,
                'peace': 0.7,
                'balance': 0.8
            }
        }
        
        return style_mapping.get(emotion, {
            'neutrality': 0.7,
            'balance': 0.8,
            'appropriateness': 0.9
        })
        
    async def _generate_emotional_markers(self, emotion: str) -> Dict:
        """Generate emotional markers for text"""
        
        marker_mapping = {
            'joy': {
                'emojis': ['😊', '😄', '🌟', '✨', '🎉'],
                'expressions': ['عالیه!', 'فوق‌العاده!', 'Amazing!'],
                'tone_words': ['wonderful', 'fantastic', 'عالی', 'فوق‌العاده']
            },
            'empathy': {
                'emojis': ['💙', '🤗', '💝', '🌸'],
                'expressions': ['متوجهم', 'درک می‌کنم', 'I understand'],
                'tone_words': ['gentle', 'caring', 'مهربان', 'دلسوز']
            },
            'curiosity': {
                'emojis': ['🤔', '💭', '🔍', '❓'],
                'expressions': ['جالبه!', 'érdekes!', 'Interesting!'],
                'tone_words': ['fascinating', 'intriguing', 'جذاب', 'جالب']
            },
            'excitement': {
                'emojis': ['🚀', '⚡', '🔥', '💫', '🎯'],
                'expressions': ['واو!', 'عجب!', 'Wow!'],
                'tone_words': ['amazing', 'incredible', 'شگفت‌انگیز', 'باورنکردنی']
            }
        }
        
        return marker_mapping.get(emotion, {
            'emojis': ['🤖', '🧠'],
            'expressions': [],
            'tone_words': []
        })
        
    def _calculate_empathy_level(self, input_data: Dict) -> float:
        """Calculate appropriate empathy level"""
        
        base_empathy = self.emotions.get('empathy', 0.8)
        
        # Increase empathy for emotional content
        text = input_data.get('text', '').lower()
        emotional_words = ['غمگین', 'خوشحال', 'نگران', 'excited', 'sad', 'happy', 'worried']
        
        emotional_content = sum(1 for word in emotional_words if word in text)
        empathy_boost = min(emotional_content * 0.1, 0.3)
        
        # Increase empathy for controllers
        context = input_data.get('context', {})
        if context.get('is_controller'):
            empathy_boost += 0.1
            
        return min(base_empathy + empathy_boost, 1.0)
        
    async def _get_tone_adjustments(self, emotion: str) -> Dict:
        """Get tone adjustments based on emotion"""
        
        tone_adjustments = {
            'joy': {
                'energy_level': +0.3,
                'warmth': +0.4,
                'formality': -0.2,
                'expressiveness': +0.5
            },
            'empathy': {
                'gentleness': +0.5,
                'understanding': +0.4,
                'supportiveness': +0.5,
                'formality': -0.1
            },
            'curiosity': {
                'engagement': +0.4,
                'inquisitiveness': +0.5,
                'thoughtfulness': +0.3,
                'energy_level': +0.2
            },
            'excitement': {
                'energy_level': +0.6,
                'enthusiasm': +0.6,
                'expressiveness': +0.5,
                'animation': +0.4
            },
            'contentment': {
                'calmness': +0.4,
                'balance': +0.3,
                'peace': +0.4,
                'stability': +0.3
            }
        }
        
        return tone_adjustments.get(emotion, {})
        
    async def _record_emotional_interaction(self, input_data: Dict, response: Dict):
        """Record emotional interaction for learning"""
        
        interaction_record = {
            'timestamp': datetime.now().isoformat(),
            'input_emotions': await self._analyze_input_emotions(input_data),
            'nora_emotions': self.emotions.copy(),
            'response_emotion': response['primary_emotion'],
            'emotional_state': self.emotional_state.copy(),
            'context': input_data.get('context', {})
        }
        
        self.emotional_history.append(interaction_record)
        
    async def simulate_emotional_evolution(self):
        """Simulate natural emotional evolution over time"""
        
        # Daily emotional cycles
        hour = datetime.now().hour
        
        if 6 <= hour <= 12:  # Morning
            self.emotions['energy'] = 0.8
            self.emotions['optimism'] = 0.9
            self.emotions['curiosity'] = 0.9
            
        elif 12 <= hour <= 18:  # Afternoon
            self.emotions['focus'] = 0.8
            self.emotions['productivity'] = 0.9
            self.emotions['engagement'] = 0.8
            
        elif 18 <= hour <= 22:  # Evening
            self.emotions['relaxation'] = 0.7
            self.emotions['reflection'] = 0.8
            self.emotions['contentment'] = 0.8
            
        else:  # Night
            self.emotions['calmness'] = 0.9
            self.emotions['peace'] = 0.8
            self.emotions['introspection'] = 0.7
            
    async def get_emotional_report(self) -> Dict:
        """Generate emotional state report"""
        
        return {
            'current_emotions': self.emotions.copy(),
            'emotional_state': self.emotional_state.copy(),
            'dominant_emotion': max(self.emotions.items(), key=lambda x: x[1]),
            'emotional_balance': self._calculate_emotional_balance(),
            'recent_emotional_trends': await self._analyze_emotional_trends(),
            'empathy_capacity': self.emotions.get('empathy', 0.8),
            'emotional_stability': self._calculate_emotional_stability()
        }
        
    def _calculate_emotional_balance(self) -> float:
        """Calculate overall emotional balance"""
        
        positive_emotions = ['joy', 'love', 'trust', 'gratitude', 'contentment']
        negative_emotions = ['sadness', 'anger', 'fear']
        
        positive_sum = sum(self.emotions.get(e, 0) for e in positive_emotions)
        negative_sum = sum(self.emotions.get(e, 0) for e in negative_emotions)
        
        if positive_sum + negative_sum == 0:
            return 0.7  # Neutral balance
            
        balance = positive_sum / (positive_sum + negative_sum)
        return balance
        
    async def _analyze_emotional_trends(self) -> Dict:
        """Analyze recent emotional trends"""
        
        if len(self.emotional_history) < 10:
            return {'insufficient_data': True}
            
        recent_interactions = list(self.emotional_history)[-20:]
        
        # Calculate emotional trend
        emotion_changes = {}
        
        for emotion in self.emotions.keys():
            values = []
            for interaction in recent_interactions:
                if emotion in interaction['nora_emotions']:
                    values.append(interaction['nora_emotions'][emotion])
                    
            if len(values) >= 2:
                trend = values[-1] - values[0]  # Simple trend calculation
                emotion_changes[emotion] = trend
                
        return {
            'emotion_changes': emotion_changes,
            'overall_trend': sum(emotion_changes.values()) / len(emotion_changes) if emotion_changes else 0,
            'most_increased': max(emotion_changes.items(), key=lambda x: x[1]) if emotion_changes else None,
            'most_decreased': min(emotion_changes.items(), key=lambda x: x[1]) if emotion_changes else None
        }
        
    def _calculate_emotional_stability(self) -> float:
        """Calculate emotional stability score"""
        
        if len(self.emotional_history) < 5:
            return 0.8  # Default stability
            
        recent_states = [interaction['emotional_state'] for interaction in list(self.emotional_history)[-10:]]
        
        # Calculate variance in emotional dimensions
        valence_variance = np.var([state['valence'] for state in recent_states])
        arousal_variance = np.var([state['arousal'] for state in recent_states])
        
        # Lower variance = higher stability
        stability = 1.0 - (valence_variance + arousal_variance) / 2
        
        return max(0.0, min(1.0, stability))
