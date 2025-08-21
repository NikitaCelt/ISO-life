# agents/iso.py (ПОЛНАЯ ВЕРСИЯ ДЛЯ ЗАДАЧИ 3.1)

import random
import math
from collections import deque

# Импортируем все части "мозга"
from .cognitive.personality import Personality
from .cognitive.brain import SimpleBrain
from .cognitive.motivation import MotivationSystem
from .cognitive.communication import CommunicationSystem
from .cognitive.culture import CulturalMemory

# Полезная функция, которую используют методы ИСО
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

AGENT_ID_SEQ = 1

class ISO:
    """
    Основной класс агента ИСО (ISO).
    Содержит физиологию, характер и сложную когнитивную архитектуру
    для принятия решений и социального взаимодействия.
    """
    __slots__ = (
        "id", "x", "y", "vx", "vy", "rad", "hue", "energy", "phase",
        "reward_ema", "last_reward_ema", "b_pred", "curiosity", "sociality",
        "boldness", "patience", "noise_base", "w_food", "w_help", "w_warn",
        "link_id", "link_ttl", "last_bounce_log", "last_handshake_log",
        "last_mark_food", "last_mark_help", "last_mark_warn", "stale_t",
        # Новые атрибуты для сложного поведения
        "fear", "empathy", "mood", "social_need", "aggression",
        "relationships", "trust_level", "social_status", "personality",
        "brain", "motivation", "communication", "cultural_memory", "memory",
        "emotional_state", "learning_rate", "adaptability", "creativity",
        "current_decision",
        # Поля для обучения
        "last_decision_time", "accumulated_reward"
    )

    def __init__(self, id_, x, y, vx, vy, rad, hue, energy, cur, soc, bld, pat):
        self.id = id_
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.rad = rad
        self.hue = hue
        self.energy = energy
        self.phase = random.random()

        self.reward_ema = 0.0
        self.last_reward_ema = 0.0
        self.b_pred = 0.0

        self.curiosity = cur
        self.sociality = soc
        self.boldness = bld
        self.patience = pat
        self.noise_base = 0.03 + 0.10 * cur

        self.w_food = 0.5 + 0.8 * self.curiosity + 0.4 * self.sociality
        self.w_help = 0.3 + 0.9 * self.sociality - 0.2 * self.boldness
        self.w_warn = 0.4 + 0.8 * (1.0 - self.boldness)

        self.link_id = -1
        self.link_ttl = 0.0
        
        self.last_bounce_log = -999.0
        self.last_handshake_log = -999.0
        self.last_mark_food = -999.0
        self.last_mark_help = -999.0
        self.last_mark_warn = -999.0
        
        self.stale_t = 0.0

        # ===== НОВАЯ СИСТЕМА ЭМОЦИЙ И ПОВЕДЕНИЯ =====
        self.fear = 0.0
        self.empathy = random.uniform(0.2, 0.8)
        self.mood = random.uniform(0.4, 0.6)
        self.social_need = random.uniform(0.3, 0.7)
        self.aggression = random.uniform(0.1, 0.4)
        self.emotional_state = "calm"

        self.relationships = {}
        self.trust_level = random.uniform(0.3, 0.7)
        self.social_status = random.uniform(0.0, 1.0)

        self.personality = Personality()
        self.brain = SimpleBrain()
        self.motivation = MotivationSystem()
        self.communication = CommunicationSystem()
        self.cultural_memory = CulturalMemory()

        self.memory = deque(maxlen=20)
        self.learning_rate = random.uniform(0.1, 0.3)
        self.adaptability = random.uniform(0.4, 0.9)
        self.creativity = random.uniform(0.2, 0.6)

        self.communication.id = self.id
        self.cultural_memory.id = self.id
        self.cultural_memory.x = self.x
        self.cultural_memory.y = self.y
        
        self.current_decision = "explore"
        
        self.last_decision_time = 0.0
        self.accumulated_reward = 0.0

    def receive_message(self, message):
        trust_factor = self.communication.trust_levels.get(message['sender_id'], 0.5)
        perceived_urgency = message['intensity'] * trust_factor

        if message['style'] == "emotional": perceived_urgency *= 1.2
        elif message['style'] == "cautious": perceived_urgency *= 0.8

        if message['type'] == 'warning' and perceived_urgency > 0.6:
            self.fear = min(1.0, self.fear + perceived_urgency * 0.3)
        elif message['type'] == 'food' and perceived_urgency > 0.4:
            self.curiosity = min(1.0, self.curiosity + perceived_urgency * 0.2)
        elif message['type'] == 'help' and perceived_urgency > 0.5:
            self.empathy = min(1.0, self.empathy + perceived_urgency * 0.3)

    def update_emotions(self, dt, predator_nearness, social_contacts, novelty):
        fear_decay = 0.2 * dt
        fear_increase = predator_nearness * (1 - self.boldness) * dt * 0.5
        self.fear = clamp(self.fear - fear_decay + fear_increase, 0, 1)

        self.curiosity = clamp(0.3 + 0.7 * novelty, 0.0, 1.0)

        social_satisfaction = min(1.0, social_contacts / 3.0)
        self.social_need = clamp(0.5 - social_satisfaction * 0.3, 0.1, 1.0)
        
        energy_factor = clamp(self.energy / 70.0, 0, 1)
        social_factor = clamp(social_contacts / 5.0, 0, 1)
        self.mood = clamp(0.6 * energy_factor + 0.4 * social_factor - 0.3 * self.fear, 0, 1)

        if self.fear > 0.7: self.emotional_state = "fearful"
        elif self.curiosity > 0.6 and novelty > 0.5: self.emotional_state = "curious"
        elif self.social_need > 0.6 and social_contacts > 2: self.emotional_state = "social"
        elif self.energy > 60 and self.mood > 0.7: self.emotional_state = "excited"
        else: self.emotional_state = "calm"

    def update_relationships(self, other_iso, dt):
        if other_iso.id not in self.relationships:
            self.relationships[other_iso.id] = 0.0

        distance = math.hypot(self.x - other_iso.x, self.y - other_iso.y)
        if distance < 4.0:
            relation_change = 0.05 * dt * (1 + self.sociality * 0.5)
            self.relationships[other_iso.id] += relation_change

        if self.fear > 0.6 and other_iso.energy > 30:
            help_factor = other_iso.empathy * 0.3
            self.relationships[other_iso.id] += help_factor

        self.relationships[other_iso.id] = clamp(self.relationships[other_iso.id], -1.0, 1.0)

    def make_decision(self, environment_inputs, current_time):
        decision_inputs = {
            'food_signal': environment_inputs.get('food', 0),
            'predator_near': environment_inputs.get('predator', 0),
            'social_signal': environment_inputs.get('social', 0),
            'novelty': environment_inputs.get('novelty', 0),
            'help_signal': environment_inputs.get('help', 0),
        }
        
        current_goal = self.motivation.current_goal
        decision, scores = self.brain.decide_action(decision_inputs)
        final_scores = self.personality.influence_behavior(scores)
        
        if current_goal in final_scores:
            final_scores[current_goal] *= (1.0 + self.motivation.goal_urgency * 1.5)

        new_decision = max(final_scores, key=final_scores.get)
        
        if new_decision != self.current_decision:
            self.current_decision = new_decision
            self.last_decision_time = current_time
            self.accumulated_reward = 0.0

        return self.current_decision

    def get_steering_modifiers(self):
        """
        ИЗМЕНЕННЫЙ МЕТОД:
        Преобразует текущее решение в модификаторы для руления.
        Теперь влияет не только на феромоны, но и на скорость/случайность.
        Возвращает словарь с множителями для весов и другими параметрами.
        """
        mods = {'food': 1.0, 'help': 1.0, 'warn': 1.0, 'noise': 1.0, 'speed': 1.0}
        decision = self.current_decision
        
        if decision == 'eat':
            mods['food'] = 5.0
            mods['warn'] = 0.5
        elif decision == 'flee':
            mods['warn'] = 10.0
            mods['food'] = 0.1
            mods['help'] = 0.1
            mods['speed'] = 1.2 # Убегает быстрее
        elif decision == 'socialize':
            mods['help'] = 3.0
            mods['food'] = 0.5
            mods['speed'] = 0.8 # Двигается медленнее для общения
        elif decision == 'help':
            mods['help'] = 5.0
            mods['warn'] = 0.7
        elif decision == 'explore':
            mods['noise'] = 2.5 # Более хаотичное движение для исследования
            mods['speed'] = 1.1

        # Добавляем цель 'rest' из MotivationSystem
        if self.motivation.current_goal == 'rest':
            mods['speed'] = 0.3
            mods['noise'] = 0.2
            
        return mods

    def communicate_with_others(self, nearby_agents, current_time, logger=None):
        if current_time - self.communication.last_message_time < 1.0 / (self.sociality + 0.1):
            return

        for other in nearby_agents:
            if random.random() < 0.3 * self.sociality:
                message_type = random.choice(list(self.communication.vocabulary))
                intensity = random.uniform(0.3, 0.8) * self.mood
                emotional_content = self.mood
                
                self.communication.send_message(other, message_type, intensity, emotional_content)
                self.communication.last_message_time = current_time

                if logger and random.random() < 0.01:
                    logger.add_iso(self.id, self.hue, f"communicates with ISO-{other.id:02d}")

    def participate_in_culture(self, all_agents, current_time, logger=None):
        if len(all_agents) >= 3 and random.random() < 0.02 * self.creativity:
            ritual = self.cultural_memory.develop_ritual(all_agents, current_time)
            if ritual:
                participants = [a for a in all_agents if math.hypot(a.x - self.x, a.y - self.y) < 8.0]
                if len(participants) >= 2:
                    self.cultural_memory.add_historical_event(
                        "ritual_created",
                        ritual['intensity'],
                        [p.id for p in participants]
                    )
                    
                    if logger and random.random() < 0.1:
                        logger.add_iso(
                            self.id, self.hue,
                            f"creates {ritual['type']} ritual with {len(participants)} participants"
                        )