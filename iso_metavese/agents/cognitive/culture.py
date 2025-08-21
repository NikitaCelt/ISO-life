# agents/cognitive/culture.py

import random
import time
from collections import deque

class CulturalMemory:
    """
    Система, позволяющая агентам формировать "традиции", "ритуалы"
    и записывать "исторические события", создавая простейшие формы культуры.
    """
    def __init__(self):
        self.traditions = []  # коллективные ритуалы
        self.collective_memory = deque(maxlen=100)  # важные события группы
        self.group_identity = random.uniform(0.1, 0.9)  # сила групповой идентичности
        self.history = []  # исторические события
        self.ritual_intensity = random.uniform(0.3, 0.7)
        self.id = -1  # Будет установлено при создании ИСО
        self.x = 0    # Будет установлено
        self.y = 0    # Будет установлено

    def develop_ritual(self, agents, current_time):
        """
        Создает новый ритуал при определенных условиях (например, при
        достаточном количестве агентов поблизости).
        """
        if len(agents) > 3 and random.random() < 0.02:
            ritual_type = random.choice(["greeting", "warning", "celebration", "mourning", "cooperation"])
            new_ritual = {
                'type': ritual_type,
                'pattern': self._generate_ritual_pattern(),
                'meaning': random.uniform(0.1, 0.9),
                'created_by': self.id,
                'adoption_rate': 0.1,
                'intensity': self.ritual_intensity,
                'created_at': current_time
            }
            self.traditions.append(new_ritual)
            return new_ritual
        return None

    def _generate_ritual_pattern(self):
        """Генерирует случайный паттерн для ритуала."""
        patterns = ["circular", "linear", "random", "symmetric", "wave"]
        return random.choice(patterns)

    def add_historical_event(self, event_type, significance, participants):
        """
        Добавляет важное событие в историческую память группы.
        """
        event = {
            'type': event_type,
            'significance': significance,
            'participants': participants,
            'timestamp': time.time(),
            'location': (self.x, self.y)
        }
        self.history.append(event)
        if len(self.history) > 50:  # Ограничение размера истории
            self.history.pop(0)