# agents/cognitive/communication.py

import random
import time

class CommunicationSystem:
    """
    Моделирует общение между агентами.
    Агенты могут обмениваться сообщениями, которые другие интерпретируют
    с учетом доверия и стиля отправителя.
    """
    def __init__(self):
        self.vocabulary = {"danger", "food", "help", "greet", "curious"}  # базовый словарь
        self.communication_style = random.choice(["direct", "emotional", "cautious", "expressive"])
        self.last_message_time = 0
        self.message_complexity = random.uniform(0.3, 0.8)
        self.trust_levels = {}  # id -> уровень доверия (0 до 1)
        self.id = -1  # Будет установлено при создании ИСО

    def send_message(self, receiver_iso, message_type, intensity, emotional_content):
        """
        Формирует и отправляет сообщение другому ИСО.
        Получатель (receiver_iso) сам обработает сообщение через свой метод receive_message.
        """
        message = {
            'type': message_type,  # warning, food, help, social, curious
            'intensity': intensity,
            'emotional_content': emotional_content,
            'sender_id': self.id,
            'timestamp': time.time(),
            'style': self.communication_style,
            'complexity': self.message_complexity
        }
        
        # Другой агент интерпретирует сообщение
        receiver_iso.receive_message(message)