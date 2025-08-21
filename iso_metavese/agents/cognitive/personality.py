# agents/cognitive/personality.py

import random

class Personality:
    """
    Модель личности, основанная на "Большой пятёрке".
    Отвечает за модуляцию базовых решений, придавая ИСО индивидуальность.
    """
    TRAITS = ['openness', 'conscientiousness', 'extraversion',
              'agreeableness', 'neuroticism']

    def __init__(self):
        self.traits = {trait: random.uniform(0.2, 0.8) for trait in self.TRAITS}
        self.learning_rate = random.uniform(0.1, 0.3)
        self.adaptability = random.uniform(0.4, 0.9)
        self.creativity = random.uniform(0.3, 0.7)

    def influence_behavior(self, scores):
        """
        Влияние личности на оценки привлекательности действий,
        рассчитанные "мозгом".
        """
        influenced = scores.copy()

        # Экстраверты более социальны
        if 'socialize' in influenced:
            influenced['socialize'] *= (1 + self.traits['extraversion'] * 0.5)

        # Невротики более осторожны (уменьшим приоритет рискованных действий)
        if 'explore' in influenced:
             influenced['explore'] *= (1 - self.traits['neuroticism'] * 0.3)

        # Доброжелательные более склонны помогать
        if 'help' in influenced:
            influenced['help'] *= (1 + self.traits['agreeableness'] * 0.4)

        # Открытые новому более любопытны
        if 'explore' in influenced:
            influenced['explore'] *= (1 + self.traits['openness'] * 0.6)

        return influenced