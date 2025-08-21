# agents/cognitive/brain.py

import random
from collections import deque

class SimpleBrain:
    """
    Упрощенная нейросетевая модель для принятия решений.
    Взвешивает различные стимулы на основе внутренних "весов приоритетов"
    и выбирает наиболее подходящее действие.
    """
    def __init__(self):
        self.weights = {
            'food_priority': random.uniform(0.5, 1.5),
            'safety_priority': random.uniform(0.8, 1.2),
            'social_priority': random.uniform(0.3, 1.0),
            'exploration_urge': random.uniform(0.4, 1.6),
            'help_priority': random.uniform(0.2, 0.8)
        }
        self.decision_history = deque(maxlen=10)
        self.learning_rate = 0.1

    def decide_action(self, inputs):
        """Принимает решение на основе входных данных от сенсоров ИСО."""
        score_food = inputs.get('food_signal', 0) * self.weights['food_priority']
        score_safety = (1 - inputs.get('predator_near', 0)) * self.weights['safety_priority']
        score_social = inputs.get('social_signal', 0) * self.weights['social_priority']
        score_explore = inputs.get('novelty', 0) * self.weights['exploration_urge']
        score_help = inputs.get('help_signal', 0) * self.weights['help_priority']

        scores = {
            'eat': score_food,
            'flee': score_safety,
            'socialize': score_social,
            'explore': score_explore,
            'help': score_help
        }

        # Выбор действия с максимальным score
        decision = max(scores, key=scores.get)
        self.decision_history.append(decision)
        return decision, scores

    def learn_from_experience(self, reward, decision):
        """
        Обучение на основе полученного вознаграждения.
        Пока не используется, но является заготовкой для будущего развития.
        """
        key_map = {
            'eat': 'food_priority',
            'flee': 'safety_priority',
            'socialize': 'social_priority',
            'explore': 'exploration_urge',
            'help': 'help_priority'
        }
        
        weight_to_update = key_map.get(decision)
        if weight_to_update in self.weights:
            # Усиливаем или ослабляем вес в зависимости от вознаграждения
            self.weights[weight_to_update] += self.learning_rate * reward
            
            # Ограничиваем веса, чтобы избежать "зацикливания"
            self.weights[weight_to_update] = max(0.1, self.weights[weight_to_update])

            # Можно добавить нормализацию, чтобы сумма весов была постоянной
            # total = sum(self.weights.values())
            # if total > 0:
            #     for key in self.weights:
            #         self.weights[key] /= total