# agents/cognitive/motivation.py

import random

class MotivationSystem:
    """
    Система, управляющая долгосрочными целями агента.
    Определяет текущую цель (например, поиск еды, общение) на основе
    внутренних потребностей (энергия, социальная нужда) и "жизненной цели".
    """
    def __init__(self):
        self.current_goal = "explore"  # explore, eat, socialize, rest, reproduce, help
        self.goal_urgency = 0.0  # срочность цели (0-1)
        self.life_purpose = random.choice(["knowledge", "power", "community", "survival", "harmony"])
        self.goal_persistence = random.uniform(0.3, 0.8)  # настойчивость в достижении цели
        self.last_goal_change = 0.0

    def update_goals(self, iso, current_time):
        """Динамически меняет цели на основе потребностей и жизненных установок."""
        needs = {
            'eat': max(0, 1 - iso.energy / 30.0),
            'socialize': iso.social_need,
            'explore': iso.curiosity * 0.7,
            'rest': max(0, 0.8 - iso.energy / 100.0),
            'help': iso.empathy * 0.5,
            'reproduce': max(0, (iso.energy - 50) / 20.0) if iso.energy > 50 else 0
        }

        # Влияние жизненной цели на приоритеты
        if self.life_purpose == "knowledge":
            needs['explore'] *= 1.5
        elif self.life_purpose == "community":
            needs['socialize'] *= 1.8
            needs['help'] *= 1.3
        elif self.life_purpose == "power":
            needs['reproduce'] *= 1.4
        elif self.life_purpose == "harmony":
            needs['rest'] *= 1.2
            needs['socialize'] *= 1.1

        # Учет эмоционального состояния
        needs['explore'] *= (0.5 + iso.curiosity)
        needs['socialize'] *= (0.6 + iso.social_need)
        needs['help'] *= (0.7 + iso.empathy)

        # Минимальное время между сменами целей, чтобы избежать "дергания"
        if current_time - self.last_goal_change < 2.0 * self.goal_persistence:
            return

        new_goal = max(needs, key=needs.get)
        
        # Цель меняется, только если новая потребность значительно сильнее текущей
        if new_goal != self.current_goal and needs[new_goal] > (needs.get(self.current_goal, 0) + 0.1):
            self.current_goal = new_goal
            self.goal_urgency = needs[new_goal]
            self.last_goal_change = current_time