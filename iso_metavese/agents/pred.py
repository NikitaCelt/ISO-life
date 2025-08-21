# agents/pred.py

import random

PRED_ID_SEQ = 1

class PRED:
    """
    Класс Хищника (Predator).
    Обладает простым поведением, нацеленным на поиск и атаку ИСО.
    """
    __slots__ = (
        "id", "x", "y", "vx", "vy", "rad", "energy", "aggr", "stealth",
        "stamina", "state", "target_id", "cooldown", "ambush_t",
        "last_hit_log", "e_max"
    )

    def __init__(self, id_, x, y, vx, vy, rad, energy, aggr, stealth, stamina):
        self.id = id_
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.rad = rad
        self.energy = energy
        self.e_max = max(80.0, energy)
        self.aggr = aggr
        self.stealth = stealth
        self.stamina = stamina
        self.state = "prowl"
        self.target_id = -1
        self.cooldown = 0.0
        self.ambush_t = random.uniform(0.5, 2.5)
        self.last_hit_log = -999.0