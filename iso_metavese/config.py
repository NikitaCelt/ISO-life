# config.py

"""
Глобальные настройки и константы для симуляции ISO-Life.
"""
import os

def detect_fancy_borders() -> bool:
    """Определяет, поддерживает ли терминал расширенные символы для рамок."""
    if os.name != 'nt': return True
    if os.environ.get('WT_SESSION'): return True
    if os.environ.get('ANSICON'): return True
    if os.environ.get('ConEmuANSI') == 'ON': return True
    return False

# ========= Опциональное ускорение =========
TRY_NUMPY = True
try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False


# ========= Сетевые эндпоинты =========
LOG_HOST = "127.0.0.1"
LOG_PORT = 51888
STATS_PORT = 51889
WEB_FRAME_PORT = 51890
AGENTS_DATA_PORT = 51891

# ========= Порты веб-моста (HTTP+WS) =========
WS_PORT = 51900
HTTP_PORT = 51901

# ========= Настройки рендеринга в терминале =========
ENABLE_COLOR = True
FANCY_BORDERS = detect_fancy_borders()

# Ускорение цветового вывода
FAST_COLOR   = True
USE_ANSI256  = False
COLOR_LEVELS = 24    # Диапазон: 8..64
MIX_LEVELS   = 5     # Диапазон: 3..7, уровни смешения с хищником
AMBER_ACCENT = False

# Умный рендеринг (обновляет только измененные строки)
SMART_DIFF  = True

# ========= Настройки поведения агентов =========
PHERO_MULTI = True
GLITCH_ON   = True
COOP_SPLIT  = True
NOVELTY_ON  = False

# ========= Палитры =========
ENCOM_DEEP=(5,15,25); ENCOM_BLUE=(0,110,220); ENCOM_CYAN=(0,210,255); ENCOM_WHITE=(240,250,255); ENCOM_AMBER=(255,190,40)
PRED_DEEP=(20,5,30);  PRED_VIOLET=(200,0,255); PRED_MAGENTA=(255,60,180)
RED=(255,80,80); GREEN=(60,220,120)

# ========= Символы для рендеринга =========
ASCII=" .:-=+*#%@"
SPARKS=" ▂▃▄▅▆▇█"