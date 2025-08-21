# rendering/ansi_helpers.py

import config

# ========= ANSI-коды управления терминалом =========
RESET = "\033[0m"
HIDE = "\033[?25l"
SHOW = "\033[?25h"
CLEAR = "\033[2J"
HOME = "\033[H"
ALT_ON = "\033[?1049h"
ALT_OFF = "\033[?1049l"
ESC = '\x1b'

# ========= Функции для работы с цветом =========
def rgb_code(r, g, b):
    """Генерирует ANSI-код для 24-битного цвета."""
    return f"\033[38;2;{int(r)};{int(g)};{int(b)}m"

def rgb(r, g, b):
    """Возвращает ANSI-код, если цвета включены, иначе пустую строку."""
    return "" if not config.ENABLE_COLOR else rgb_code(r, g, b)

def rgb256(r, g, b):
    """Преобразует RGB в приближенный 256-цветный ANSI-код."""
    def q(v):
        return int(round(v / 255 * 5))
    return f"\033[38;5;{16 + 36 * q(r) + 6 * q(g) + q(b)}m"

# ========= Функции для безопасной работы с ANSI-строками =========
def _ansi_strip(s: str) -> str:
    """Удаляет все ANSI-последовательности из строки."""
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == ESC and i + 1 < n and s[i + 1] == '[':
            i += 2
            while i < n and not ('@' <= s[i] <= '~'):
                i += 1
            if i < n:
                i += 1
        else:
            out.append(s[i])
            i += 1
    return ''.join(out)

def visible_len(s: str) -> int:
    """Возвращает видимую длину строки (без учета ANSI-кодов)."""
    return len(_ansi_strip(s))

def crop_ansi(s: str, w: int) -> str:
    """Обрезает строку до видимой длины w, сохраняя ANSI-коды."""
    if w <= 0:
        return ''
    out = []
    vis = 0
    i = 0
    n = len(s)
    while i < n and vis < w:
        if s[i] == ESC and i + 1 < n and s[i + 1] == '[':
            j = i + 2
            while j < n and not ('@' <= s[j] <= '~'):
                j += 1
            if j < n:
                j += 1
            out.append(s[i:j])
            i = j
            continue
        out.append(s[i])
        vis += 1
        i += 1
    return ''.join(out)

def fit_line(s: str, w: int) -> str:
    """Обрезает или дополняет строку пробелами до видимой длины w."""
    c = crop_ansi(s, w)
    pad = max(0, w - visible_len(c))
    return c + (' ' * pad)