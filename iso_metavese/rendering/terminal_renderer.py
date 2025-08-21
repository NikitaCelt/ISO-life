# rendering/terminal_renderer.py

import os
import sys
import shutil
import math

import config
from .ansi_helpers import (
    RESET, HIDE, SHOW, CLEAR, HOME, rgb, rgb_code, rgb256
)

# ========= LUT (Look-Up Table) для ускорения рендеринга =========
_LUT_T = None
LUT_TRUE_ISO = []
LUT_256_ISO = []
LUT_TRUE_MIX = []
LUT_256_MIX = []

def tone_map(v, k=1.3):
    return 1.0 - math.exp(-k * max(0.0, float(v)))

def lerp(a, b, t):
    return a + (b - a) * t

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def lerp3(c1, c2, t):
    t = clamp(t, 0, 1)
    return (
        int(lerp(c1[0], c2[0], t)),
        int(lerp(c1[1], c2[1], t)),
        int(lerp(c1[2], c2[2], t))
    )

def mix3(a, b, k):
    return (
        int(a[0] * (1 - k) + b[0] * k),
        int(a[1] * (1 - k) + b[1] * k),
        int(a[2] * (1 - k) + b[2] * k)
    )

def pal_iso(t):
    t = clamp(t, 0, 1)
    if t < 0.55: return lerp3(config.ENCOM_DEEP, config.ENCOM_BLUE, t / 0.55)
    if t < 0.85: return lerp3(config.ENCOM_BLUE, config.ENCOM_CYAN, (t - 0.55) / 0.30)
    if config.AMBER_ACCENT and t > 0.98: return lerp3(config.ENCOM_WHITE, config.ENCOM_AMBER, (t - 0.98) / 0.02)
    return lerp3(config.ENCOM_CYAN, config.ENCOM_WHITE, (t - 0.85) / 0.15)

def pal_pred(t):
    t = clamp(t, 0, 1)
    if t < 0.6: return lerp3(config.PRED_DEEP, config.PRED_VIOLET, t / 0.6)
    if t < 0.9: return lerp3(config.PRED_VIOLET, config.PRED_MAGENTA, (t - 0.6) / 0.3)
    return lerp3(config.PRED_MAGENTA, config.ENCOM_WHITE, (t - 0.9) / 0.1)

def char_for(t):
    return config.ASCII[int(clamp(t, 0, 1) * (len(config.ASCII) - 1) + 1e-6)]


def rebuild_luts(levels: int, mix_levels: int):
    global _LUT_T, LUT_TRUE_ISO, LUT_256_ISO, LUT_TRUE_MIX, LUT_256_MIX
    _LUT_T = (levels, mix_levels)
    LUT_TRUE_ISO = [None] * levels
    LUT_256_ISO = [None] * levels
    LUT_TRUE_MIX = [[None] * mix_levels for _ in range(levels)]
    LUT_256_MIX = [[None] * mix_levels for _ in range(levels)]
    for i in range(levels):
        t = i / max(1, levels - 1)
        col_iso = pal_iso(t)
        LUT_TRUE_ISO[i] = rgb_code(*col_iso)
        LUT_256_ISO[i] = rgb256(*col_iso)
        pred_col = pal_pred(t)
        for j in range(mix_levels):
            r = j / max(1, mix_levels - 1)
            col_mix = mix3(col_iso, pred_col, r)
            LUT_TRUE_MIX[i][j] = rgb_code(*col_mix)
            LUT_256_MIX[i][j] = rgb256(*col_mix)

def ensure_luts():
    if _LUT_T != (config.COLOR_LEVELS, config.MIX_LEVELS) or not LUT_TRUE_ISO:
        rebuild_luts(config.COLOR_LEVELS, config.MIX_LEVELS)

def render_row_mix_fast(iso_row, pred_row, scale):
    ensure_luts()
    lut_iso = LUT_256_ISO if config.USE_ANSI256 else LUT_TRUE_ISO
    lut_mix = LUT_256_MIX if config.USE_ANSI256 else LUT_TRUE_MIX
    out = []
    prev = None
    ap = out.append
    for ib, ip in zip(iso_row, pred_row):
        tot = ib + ip
        if tot <= 1e-12:
            if config.ENABLE_COLOR:
                code = lut_iso[0]
                if code is not prev: ap(code); prev = code
                ap(' ')
            else:
                ap(' ')
            continue
        t = tone_map(tot / scale)
        qi = int(t * (config.COLOR_LEVELS - 1) + 0.5)
        qi = max(0, min(config.COLOR_LEVELS - 1, qi))
        ratio = clamp((ip / (tot + 1e-6)) * 1.3, 0.0, 1.0)
        qj = int(ratio * (config.MIX_LEVELS - 1) + 0.5)
        if config.ENABLE_COLOR:
            code = lut_mix[qi][qj] if qj > 0 else lut_iso[qi]
            if code is not prev: ap(code); prev = code
            ap(char_for(t))
        else:
            ch = 'X' if ratio >= 0.25 else char_for(t)
            ap(ch)
    return "".join(out)


# ========= Классы для работы с терминалом =========

class NeonTerm:
    """Контекстный менеджер для инициализации и очистки терминала."""
    def __enter__(self):
        if os.name == 'nt':
            try: os.system('')
            except: pass
        sys.stdout.write(HIDE + CLEAR + HOME)
        sys.stdout.flush()
        return self

    def __exit__(self, *_):
        sys.stdout.write((RESET if config.ENABLE_COLOR else "") + SHOW)
        sys.stdout.flush()

class KeyPoller:
    """Неблокирующий опрос клавиатуры для Windows и Unix."""
    def __init__(self):
        self.is_win = (os.name == 'nt')
        if not self.is_win:
            import termios, tty, fcntl
            self.termios, self.tty, self.fcntl = termios, tty, fcntl

    def __enter__(self):
        if self.is_win: return self
        self.fd = sys.stdin.fileno()
        self.old_attr = self.termios.tcgetattr(self.fd)
        na = self.termios.tcgetattr(self.fd)
        na[3] = na[3] & ~(self.termios.ICANON | self.termios.ECHO)
        self.termios.tcsetattr(self.fd, self.termios.TCSANOW, na)
        self.old_flags = self.fcntl.fcntl(self.fd, self.fcntl.F_GETFL)
        self.fcntl.fcntl(self.fd, self.fcntl.F_SETFL, self.old_flags | os.O_NONBLOCK)
        return self

    def __exit__(self, *_):
        if self.is_win: return
        self.termios.tcsetattr(self.fd, self.termios.TCSANOW, self.old_attr)
        self.fcntl.fcntl(self.fd, self.fcntl.F_SETFL, self.old_flags)

    def poll(self):
        if self.is_win:
            try:
                import msvcrt
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ('\x00', '\xe0'):
                        _ = msvcrt.getwch()
                        return None
                    return ch
            except Exception: pass
            return None
        else:
            try: return sys.stdin.read(1)
            except Exception: return None

class SmartRenderer:
    """Обновляет только измененные строки терминала для гладкого вывода."""
    def __init__(self):
        self.prev_lines = []
        self.inited = False

    def render(self, lines):
        if not self.inited:
            sys.stdout.write(HOME + "\n".join(lines))
            sys.stdout.flush()
            self.prev_lines = list(lines)
            self.inited = True
            return

        max_lines = max(len(lines), len(self.prev_lines))
        for i in range(max_lines):
            new = lines[i] if i < len(lines) else ""
            old = self.prev_lines[i] if i < len(self.prev_lines) else ""
            if new != old:
                sys.stdout.write(f"\033[{i + 1};1H{new}\033[K")
        
        if config.ENABLE_COLOR:
            sys.stdout.write(RESET)
        sys.stdout.flush()
        self.prev_lines = list(lines)

def print_frame(s: str):
    """Выводит одну строку в начало терминала."""
    sys.stdout.write(HOME + s + (RESET if config.ENABLE_COLOR else ""))
    sys.stdout.flush()