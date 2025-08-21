# main.py

import os
import sys
import time
import shutil

# --- Импортируем наши модули ---
# Конфигурация
import config

# Основная логика симуляции
from core.simulation import ISOLife
from core.web_bridge import run_web_bridge, launch_web_viewer

# Утилиты для рендеринга и ввода
from rendering.terminal_renderer import (
    NeonTerm, KeyPoller, SmartRenderer, rebuild_luts, render_row_mix_fast
)
from rendering.ansi_helpers import (
    rgb, RESET, fit_line
)

# Импортируем UDP-логгер отдельно, так как он используется в `run`
from core.udp_logger import UDPLogger

# --- Вспомогательные функции, специфичные для запуска ---

def get_size():
    """Получает размер терминала."""
    try:
        sz = shutil.get_terminal_size((100, 30))
        return max(60, sz.columns), max(24, sz.lines)
    except:
        return 100, 30

def current_fps_cap():
    """Определяет лимит FPS для цикла."""
    if not config.ENABLE_COLOR:
        return 60.0
    return 45.0 if config.FAST_COLOR else 30.0

# ========= Главный цикл симуляции =========
def run():
    """Основная функция, запускающая симуляцию в терминале."""
    cols, rows = get_size()
    w = max(60, min(160, cols - 2))
    h = max(20, min(rows - 6, 60))

    # Инициализация утилит
    logger = UDPLogger(config.LOG_HOST, config.LOG_PORT, config.STATS_PORT)
    web_streamer = None  # WebStreamer будет создан внутри ISOLife
    
    # Создаем экземпляр симуляции
    sim = ISOLife(w, h, agents=18, logger=logger, use_numpy=config.NP_AVAILABLE)
    
    sim_speed = 1.0
    paused = False
    show_help = False
    
    t_prev = time.perf_counter()
    fps_dt = 0.0
    fps_frames = 0
    fps_val = 0.0
    
    smart = SmartRenderer()

    with NeonTerm(), KeyPoller() as keys:
        while True:
            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            
            if not paused:
                sim.step(min(dt * sim_speed, 0.08))

            # --- Рендеринг кадра ---
            # Эта логика раньше была внутри ISOLife.render(), теперь она здесь,
            # чтобы класс симуляции не занимался отрисовкой.
            frame_lines = sim.render_to_lines(
                render_func=render_row_mix_fast
            )
            
            # --- Статус-бар ---
            status_parts = [
                f"{rgb(*config.ENCOM_CYAN)}ISO-LIFE By C3Dex × NeonMuse ",
                f"{rgb(*config.ENCOM_AMBER)}ISOs:{len(sim.agents)} ",
                f"{rgb(*config.PRED_VIOLET)}PREDs:{len(sim.preds)} ",
                f"{rgb(*config.ENCOM_CYAN)}spd:{sim_speed:.2f} ",
                f"{rgb(*config.ENCOM_AMBER)}beams:{'on' if sim.beams_enabled else 'off'} ",
                f"{rgb(*config.ENCOM_CYAN)}glow:{sim.glow_gain:.2f} ",
                f"{rgb(*config.ENCOM_CYAN)}fast:{'on' if config.FAST_COLOR else 'off'} ",
                f"{rgb(*config.ENCOM_CYAN)}256:{'on' if config.USE_ANSI256 else 'off'} ",
                f"{rgb(*config.ENCOM_CYAN)}lv:{config.COLOR_LEVELS}/{config.MIX_LEVELS} ",
                f"{rgb(*config.ENCOM_CYAN)}amber:{'on' if config.AMBER_ACCENT else 'off'} ",
                f"{rgb(*config.ENCOM_CYAN)}m:{'on' if config.PHERO_MULTI else 'off'} ",
                f"{rgb(*config.ENCOM_CYAN)}g:{'on' if config.GLITCH_ON else 'off'} ",
                f"{rgb(*config.ENCOM_CYAN)}k:{'on' if config.COOP_SPLIT else 'off'} ",
                f"{rgb(*config.ENCOM_CYAN)}n:{'on' if config.NOVELTY_ON else 'off'} ",
                f"{rgb(*config.ENCOM_CYAN)}R:{'diff' if config.SMART_DIFF else 'full'} ",
                f"{rgb(*config.ENCOM_BLUE)}fps:{fps_val:5.1f}",
                RESET if config.ENABLE_COLOR else ''
            ]
            status = fit_line("".join(status_parts), cols)

            # --- Сборка финального вывода ---
            output_lines = frame_lines
            output_lines.append(status)
            if show_help:
                help_text = "p=Pause r=Reset +/-=Speed ]/[=Add/Remove ISO ... (нажмите ?/h для полного списка)"
                output_lines.append(fit_line(help_text, cols))

            if config.SMART_DIFF:
                smart.render(output_lines)
            else:
                print_frame("\n".join(output_lines))
            
            # --- Подсчет FPS ---
            fps_dt += dt
            fps_frames += 1
            if fps_dt >= 1.0:
                fps_val = fps_frames / fps_dt
                fps_dt = 0.0
                fps_frames = 0

            # --- Обработка ввода ---
            ch = keys.poll()
            if ch:
                if ch in ('q', '\x1b'): break
                if ch == 'p': paused = not paused
                if ch == 'r': sim = ISOLife(w, h, agents=18, logger=logger, use_numpy=config.NP_AVAILABLE)
                if ch == '+': sim_speed = min(8.0, sim_speed * 1.15)
                if ch == '-': sim_speed = max(0.05, sim_speed / 1.15)
                if ch == ']': sim.spawn()
                if ch == '[': sim.remove_one()
                if ch in ('j', 'J', 'y', 'Y'): sim.spawn_pred()
                if ch in ('u', 'U'): sim.remove_pred()
                if ch == 'd': sim.beams_enabled = not sim.beams_enabled
                if ch == 'z': sim.glow_gain = max(0.2, sim.glow_gain - 0.1)
                if ch == 'x': sim.glow_gain = min(3.0, sim.glow_gain + 0.1)
                if ch == 'b': config.FANCY_BORDERS = not config.FANCY_BORDERS
                if ch == 'c': config.ENABLE_COLOR = not config.ENABLE_COLOR
                if ch == 'f': config.FAST_COLOR = not config.FAST_COLOR
                if ch == '2': config.USE_ANSI256 = not config.USE_ANSI256
                if ch == '.': config.COLOR_LEVELS = min(64, config.COLOR_LEVELS + 2); rebuild_luts(config.COLOR_LEVELS, config.MIX_LEVELS)
                if ch == ',': config.COLOR_LEVELS = max(8, config.COLOR_LEVELS - 2); rebuild_luts(config.COLOR_LEVELS, config.MIX_LEVELS)
                if ch == 'a':
                    config.AMBER_ACCENT = not config.AMBER_ACCENT
                    rebuild_luts(config.COLOR_LEVELS, config.MIX_LEVELS)
                    # sim.web.send_flags(config.AMBER_ACCENT) # Эта логика теперь внутри sim
                if ch == 'm': config.PHERO_MULTI = not config.PHERO_MULTI
                if ch == 'g': config.GLITCH_ON = not config.GLITCH_ON
                if ch == 'k': config.COOP_SPLIT = not config.COOP_SPLIT
                if ch == 'n': config.NOVELTY_ON = not config.NOVELTY_ON
                if ch == '3': config.SMART_DIFF = not config.SMART_DIFF
                if ch == 's': sim.purge_wave()
                if ch == 'o': launch_web_viewer()
                if ch in ('h', '?'): show_help = not show_help
            
            time.sleep(1.0 / current_fps_cap())

# ========= Точка входа в приложение =========
def main():
    """Главная функция, разбирает аргументы командной строки."""
    if len(sys.argv) > 1 and sys.argv[1] == "--web-bridge":
        # Запускаем только веб-мост
        run_web_bridge(
            udp_port=config.WEB_FRAME_PORT,
            ws_port=config.WS_PORT,
            http_port=config.HTTP_PORT,
            log_port=config.LOG_PORT,
            stats_port=config.STATS_PORT
        )
        return
        
    # Запускаем основную симуляцию
    try:
        run()
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        # В случае падения, выводим трейсбек для отладки
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()