# core/udp_logger.py

import socket
import config
from rendering.ansi_helpers import RESET, rgb_code

class UDPLogger:
    """Отправляет логи и метрики по UDP в веб-мост."""
    def __init__(self, host=config.LOG_HOST, log_port=config.LOG_PORT, stats_port=config.STATS_PORT):
        self.log_addr = (host, log_port)
        self.stats_addr = (host, stats_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Установка неблокирующего режима важна, чтобы отправка не "замораживала" симуляцию
        self.sock.setblocking(False)

    def _send(self, addr, msg: str):
        """Обертка для безопасной отправки UDP-пакета."""
        try:
            self.sock.sendto(msg.encode('utf-8', 'ignore'), addr)
        except Exception:
            # Игнорируем ошибки, если сокет занят или мост не запущен
            pass

    def add(self, text, color=None):
        """Отправляет общее текстовое сообщение в лог."""
        if color:
            self._send(self.log_addr, f"{rgb_code(*color)}{text}{RESET}")
        else:
            self._send(self.log_addr, text)

    def add_iso(self, iso_id, hue, text):
        """Отправляет лог, связанный с конкретным ИСО."""
        self._send(
            self.log_addr,
            f"{rgb_code(*config.ENCOM_CYAN)}ISO-{iso_id:02d}{RESET} "
            f"{rgb_code(*config.ENCOM_AMBER)}{text}{RESET}"
        )

    def add_pred(self, pred_id, text):
        """Отправляет лог, связанный с хищником."""
        self._send(
            self.log_addr,
            f"{rgb_code(*config.PRED_VIOLET)}PRED-{pred_id:02d}{RESET} "
            f"{rgb_code(*config.ENCOM_AMBER)}{text}{RESET}"
        )

    def metrics(self, d: dict):
        """Отправляет словарь с метриками."""
        parts = [f"{k}={v}" for k, v in d.items()]
        self._send(self.stats_addr, "@METR " + ";".join(parts))