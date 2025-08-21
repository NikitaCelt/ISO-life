# core/web_streamer.py
import socket
import time
import numpy as np
import config

class WebStreamer:
    def __init__(self, host=config.LOG_HOST, port=config.WEB_FRAME_PORT, fps=15, use_np=False):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.fps = fps
        self.last_ts = 0.0
        self.use_np = use_np and config.NP_AVAILABLE

    def _pack_frame(self, field, pred_field, scale):
        if self.use_np:
            h, w = field.shape
        else:
            h = len(field)
            w = len(field[0]) if h > 0 else 0
        
        flags = 0
        if config.USE_ANSI256: flags |= 1
        
        buf = bytearray()
        buf += b"FRME"
        buf += (w & 0xFFFF).to_bytes(2, 'little')
        buf += (h & 0xFFFF).to_bytes(2, 'little')
        buf += bytes([flags])
        inv_scale = 1.0 / scale if scale > 1e-9 else 1.0
        
        if self.use_np:
            iso = field.astype(np.float32, copy=False)
            prd = pred_field.astype(np.float32, copy=False)
            tot = iso + prd
            tot_norm = np.clip(tot * inv_scale, 0.0, 1.0)
            ratio = np.divide(prd, tot + 1e-6, out=np.zeros_like(prd), where=(tot > 1e-6))
            ratio = np.clip(ratio * 1.3, 0.0, 1.0)
            t8 = (tot_norm * 255.0 + 0.5).astype(np.uint8, copy=False)
            r8 = (ratio * 255.0 + 0.5).astype(np.uint8, copy=False)
            interleaved = np.empty((h, w * 2), dtype=np.uint8)
            interleaved[:, 0::2] = t8
            interleaved[:, 1::2] = r8
            buf += interleaved.tobytes()
        else:
            for y in range(h):
                rowI, rowP = field[y], pred_field[y]
                for x in range(w):
                    ib, ip = rowI[x], rowP[x]
                    tot = ib + ip
                    tot_norm = min(1.0, max(0.0, tot * inv_scale))
                    ratio = 0.0 if tot <= 1e-6 else min(1.0, max(0.0, (ip / tot) * 1.3))
                    buf.append(int(tot_norm * 255 + 0.5))
                    buf.append(int(ratio * 255 + 0.5))
        return bytes(buf)

    def maybe_send(self, field, pred_field, scale):
        now = time.perf_counter()
        if now - self.last_ts < (1.0 / self.fps):
            return
        self.last_ts = now
        try:
            data = self._pack_frame(field, pred_field, scale)
            self.sock.sendto(data, self.addr)
        except Exception:
            pass

    def send_flags(self, amber: bool):
        try:
            payload = b"FLAG" + (b"\x01" if amber else b"\x00")
            self.sock.sendto(payload, self.addr)
        except Exception:
            pass