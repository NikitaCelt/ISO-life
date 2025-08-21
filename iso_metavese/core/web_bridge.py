# core/web_bridge.py

import asyncio
import json
import socket
import threading
import time
import webbrowser
import os
import sys
import shutil
import subprocess
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from http import HTTPStatus

# Используем try-except для опциональной зависимости
try:
    import websockets
except ImportError:
    print("Web bridge requires 'websockets' package. Install: pip install websockets")
    websockets = None

import config

# HTML-код страницы. Обратите внимание на исправленное регулярное выражение.
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ISO-Life Web Viewer</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
  :root { --bg:#0b0f14; --panel:#0f1520; --line:#1a2a3c; --cyan:#7ad2ff; --amber:#ffbe46; --vio:#d66bff; }
  html, body { background:var(--bg); color:#cfe9ff; margin:0; height:100%; font: 14px/1.4 system-ui, Segoe UI, Roboto, sans-serif; }
  header { padding:10px 14px; background:var(--panel); position:sticky; top:0; z-index:10; box-shadow:0 1px 0 rgba(0,0,0,0.35);}
  .row { display:flex; align-items:center; gap:12px; flex-wrap:wrap;}
  .pill { background:#133a66; color:#cfe9ff; padding:4px 8px; border-radius:999px; }
  .ok { background:#135a2e; } .bad { background:#663a13; }
  main { display:grid; grid-template-columns: 1fr 340px; gap: 12px; padding: 12px; }
  #canvasWrap { display:flex; align-items:center; justify-content:center; }
  #canvas { display:block; border:2px solid #1363c6; box-shadow:0 0 16px #0cf6; background:#081018; }
  #panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px; display:flex; flex-direction:column; gap:10px; }
  #stats { display:grid; grid-template-columns: 1fr 1fr; gap:6px 10px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
  #stats div span { opacity:.8 }
  #toggles label { display:inline-flex; align-items:center; gap:6px; margin-right:10px; margin-bottom:6px; }
  #logs { height: 260px; overflow:auto; padding:8px; background:#0b121c; border:1px solid var(--line); border-radius:8px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
  .log-line { white-space: pre-wrap; }
  .log-line .iso { color: var(--cyan); font-weight:600; }
  .log-line .pred { color: var(--vio); font-weight:600; }
  .muted { opacity:.75 }
  footer { text-align:center; opacity:.6; padding:8px 12px 16px; }
  @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <div class="row">
    <strong>ISO-Life Web Viewer</strong>
    <span id="status" class="pill">Connecting...</span>
    <span id="wsurl" class="pill">WS: ws://…</span>
  </div>
</header>
<main>
  <section id="canvasWrap">
    <canvas id="canvas" width="320" height="180"></canvas>
  </section>
  <aside id="panel">
    <div class="row muted">Metrics</div>
    <div id="stats"></div>
    <div class="row muted">Render</div>
    <div id="toggles">
      <label><input type="checkbox" id="hq" checked> HQ upscale</label>
      <label><input type="checkbox" id="bloom" checked> Bloom</label>
      <label><input type="checkbox" id="dither" checked> Dither</label>
      <label><input type="range" id="gamma" min="0.85" max="1.6" step="0.05" value="1.10"> <span>Gamma</span></label>
    </div>
    <div class="row muted">Logs</div>
    <div id="logs"></div>
  </aside>
</main>
<footer>Canvas renders neon mix of ISO + PRED with tone-mapping, palette mix and post-effects.</footer>

<script>
const ENCOM_DEEP=[5,15,25], ENCOM_BLUE=[0,110,220], ENCOM_CYAN=[0,210,255], ENCOM_WHITE=[240,250,255], ENCOM_AMBER=[255,190,40];
const PRED_DEEP=[20,5,30], PRED_VIOLET=[200,0,255], PRED_MAGENTA=[255,60,180];
let AMBER=false, BLOOM=true, HQ=true, DITHER=true, GAMMA=1.10;
const WS_PORT = 51900;
function lerp(a,b,t){return a+(b-a)*t}
function pal_iso(t){
  t=Math.max(0,Math.min(1,t));
  if(t<0.55) return [lerp(ENCOM_DEEP[0],ENCOM_BLUE[0],t/0.55)|0,lerp(ENCOM_DEEP[1],ENCOM_BLUE[1],t/0.55)|0,lerp(ENCOM_DEEP[2],ENCOM_BLUE[2],t/0.55)|0];
  if(t<0.85) return [lerp(ENCOM_BLUE[0],ENCOM_CYAN[0],(t-0.55)/0.30)|0,lerp(ENCOM_BLUE[1],ENCOM_CYAN[1],(t-0.55)/0.30)|0,lerp(ENCOM_BLUE[2],ENCOM_CYAN[2],(t-0.55)/0.30)|0];
  if(AMBER && t>0.98) return [lerp(ENCOM_WHITE[0],ENCOM_AMBER[0],(t-0.98)/0.02)|0,lerp(ENCOM_WHITE[1],ENCOM_AMBER[1],(t-0.98)/0.02)|0,lerp(ENCOM_WHITE[2],ENCOM_AMBER[2],(t-0.98)/0.02)|0];
  return [lerp(ENCOM_CYAN[0],ENCOM_WHITE[0],(t-0.85)/0.15)|0,lerp(ENCOM_CYAN[1],ENCOM_WHITE[1],(t-0.85)/0.15)|0,lerp(ENCOM_CYAN[2],ENCOM_WHITE[2],(t-0.85)/0.15)|0];
}
function pal_pred(t){
  t=Math.max(0,Math.min(1,t));
  if(t<0.6) return [lerp(PRED_DEEP[0],PRED_VIOLET[0],t/0.6)|0,lerp(PRED_DEEP[1],PRED_VIOLET[1],t/0.6)|0,lerp(PRED_DEEP[2],PRED_VIOLET[2],t/0.6)|0];
  if(t<0.9) return [lerp(PRED_VIOLET[0],PRED_MAGENTA[0],(t-0.6)/0.3)|0,lerp(PRED_VIOLET[1],PRED_MAGENTA[1],(t-0.6)/0.3)|0,lerp(PRED_VIOLET[2],PRED_MAGENTA[2],(t-0.6)/0.3)|0];
  return [lerp(PRED_MAGENTA[0],ENCOM_WHITE[0],(t-0.9)/0.1)|0,lerp(PRED_MAGENTA[1],ENCOM_WHITE[1],(t-0.9)/0.1)|0,lerp(PRED_MAGENTA[2],ENCOM_WHITE[2],(t-0.9)/0.1)|0];
}
function toneMap(v,k=1.3){ return 1.0 - Math.exp(-k*Math.max(0,v)); }
function gammaApply(u){ return Math.pow(Math.max(0,Math.min(1,u)), 1.0/GAMMA); }
const B4 = new Uint8Array([0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5]);
function bayer4(x,y){ return (B4[((y&3)<<2)|(x&3)]/15)-0.5; }
const statusEl = document.getElementById('status'), wsurlEl = document.getElementById('wsurl'), canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d'), statsEl = document.getElementById('stats'), logsEl = document.getElementById('logs');
const hqEl = document.getElementById('hq'), bloomEl = document.getElementById('bloom'), ditherEl = document.getElementById('dither'), gammaEl = document.getElementById('gamma');
hqEl.onchange = ()=>{ HQ = hqEl.checked; };
bloomEl.onchange = ()=>{ BLOOM = bloomEl.checked; };
ditherEl.onchange = ()=>{ DITHER = ditherEl.checked; };
gammaEl.oninput = ()=>{ GAMMA = parseFloat(gammaEl.value || "1.10"); };
const srcCanvas = document.createElement('canvas'), srcCtx = srcCanvas.getContext('2d', { willReadFrequently: true });
let latest = null, W=0,H=0;
function setCanvasSize(W,H){
  srcCanvas.width = W; srcCanvas.height = H;
  const maxW = Math.min(1100, window.innerWidth - (window.innerWidth>980 ? 360 : 24));
  const scale = Math.max(1, Math.floor(maxW / W));
  canvas.width = Math.floor(W*scale); canvas.height = Math.floor(H*scale);
}
function stripAnsi(s){ return s.replace(/\x1b```math[0-9;]*[ -\\/]*[@-~]/g,""); }
function classifyLog(s){
  if(s.includes("PRED-")) return "pred"; if(s.includes("ISO-")) return "iso"; return "";
}
function pushLog(msg){
  const line = document.createElement('div'); line.className = 'log-line';
  const cls = classifyLog(msg); const clean = stripAnsi(msg);
  if(cls==="iso"){ line.innerHTML = clean.replace(/(ISO-\d{2})/g,'<span class="iso">$1</span>');
  }else if(cls==="pred"){ line.innerHTML = clean.replace(/(PRED-\d{2})/g,'<span class="pred">$1</span>');
  }else{ line.textContent = clean; }
  logsEl.appendChild(line);
  while(logsEl.childElementCount > 300) logsEl.firstChild.remove();
  logsEl.scrollTop = logsEl.scrollHeight;
}
function renderMetrics(d){
  const kv = (k,v)=>`<div><span>${k}</span></div><div><strong>${v}</strong></div>`;
  statsEl.innerHTML =
    kv("pop", d.pop ?? "-") + kv("pred", d.pred ?? "-") + kv("avg", d.avg ?? "-") + kv("med", d.med ?? "-") + kv("min", d.min ?? "-") + kv("max", d.max ?? "-") + kv("predAvg", d.predavg ?? "-") + kv("links", d.links ?? "-") + kv("split/min", d.splitpm ?? "-") + kv("pack/min", d.packpm ?? "-") + kv("tension", d.tension ?? "-") + kv("beams", d.beams ?? "-") + kv("fast", d.fast ?? "-") + kv("ansi256", d.ansi256 ?? "-") + kv("levels", (d.lv??"-")+"/"+(d.mix??"-")) + kv("amber", d.amber ?? "-") + kv("phero", d.phero ?? "-") + kv("glitch", d.glitch ?? "-") + kv("coop", d.coop ?? "-") + kv("novel", d.novel ?? "-") + kv("mood", d.mood_avg ?? "-") + kv("fear", d.fear_avg ?? "-") + kv("relations", d.avg_relations ?? "-");
}
function drawFrame(){
  if(!latest) return; const bytes = latest;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength); let off=0;
  if(String.fromCharCode(bytes[0],bytes[1],bytes[2],bytes[3])!=="FRME") return; off+=4;
  const w = dv.getUint16(off,true); off+=2; const h = dv.getUint16(off,true); off+=2; const flags = bytes[off]; off+=1;
  if(w!==W || h!==H){ W=w; H=h; setCanvasSize(W,H); }
  const img = srcCtx.createImageData(W,H); const data = img.data; let di=0;
  for(let y=0;y<H;y++){ for(let x=0;x<W;x++){
    const t8 = bytes[off++]; const r8 = bytes[off++]; let tot = t8/255.0; let mix = r8/255.0;
    if(DITHER){ const d = bayer4(x,y) * (1.0/24); tot = Math.min(1, Math.max(0, tot + d)); }
    const t = toneMap(tot, 1.3); const c1 = pal_iso(t); const c2 = pal_pred(t);
    let r = (c1[0]*(1-mix)+c2[0]*mix)/255; let g = (c1[1]*(1-mix)+c2[1]*mix)/255; let b = (c1[2]*(1-mix)+c2[2]*mix)/255;
    r = gammaApply(r); g = gammaApply(g); b = gammaApply(b);
    data[di++] = (r*255)|0; data[di++] = (g*255)|0; data[di++] = (b*255)|0; data[di++] = 255;
  }}
  srcCtx.putImageData(img,0,0);
  ctx.save(); ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.imageSmoothingEnabled = !!HQ; ctx.drawImage(srcCanvas, 0,0, canvas.width, canvas.height);
  if(BLOOM){
    ctx.globalCompositeOperation = 'lighter'; ctx.globalAlpha = 0.5; ctx.filter = 'blur(1.2px)';
    ctx.drawImage(srcCanvas, 0,0, canvas.width, canvas.height);
    ctx.filter = 'none'; ctx.globalAlpha = 1.0; ctx.globalCompositeOperation = 'source-over';
  }
  ctx.restore();
}
function frameLoop(){ drawFrame(); requestAnimationFrame(frameLoop); }
function makeHost(){
  let h = location.hostname || "127.0.0.1";
  if(h === "" || h === "0.0.0.0" || h === "localhost" || h === "[::1]" || h === "::1") h = "127.0.0.1";
  return h;
}
function updateWsUrlLabel(host, path){ wsurlEl.textContent = "WS: ws://" + host + ":" + WS_PORT + path; }
function connect(){
  const host = makeHost(); const tryPaths = ["/ws", "/"]; let idx = 0;
  function tryConnect(){
    const path = tryPaths[idx]; const url = "ws://" + host + ":" + WS_PORT + path;
    updateWsUrlLabel(host, path); console.log("[WS] connecting", url);
    const ws = new WebSocket(url); ws.binaryType = "arraybuffer";
    ws.onopen = ()=>{ console.log("[WS] open", url); statusEl.textContent="Connected"; statusEl.classList.remove('bad'); statusEl.classList.add('ok'); };
    ws.onclose= ()=>{
      console.log("[WS] close", url);
      if(idx+1 < tryPaths.length){ idx += 1; setTimeout(tryConnect, 200);
      }else{ statusEl.textContent="Disconnected (retrying...)"; statusEl.classList.remove('ok'); statusEl.classList.add('bad'); setTimeout(()=>{ idx=0; tryConnect(); }, 1000); }
    };
    ws.onerror= (e)=>{ console.warn("[WS] error", e); ws.close(); };
    ws.onmessage=(ev)=>{
      if(ev.data instanceof ArrayBuffer){ latest = new Uint8Array(ev.data);
      }else{
        try{
          const m = JSON.parse(ev.data);
          if(m.type === "flags"){ AMBER = !!m.amber;
          }else if(m.type === "log"){ pushLog(m.msg || "");
          }else if(m.type === "metrics"){ renderMetrics(m.data || {}); }
        }catch(e){}
      }
    };
  }
  tryConnect();
}
connect();
requestAnimationFrame(frameLoop);
window.addEventListener('resize', ()=>{ if(W && H) setCanvasSize(W,H); });
</script>
</body>
</html>
"""

def run_web_bridge(udp_port, ws_port, http_port, log_port, stats_port):
    if websockets is None:
        print("Cannot run web bridge because 'websockets' is not installed.")
        return

    class HttpHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path in ('/', '/index.html', '/ws'):
                    data = INDEX_HTML.encode('utf-8')
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == '/favicon.ico':
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                else:
                    self.send_response(HTTPStatus.NOT_FOUND)
                    self.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                pass # Client closed connection, ignore
            except Exception as e:
                print(f"[HTTP ERR] {e}")

        def log_message(self, *args):
            pass

    httpd = HTTPServer((config.LOG_HOST, http_port), HttpHandler)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()
    print(f"[WEB] HTTP on http://{config.LOG_HOST}:{http_port}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    clients = set()
    last_frame = {"data": None}
    last_metrics = {"data": None}
    flags_state = {"amber": config.AMBER_ACCENT}
    pending_text = deque()
    ev = asyncio.Event()
    counters = {"fr": 0, "lg": 0, "st": 0}
    last_tick = time.time()

    async def send_to_all(data):
        if not clients: return
        tasks = [ws.send(data) for ws in clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for ws, result in zip(list(clients), results):
            if isinstance(result, Exception):
                clients.discard(ws)

    async def ws_handler(ws, path=None):
        clients.add(ws)
        peer = getattr(ws, "remote_address", "unknown")
        print(f"[WEB] WS client connected: {peer} path={path}")
        try:
            if flags_state["amber"] is not None:
                await ws.send(json.dumps({"type": "flags", "amber": flags_state["amber"]}))
            if last_metrics["data"] is not None:
                await ws.send(json.dumps({"type": "metrics", "data": last_metrics["data"]}))
            if last_frame["data"] is not None:
                await ws.send(last_frame["data"])
            await ws.wait_closed()
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            clients.discard(ws)
            print(f"[WEB] WS client disconnected: {peer}")

    async def broadcaster():
        while True:
            await ev.wait()
            ev.clear()
            if last_frame["data"] is not None:
                await send_to_all(last_frame["data"])
            while pending_text:
                await send_to_all(json.dumps(pending_text.popleft()))

    def udp_thread_runner(sock, callback):
        nonlocal last_tick
        while True:
            try:
                data, _ = sock.recvfrom(2**20)
                loop.call_soon_threadsafe(callback, data)
                if time.time() - last_tick > 2.0:
                    print(f"[WEB] UDP in (2s): frames={counters['fr']} logs={counters['lg']} stats={counters['st']}")
                    counters["fr"] = counters["lg"] = counters["st"] = 0
                    last_tick = time.time()
            except socket.timeout:
                continue
            except (OSError, socket.error):
                break

    def on_frame_data(data):
        if data.startswith(b"FRME"):
            last_frame["data"] = data
            counters["fr"] += 1
            ev.set()
        elif data.startswith(b"FLAG"):
            amber = bool(data[4]) if len(data) > 4 else False
            if flags_state["amber"] != amber:
                flags_state["amber"] = amber
                pending_text.append({"type": "flags", "amber": amber})
                ev.set()

    def on_log_data(data):
        msg = data.decode('utf-8', 'ignore')
        counters["lg"] += 1
        pending_text.append({"type": "log", "ts": time.time(), "msg": msg})
        ev.set()

    def on_stats_data(data):
        msg = data.decode('utf-8', 'ignore')
        if msg.startswith("@METR "):
            body = msg[6:].strip()
            d = dict(kv.split("=", 1) for kv in body.split(";") if "=" in kv)
            last_metrics["data"] = d
            counters["st"] += 1
            pending_text.append({"type": "metrics", "data": d})
            ev.set()

    socks = []
    for port, callback in [(udp_port, on_frame_data), (log_port, on_log_data), (stats_port, on_stats_data)]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((config.LOG_HOST, port))
            sock.settimeout(2.5)
            socks.append(sock)
            threading.Thread(target=udp_thread_runner, args=(sock, callback), daemon=True).start()
        except OSError as e:
            print(f"[WEB] Cannot bind UDP {port}: {e}. Web bridge may not function correctly.")

    async def main_async():
        server_args = {"ping_interval": 20, "ping_timeout": 20}
        async with websockets.serve(ws_handler, config.LOG_HOST, ws_port, **server_args):
            print(f"[WEB] WS on ws://{config.LOG_HOST}:{ws_port} (paths: /ws or /)")
            await broadcaster()

    try:
        loop.run_until_complete(main_async())
    except KeyboardInterrupt:
        pass
    finally:
        for sock in socks: sock.close()
        httpd.shutdown()
        try: loop.stop()
        except Exception: pass

def launch_web_viewer():
    script = os.path.abspath(sys.argv[0])
    py = sys.executable
    args = [py, script, "--web-bridge"]
    
    ok = False
    try:
        if os.name == 'nt':
            flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
            subprocess.Popen(args, creationflags=flags)
            ok = True
        else:
            for exe, prefix in [("x-terminal-emulator", ["-e"]), ("xterm", ["-e"]), ("gnome-terminal", ["--"]),
                               ("konsole", ["-e"]), ("alacritty", ["-e"]), ("kitty", ["-e"]),
                               ("lxterminal", ["-e"]), ("mate-terminal", ["-e"])]:
                if shutil.which(exe):
                    subprocess.Popen([exe] + prefix + args)
                    ok = True
                    break
    except Exception:
        ok = False

    url = f"http://{config.LOG_HOST}:{config.HTTP_PORT}/"
    print(f"[HINT] Opening web viewer at {url}")
    time.sleep(1) # Даем серверу время на запуск
    try:
        webbrowser.open(url, new=2)
    except Exception as e:
        print(f"[WARN] Could not open web browser: {e}")

    if not ok:
        print(f"\n[HINT] Could not auto-start web bridge.")
        print(f"[HINT] Please start it in another shell: {py} {script} --web-bridge")