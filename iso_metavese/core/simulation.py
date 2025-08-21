# core/simulation.py

import math
import random
import time
import socket
import json

try:
    import numpy as np
except ImportError:
    pass

# Импортируем модули проекта
from agents import iso as iso_module
from agents import pred as pred_module
import config
from core.web_streamer import WebStreamer
from core.udp_logger import UDPLogger

# Полезная функция, которую использует симуляция
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

class ISOLife:
    """
    Основной класс симуляции. Управляет состоянием мира (полями),
    всеми агентами (ИСО и Хищниками), их взаимодействием и физикой.
    """
    def __init__(self, w, h, agents=18, seed=None, logger=None, use_numpy=False):
        self.w = w
        self.h = h
        self.use_np = bool(use_numpy and config.NP_AVAILABLE)
        
        if self.use_np:
            self.field = np.zeros((h, w), dtype=np.float32)
            self.pred_field = np.zeros((h, w), dtype=np.float32)
            self.S_food = np.zeros((h, w), dtype=np.float32)
            self.S_help = np.zeros((h, w), dtype=np.float32)
            self.S_warn = np.zeros((h, w), dtype=np.float32)
        else:
            self.field = [[0.0] * w for _ in range(h)]
            self.pred_field = [[0.0] * w for _ in range(h)]
            self.S_food = [[0.0] * w for _ in range(h)]
            self.S_help = [[0.0] * w for _ in range(h)]
            self.S_warn = [[0.0] * w for _ in range(h)]
            
        self.s_blur = 0.18
        self.s_evap = 0.055
        self.agents = []
        self.preds = []
        self.decay = 0.92
        self.blur = 0.15
        self.t = 0.0
        self.glow_gain = 1.0
        self.beams_enabled = True
        
        self._rnd = random.Random(seed)
        self._beam_timer = 0.0
        self._beam_interval = self._rnd.uniform(3.5, 7.5)
        
        self.logger = logger
        self.purge_vis_time = 0.0
        self.purge_vis_total = 0.8
        
        self.web = WebStreamer(
            host=config.LOG_HOST,
            port=config.WEB_FRAME_PORT,
            use_np=self.use_np
        )
        self.web.send_flags(config.AMBER_ACCENT)

        self.agents_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.agents_addr = (config.LOG_HOST, config.AGENTS_DATA_PORT)

        self.metr_timer = 0.0
        self.metr_start = time.perf_counter()
        self.metr_splits = 0
        self.metr_pack = 0

        iso_module.AGENT_ID_SEQ = 1
        pred_module.PRED_ID_SEQ = 1
        
        for _ in range(agents):
            self.spawn()
            
        self._last_social = 0.0
        self._social_dt = 0.25

    # --- NumPy Helpers ---
    @staticmethod
    def _neighbor_avg_np(A):
        left  = np.zeros_like(A); left[:, 1:]  = A[:, :-1]
        right = np.zeros_like(A); right[:, :-1]= A[:, 1:]
        up    = np.zeros_like(A); up[1:, :]    = A[:-1, :]
        down  = np.zeros_like(A); down[:-1, :] = A[1:, :]
        return 0.25 * (left + right + up + down)

    def _diffuse_main_np(self, A, decay, blur):
        if blur > 0.0:
            n = self._neighbor_avg_np(A)
            A *= decay
            A *= (1.0 - blur)
            A += n * blur
        else:
            A *= decay

    def _diffuse_pred_np(self, A):
        n = self._neighbor_avg_np(A)
        A *= 0.72
        A += n * 0.20

    def _diffuse_evap_np(self, A, blur, evap):
        if blur > 0.0:
            n = self._neighbor_avg_np(A)
            A *= (1.0 - blur)
            A += n * blur
        if evap > 0.0:
            A *= (1.0 - evap)

    # --- Spawn/Remove Agents ---
    def spawn(self):
        r = self._rnd
        x = r.uniform(2, self.w - 3); y = r.uniform(2, self.h - 3)
        ang = r.uniform(0, 2 * math.pi); spd = r.uniform(3.0, 7.0)
        rad = r.uniform(0.8, 1.6); hue = r.random(); energy = r.uniform(25, 45)
        cur = r.uniform(0.2, 1.0); soc = r.uniform(0.1, 1.0); bld = r.uniform(0.1, 1.0); pat = r.uniform(0.2, 1.0)
        
        iso = iso_module.ISO(iso_module.AGENT_ID_SEQ, x, y, math.cos(ang) * spd, math.sin(ang) * spd, rad, hue, energy, cur, soc, bld, pat)
        iso_module.AGENT_ID_SEQ += 1
        self.agents.append(iso)
        if self.logger: self.logger.add_iso(iso.id, iso.hue, "spawned")

    def spawn_pred(self):
        r = self._rnd
        x = r.uniform(2, self.w - 3); y = r.uniform(2, self.h - 3)
        ang = r.uniform(0, 2 * math.pi); spd = r.uniform(3.5, 6.5)
        rad = r.uniform(1.0, 1.9); energy = r.uniform(50, 80)
        aggr = r.uniform(0.4, 1.0); stealth = r.uniform(0.2, 1.0); stamina = r.uniform(0.5, 1.0)
        
        pr = pred_module.PRED(pred_module.PRED_ID_SEQ, x, y, math.cos(ang) * spd, math.sin(ang) * spd, rad, energy, aggr, stealth, stamina)
        pred_module.PRED_ID_SEQ += 1
        self.preds.append(pr)
        if self.logger: self.logger.add_pred(pr.id, "predator spawned")

    def remove_one(self):
        if not self.agents: return
        iso = self.agents.pop()
        if self.logger: self.logger.add_iso(iso.id, iso.hue, "removed")

    def remove_pred(self):
        if not self.preds: return
        pr = self.preds.pop()
        if self.logger: self.logger.add_pred(pr.id, "predator removed")

    # --- Grid Operations ---
    def sample(self, grid, x, y):
        if self.use_np:
            return float(grid[int(y) % self.h, int(x) % self.w])
        else:
            return grid[int(y) % self.h][int(x) % self.w]

    def deposit_blob(self, grid, cx, cy, rad, amp):
        ir = max(1, int(rad) + 1); xi = int(cx); yi = int(cy)
        sig2 = (rad * 0.60)**2 + 1e-6
        y0 = max(0, yi - ir); y1 = min(self.h, yi + ir + 1)
        x0 = max(0, xi - ir); x1 = min(self.w, xi + ir + 1)
        if y0 >= y1 or x0 >= x1: return
        
        if self.use_np:
            yy = np.arange(y0, y1, dtype=np.float32)[:, None]
            xx = np.arange(x0, x1, dtype=np.float32)[None, :]
            dy = yy - float(cy); dx = xx - float(cx)
            d2 = dx * dx + dy * dy
            mask = d2 <= float((ir + 1)**2)
            kernel = amp * np.exp(-0.5 * d2 / sig2)
            if mask.any():
                gview = grid[y0:y1, x0:x1]
                gview[mask] += kernel[mask].astype(np.float32, copy=False)
        else:
            for yy in range(y0, y1):
                dy = yy - cy; row = grid[yy]
                for xx in range(x0, x1):
                    dx = xx - cx; d2 = dx * dx + dy * dy
                    if d2 <= (ir + 1)**2:
                        row[xx] += amp * math.exp(-0.5 * d2 / sig2)
                        
    def deposit_phero_point(self, grid, cx, cy, amount):
        xi = int(cx) % self.w; yi = int(cy) % self.h
        if self.use_np:
            grid[yi, xi] += amount
            if xi > 0: grid[yi, xi - 1] += amount * 0.5
            if xi + 1 < self.w: grid[yi, xi + 1] += amount * 0.5
            if yi > 0: grid[yi - 1, xi] += amount * 0.5
            if yi + 1 < self.h: grid[yi + 1, xi] += amount * 0.5
        else:
            grid[yi][xi] += amount
            if xi > 0: grid[yi][xi - 1] += amount * 0.5
            if xi + 1 < self.w: grid[yi][xi + 1] += amount * 0.5
            if yi > 0: grid[yi - 1][xi] += amount * 0.5
            if yi + 1 < self.h: grid[yi + 1][xi] += amount * 0.5

    def _diffuse_evap(self, grid, blur, evap):
        if self.use_np:
            self._diffuse_evap_np(grid, blur, evap)
            return
        w = self.w; h = self.h
        for y in range(h):
            row = grid[y]
            up = grid[y - 1] if y > 0 else [0.0] * w
            dn = grid[y + 1] if y + 1 < h else [0.0] * w
            for x in range(w):
                n = ((row[x - 1] if x > 0 else 0.0) + (row[x + 1] if x + 1 < w else 0.0) + up[x] + dn[x]) * 0.25
                row[x] = (row[x] * (1 - blur) + n * blur) * (1 - evap)

    # --- World Events ---
    def purge_wave(self, factor=0.35):
        if self.use_np:
            self.field *= factor; self.pred_field *= factor
            self.S_food *= factor; self.S_help *= factor; self.S_warn *= factor
        else:
            for y in range(self.h):
                for x in range(self.w):
                    self.field[y][x] *= factor; self.pred_field[y][x] *= factor
                    self.S_food[y][x] *= factor; self.S_help[y][x] *= factor; self.S_warn[y][x] *= factor
        self.purge_vis_time = self.purge_vis_total
        if self.logger: self.logger.add("purge wave", config.ENCOM_BLUE)

    def _beam_pulse(self):
        if not self.beams_enabled: return
        r = self._rnd
        if r.random() < 0.5:
            col = r.randrange(2, self.w - 2)
            if self.use_np: self.field[:, col] += 2.0
            else:
                for y in range(self.h): self.field[y][col] += 2.0
            if self.logger: self.logger.add("databeam: vertical", config.ENCOM_BLUE)
        else:
            rowi = r.randrange(2, self.h - 2)
            if self.use_np: self.field[rowi, :] += 2.0
            else:
                for x in range(self.w): self.field[rowi][x] += 2.0
            if self.logger: self.logger.add("databeam: horizontal", config.ENCOM_BLUE)

    def _do_handshakes(self):
        if (self.t - self._last_social) < self._social_dt: return
        self._last_social = self.t
        if len(self.agents) < 2: return
        
        attempts = min(6, len(self.agents))
        for _ in range(attempts):
            if len(self.agents) < 2: break
            a, b = self._rnd.sample(self.agents, 2)
            if a.link_id >= 0 or b.link_id >= 0: continue
            
            d2 = (a.x - b.x)**2 + (a.y - b.y)**2
            if d2 > 9.0: continue
            
            ang_a = math.atan2(a.vy, a.vx)
            ang_b = math.atan2(b.vy, b.vx)
            dang = abs((ang_b - ang_a + math.pi) % (2 * math.pi) - math.pi)
            if dang > 0.45: continue

            p = 0.20 + 0.60 * (0.5 * (a.sociality + b.sociality))
            if self._rnd.random() < p:
                ttl = 2.5 + 0.8 * 0.5 * (a.sociality + b.sociality)
                a.link_id = b.id; a.link_ttl = ttl
                b.link_id = a.id; b.link_ttl = ttl
                if self.logger and (self.t - a.last_handshake_log) > 0.8:
                    a.last_handshake_log = self.t
                    self.logger.add_iso(a.id, a.hue, f"handshake with ISO-{b.id:02d}")

    # --- Main Simulation Step ---
    def step(self, dt):
        self.t += dt

        if self.use_np:
            self._diffuse_main_np(self.field, self.decay, self.blur)
            self._diffuse_pred_np(self.pred_field)
        else:
            for y in range(self.h):
                row=self.field[y]; up=self.field[y-1] if y>0 else [0.0]*self.w; dn=self.field[y+1] if y+1<self.h else [0.0]*self.w
                for x in range(self.w):
                    n=((row[x-1] if x>0 else 0.0)+(row[x+1] if x+1<self.w else 0.0)+up[x]+dn[x])*0.25
                    row[x]=row[x]*self.decay
                    if self.blur>0: row[x]=row[x]*(1-self.blur)+n*self.blur
            for y in range(self.h):
                row=self.pred_field[y]; up=self.pred_field[y-1] if y>0 else [0.0]*self.w; dn=self.pred_field[y+1] if y+1<self.h else [0.0]*self.w
                for x in range(self.w):
                    n=((row[x-1] if x>0 else 0.0)+(row[x+1] if x+1<self.w else 0.0)+up[x]+dn[x])*0.25
                    row[x]=row[x]*0.90; row[x]=row[x]*(1-0.20)+n*0.20
        if config.PHERO_MULTI:
            self._diffuse_evap(self.S_food, self.s_blur, self.s_evap)
            self._diffuse_evap(self.S_help, self.s_blur, self.s_evap)
            self._diffuse_evap(self.S_warn, self.s_blur, self.s_evap)
        self._beam_timer += dt
        if self._beam_timer >= self._beam_interval:
            self._beam_timer = 0.0; self._beam_interval = self._rnd.uniform(3.5, 7.5)
            self._beam_pulse()
        if self.purge_vis_time > 0: self.purge_vis_time = max(0.0, self.purge_vis_time - dt)
        self._do_handshakes()

        for a in self.agents:
            env_inputs = {'predator': self.sample(self.pred_field, a.x, a.y),
                'social': self.sample(self.S_help, a.x, a.y), 'food': self.sample(self.S_food, a.x, a.y),
                'help': self.sample(self.S_help, a.x, a.y),
                'novelty': self._rnd.random() * 0.3 + 0.7 * (1 - clamp(a.b_pred, 0, 1))}
            social_contacts = sum(1 for b in self.agents if b is not a and math.hypot(a.x - b.x, a.y - b.y) < 5.0)
            a.update_emotions(dt, env_inputs['predator'], social_contacts, env_inputs['novelty'])
            for other in self.agents:
                if other is not a and math.hypot(a.x - other.x, a.y - other.y) < 8.0:
                    a.update_relationships(other, dt)
            a.motivation.update_goals(a, self.t)
            a.make_decision(env_inputs, self.t)
            nearby_agents = [b for b in self.agents if b is not a and math.hypot(a.x - b.x, a.y - b.y) < 6.0]
            if nearby_agents: a.communicate_with_others(nearby_agents, self.t, self.logger)
            if self._rnd.random() < 0.01: a.participate_in_culture(self.agents, self.t, self.logger)

        id_map = {a.id: a for a in self.agents}
        for a in self.agents:
            steering_mods = a.get_steering_modifiers()
            s = math.hypot(a.vx, a.vy) + 1e-9; dx, dy = a.vx / s, a.vy / s
            trend = a.reward_ema - a.last_reward_ema
            if abs(trend) < 0.005: a.stale_t = min(5.0, a.stale_t + dt)
            else: a.stale_t = max(0.0, a.stale_t - 2.0 * dt)
            sdist = clamp(1.8 + 1.8 * a.curiosity + (0.5 if trend < -0.01 else -0.2 if trend > 0.01 else 0.0), 1.0, 4.0)
            lx, ly = a.x - dy * sdist, a.y + dx * sdist; rx, ry = a.x + dy * sdist, a.y - dx * sdist
            chem_gain = 0.6 + 0.9 * a.boldness; chem_gain *= 1.1 if trend > 0.01 else (0.9 if trend < -0.01 else 1.0)
            if config.NOVELTY_ON: chem_gain *= (1.0 + 0.15 * clamp(a.stale_t / 2.0, 0.0, 1.5))
            lG = self.sample(self.field, lx, ly); rG = self.sample(self.field, rx, ry)
            turn = chem_gain * (rG - lG)
            lP = self.sample(self.pred_field, lx, ly); rP = self.sample(self.pred_field, rx, ry)
            near = sum(1 for b in self.agents if b is not a and (b.x - a.x)**2 + (b.y - a.y)**2 < 9.0)
            courage = clamp(near / 6.0, 0.0, 0.7)
            w_fear = (0.6 + 0.9 * (1.0 - a.boldness)) * (1.0 - 0.5 * courage) * steering_mods['warn']
            turn -= w_fear * (rP - lP)
            if near >= 5:
                w_attack = 0.25 + 0.15 * (near - 5)
                turn += w_attack * (lP - rP)
            if config.PHERO_MULTI:
                lF=self.sample(self.S_food,lx,ly); rF=self.sample(self.S_food,rx,ry)
                lH=self.sample(self.S_help,lx,ly); rH=self.sample(self.S_help,rx,ry)
                lW=self.sample(self.S_warn,lx,ly); rW=self.sample(self.S_warn,rx,ry)
                w_food_mod = a.w_food * steering_mods['food']; w_help_mod = a.w_help * steering_mods['help']; w_warn_mod = a.w_warn * steering_mods['warn']
                turn += w_food_mod * (rF - lF) + w_help_mod * (rH - lH) - w_warn_mod * (rW - lW)
            if a.link_id >= 0 and a.link_ttl > 0:
                b = id_map.get(a.link_id)
                if b:
                    ang_cur=math.atan2(a.vy,a.vx); ang_ali=math.atan2(b.vy,b.vx)
                    turn += 0.4*((ang_ali-ang_cur+math.pi)%(2*math.pi)-math.pi)
                a.link_ttl -= dt
                if a.link_ttl <= 0: a.link_id = -1
            noise = a.noise_base * steering_mods['noise']
            if config.NOVELTY_ON: noise *= (1.0 + 0.6*clamp(a.stale_t/3.0,0.0,1.5))
            if trend>0.01: noise *= 0.8
            if trend<-0.01: noise *= 1.25
            turn += self._rnd.uniform(-noise,noise)
            if config.GLITCH_ON and self._rnd.random()<0.001: turn += self._rnd.uniform(-0.8,0.8)
            ct,st=math.cos(turn),math.sin(turn); vx,vy=a.vx,a.vy
            a.vx = vx*ct-vy*st; a.vy = vx*st+vy*ct
            base_speed = 4.5+2.5*clamp(a.energy/60.0,0,1)
            speed = base_speed * steering_mods['speed']
            s2 = math.hypot(a.vx,a.vy)+1e-9
            a.vx=a.vx/s2*speed; a.vy=a.vy/s2*speed

        new_agents=[]
        for a in self.agents:
            prev_energy = a.energy
            a.x+=a.vx*dt; a.y+=a.vy*dt
            margin=2.0+a.rad; bounced=False
            if a.x<margin: a.vx+=(margin-a.x)*8*dt; a.x=margin; bounced=True
            if a.x>self.w-1-margin: a.vx-=(a.x-(self.w-1-margin))*8*dt; a.x=self.w-1-margin; bounced=True
            if a.y<margin: a.vy+=(margin-a.y)*8*dt; a.y=margin; bounced=True
            if a.y>self.h-1-margin: a.vy-=(a.y-(self.h-1-margin))*8*dt; a.y=self.h-1-margin; bounced=True
            if bounced and self.logger and (self.t-a.last_bounce_log)>1.0:
                a.last_bounce_log=self.t; self.logger.add_iso(a.id,a.hue,"deflects at boundary")
            a.phase=(a.phase+dt*(0.6+0.5*clamp(a.energy/60.0,0,1)))%1.0
            beat=0.85+0.35*math.sin(a.phase*2*math.pi)
            local=self.sample(self.field,a.x,a.y); r_inst=0.06*local-0.04
            tau=0.8+2.2*a.patience; alpha=dt/max(1e-6,tau)
            a.last_reward_ema=a.reward_ema; a.reward_ema+=(r_inst-a.reward_ema)*alpha
            a.b_pred+=(local-a.b_pred)*alpha; a.energy+=r_inst*dt
            amp=self.glow_gain*2.0*beat*(0.4+0.6*clamp(a.energy/60.0,0,1))
            self.deposit_blob(self.field,a.x,a.y,a.rad,amp)
            
            energy_delta = a.energy - prev_energy
            a.accumulated_reward += energy_delta * 2.0
            if self.t - a.last_decision_time > 1.0:
                time_elapsed = self.t - a.last_decision_time
                normalized_reward = a.accumulated_reward / time_elapsed if time_elapsed > 0 else 0
                a.brain.learn_from_experience(normalized_reward, a.current_decision)
                a.last_decision_time = self.t; a.accumulated_reward = 0.0

            if config.PHERO_MULTI:
                pred_near=self.sample(self.pred_field,a.x,a.y)
                if r_inst>0.03 and (self.t-a.last_mark_food)>(0.4+0.8*(1.0-a.curiosity)):
                    self.deposit_phero_point(self.S_food,a.x,a.y,1.0+0.6*a.curiosity); a.last_mark_food=self.t
                if (a.energy<20.0 or pred_near>0.6) and (self.t-a.last_mark_help)>(1.0-0.5*a.sociality):
                    self.deposit_phero_point(self.S_help,a.x,a.y,1.2+0.6*a.sociality); a.last_mark_help=self.t
                trend=a.reward_ema-a.last_reward_ema
                if (pred_near>0.45 or trend<-0.02) and (self.t-a.last_mark_warn)>(0.6+0.6*a.patience):
                    warn_amt=1.0+0.8*(1.0-a.boldness)
                    self.deposit_phero_point(self.S_warn,a.x,a.y,warn_amt); a.last_mark_warn=self.t
                f_loc=self.sample(self.S_food,a.x,a.y); h_loc=self.sample(self.S_help,a.x,a.y); w_loc=self.sample(self.S_warn,a.x,a.y)
                delta=r_inst-a.reward_ema; lr=0.15
                a.w_food=clamp(a.w_food+lr*delta*f_loc,0.0,2.5)
                help_need=clamp((20.0-a.energy)/20.0,0.0,1.0)
                a.w_help=clamp(a.w_help+lr*(delta*help_need)*h_loc-0.03*lr*(1.0-help_need),0.0,2.0)
                a.w_warn=clamp(a.w_warn+lr*(-delta)*w_loc,0.0,3.0)

            if a.energy>70.0 and len(self.agents)<160:
                base_p=0.02
                if config.COOP_SPLIT:
                    near=0; soc_sum=0.0
                    for b in self.agents:
                        if b is a: continue
                        if (b.x-a.x)**2+(b.y-a.y)**2<9.0:
                            near+=1; soc_sum+=b.sociality
                    mean_soc=(soc_sum/max(1,near)) if near>0 else 0
                    coop_gain=clamp(near/6.0,0.0,1.0)*(0.5+0.5*mean_soc)
                    base_p*=(1.0+0.8*coop_gain)
                if self._rnd.random()<base_p:
                    child=self._create_child_iso(a)
                    new_agents.append(child)
                    a.energy*=0.55
                    self.metr_splits+=1
                    if self.logger: self.logger.add_iso(a.id,a.hue,f"split -> ISO-{child.id:02d}")
            if a.energy<=0.0:
                self.deposit_blob(self.field,a.x,a.y,1.6,2.5)
                if self.logger: self.logger.add_iso(a.id,a.hue,"expired")
                continue
            new_agents.append(a)
        self.agents = new_agents

        for pr in list(self.preds):
            if pr.cooldown>0: pr.cooldown=max(0.0,pr.cooldown-dt)
            view=7.5+3.5*pr.aggr; target=None; best=1e9
            for a in self.agents:
                d2=(a.x-pr.x)**2+(a.y-pr.y)**2
                if d2<view*view:
                    score=d2*(1.0+1.5*clamp(a.energy/50.0,0,2))
                    if score<best: best=score; target=a
            if target is None:
                pr.ambush_t-=dt*(1.0+pr.stealth*0.4)
                if pr.ambush_t>0: pr.vx*=0.985; pr.vy*=0.985
                else:
                    ang=self._rnd.uniform(0,2*math.pi); spd=1.8+1.8*pr.stealth
                    pr.vx=math.cos(ang)*spd; pr.vy=math.sin(ang)*spd; pr.ambush_t=self._rnd.uniform(1.0,2.4)
            else:
                dx=target.x-pr.x; dy=target.y-pr.y; d=math.hypot(dx,dy)+1e-9
                dirx,diry=dx/d,dy/d; speed=4.8+2.5*pr.aggr
                pr.vx=pr.vx*0.75+dirx*speed*0.25; pr.vy=pr.vy*0.75+diry*speed*0.25
                if pr.cooldown<=0 and d<1.6:
                    bite=5.0+3.0*pr.aggr
                    target.energy-=bite; target.accumulated_reward-=bite*5.0
                    pr.energy=min(pr.e_max,pr.energy+0.4*bite)
                    pr.cooldown=0.9
                    if self.logger and (self.t-pr.last_hit_log)>0.4:
                        pr.last_hit_log=self.t; self.logger.add_pred(pr.id,f"strike ISO-{target.id:02d}")
            pr.x+=pr.vx*dt; pr.y+=pr.vy*dt
            margin=2.0+pr.rad
            if pr.x<margin: pr.vx+=(margin-pr.x)*6*dt; pr.x=margin
            if pr.x>self.w-1-margin: pr.vx-=(pr.x-(self.w-1-margin))*6*dt; pr.x=self.w-1-margin
            if pr.y<margin: pr.vy+=(margin-pr.y)*6*dt; pr.y=margin
            if pr.y>self.h-1-margin: pr.vy-=(pr.y-(self.h-1-margin))*6*dt; pr.y=self.h-1-margin
            self.deposit_blob(self.pred_field, pr.x,pr.y, pr.rad*1.1, 1.8)
            near=sum(1 for a in self.agents if (a.x-pr.x)**2+(a.y-pr.y)**2<9.0)
            if near>=4:
                surplus=near-3; dps=9.0*surplus
                pr.energy-=dps*dt; pr.e_max=max(50.0,pr.e_max-0.5*dt)
                pr.cooldown=max(pr.cooldown,0.8)
                ang=self._rnd.uniform(0,2*math.pi)
                pr.vx=pr.vx*0.4+math.cos(ang)*3.2; pr.vy=pr.vy*0.4+math.sin(ang)*3.2
                self.metr_pack+=1
                if self.logger: self.logger.add_pred(pr.id,"pack-attack")
            pr.energy-=(0.05+0.010*(abs(pr.vx)+abs(pr.vy)))*dt
            if pr.energy<=0:
                self.deposit_blob(self.pred_field,pr.x,pr.y,1.8,3.0)
                if self.logger: self.logger.add_pred(pr.id,"predator down")
                self.preds.remove(pr)
        
        self.metr_timer += dt
        if self.metr_timer >= 0.5:
            self._send_updates()

    def _create_child_iso(self, parent):
        ang = self._rnd.uniform(0, 2 * math.pi); spd = self._rnd.uniform(3.0, 7.0)
        cur = clamp(parent.curiosity * (1.0 + self._rnd.gauss(0, 0.05)), 0.05, 1.2)
        soc = clamp(parent.sociality * (1.0 + self._rnd.gauss(0, 0.05)), 0.05, 1.2)
        bld = clamp(parent.boldness * (1.0 + self._rnd.gauss(0, 0.05)), 0.05, 1.2)
        pat = clamp(parent.patience * (1.0 + self._rnd.gauss(0, 0.05)), 0.05, 1.2)
        child = iso_module.ISO(iso_module.AGENT_ID_SEQ,
            clamp(parent.x+self._rnd.uniform(-1.0,1.0),0.0,self.w-1.0),
            clamp(parent.y+self._rnd.uniform(-1.0,1.0),0.0,self.h-1.0),
            math.cos(ang)*spd,math.sin(ang)*spd,
            clamp(parent.rad*(1.0+self._rnd.uniform(-0.08,0.08)),0.7,1.8),
            (parent.hue+self._rnd.uniform(-0.05,0.05))%1.0,parent.energy*0.45,
            cur,soc,bld,pat)
        iso_module.AGENT_ID_SEQ += 1
        child.w_food = clamp(parent.w_food*(1.0+self._rnd.gauss(0,0.06)),0.0,2.5)
        child.w_help = clamp(parent.w_help*(1.0+self._rnd.gauss(0,0.06)),0.0,2.0)
        child.w_warn = clamp(parent.w_warn*(1.0+self._rnd.gauss(0,0.06)),0.0,3.0)
        child.empathy = clamp(parent.empathy*(1.0+self._rnd.gauss(0,0.08)),0.1,0.9)
        child.creativity = clamp(parent.creativity*(1.0+self._rnd.gauss(0,0.08)),0.1,0.9)
        child.adaptability = clamp(parent.adaptability*(1.0+self._rnd.gauss(0,0.06)),0.3,0.95)
        for trait in child.personality.traits:
            child.personality.traits[trait]=clamp(parent.personality.traits[trait]*(1.0+self._rnd.gauss(0,0.1)),0.1,0.9)
        for key, value in parent.brain.weights.items():
            child.brain.weights[key]=clamp(value*(1.0+self._rnd.gauss(0,0.1)),0.1,2.0)
        return child

    def _send_updates(self):
        self.metr_timer = 0.0
        if self.logger:
            energies=[a.energy for a in self.agents]
            avgE=sum(energies)/len(energies) if energies else 0.0
            pred_avgE=sum(p.energy for p in self.preds)/max(1,len(self.preds)) if self.preds else 0.0
            links=sum(1 for a in self.agents if a.link_id>=0)//2
            emotional_states={}; avg_relationships=0.0; relationship_count=0
            if self.agents:
                for a in self.agents:
                    emotional_states[a.emotional_state]=emotional_states.get(a.emotional_state,0)+1
                    if a.relationships:
                        avg_relationships+=sum(a.relationships.values())
                        relationship_count+=len(a.relationships)
                if relationship_count > 0: avg_relationships/=relationship_count
            elapsed=max(1e-6,time.perf_counter()-self.metr_start)
            splitpm=self.metr_splits/(elapsed/60.0); packpm=self.metr_pack/(elapsed/60.0)
            emotion_str=",".join([f"{k}:{v}" for k,v in emotional_states.items()])
            self.logger.metrics({
                "avg_E": f"{avgE:.1f}","pop":len(self.agents),"pred":len(self.preds),
                "pred_E":f"{pred_avgE:.1f}","links":int(links),"split_pm":f"{splitpm:.2f}",
                "pack_pm":f"{packpm:.2f}","beams":int(self.beams_enabled),
                "fast":int(config.FAST_COLOR),"ansi256":int(config.USE_ANSI256),
                "lv":f"{config.COLOR_LEVELS}/{config.MIX_LEVELS}","amber":int(config.AMBER_ACCENT),
                "phero":int(config.PHERO_MULTI),"glitch":int(config.GLITCH_ON),
                "coop":int(config.COOP_SPLIT),"novel":int(config.NOVELTY_ON),
                "emotions":emotion_str,"avg_rel":f"{avg_relationships:.2f}",
                "mood_avg":f"{sum(a.mood for a in self.agents)/max(1,len(self.agents)):.2f}" if self.agents else "0.0",
                "fear_avg":f"{sum(a.fear for a in self.agents)/max(1,len(self.agents)):.2f}" if self.agents else "0.0"})

        agents_data=[]
        for a in self.agents:
            agents_data.append({
                "id":a.id, "energy":f"{a.energy:.1f}", "pos":f"{int(a.x)},{int(a.y)}",
                "mood":f"{a.mood:.2f}","fear":f"{a.fear:.2f}","social_need":f"{a.social_need:.2f}",
                "goal":a.motivation.current_goal,"decision":a.current_decision,
                "traits":{k:f"{v:.2f}" for k,v in a.personality.traits.items()},
                "brain_w":{k:f"{v:.2f}" for k,v in a.brain.weights.items()}})
        try:
            payload=("@AGNT "+json.dumps(agents_data)).encode('utf-8')
            if len(payload) < 65500: self.agents_sock.sendto(payload,self.agents_addr)
        except Exception: pass

    def render_to_lines(self, render_func):
        lines=[]; w,h=self.w,self.h
        samples=0.0; cnt=0
        for _ in range(256):
            yy=self._rnd.randrange(0,h); xx=self._rnd.randrange(0,w)
            if self.use_np: samples+=float(self.field[yy,xx]+self.pred_field[yy,xx])
            else: samples+=self.field[yy][xx]+self.pred_field[yy][xx]
            cnt+=1
        scale=(samples/max(1,cnt))*2.0+1e-9
        if self.web: self.web.maybe_send(self.field,self.pred_field,scale)
        lines.append(self._border_top(w))
        y_line=-1
        if self.purge_vis_time>0:
            prog=1.0-self.purge_vis_time/self.purge_vis_total
            y_line=int(prog*(h-1))
        for y in range(h):
            if y==y_line:
                from rendering.ansi_helpers import rgb
                content=(rgb(*config.ENCOM_BLUE)+"═"*w) if config.ENABLE_COLOR else "-"*w
            else:
                rowI=self.field[y]; rowP=self.pred_field[y]
                content=render_func(rowI,rowP,scale)
            lines.append(self._wrap_row(content))
        lines.append(self._border_bottom(w))
        return lines

    def _border_top(self,w):
        from rendering.ansi_helpers import rgb,RESET
        if config.FANCY_BORDERS: return (rgb(*config.ENCOM_BLUE)+"█"+"▀"*w+"█"+(RESET if config.ENABLE_COLOR else ""))
        return "+"+"-"*w+"+"

    def _border_bottom(self,w):
        from rendering.ansi_helpers import rgb,RESET
        if config.FANCY_BORDERS: return (rgb(*config.ENCOM_BLUE)+"█"+"▄"*w+"█"+(RESET if config.ENABLE_COLOR else ""))
        return "+"+"-"*w+"+"

    def _wrap_row(self,content):
        from rendering.ansi_helpers import rgb,RESET
        if config.FANCY_BORDERS: return (rgb(*config.ENCOM_BLUE)+"█"+content+"█"+(RESET if config.ENABLE_COLOR else ""))
        return "|"+content+"|"