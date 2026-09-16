# ===== AMO on Isaac Sim (G1) — v3 robust (update push + diag logs) =====
# Pega en Script Editor (Isaac Sim 4.5), pulsa Run, luego PLAY.

import math, os, atexit, traceback, threading, time
import carb
from pxr import Usd
import numpy as np
import torch

import omni.usd, omni.timeline
from omni.kit.app import get_app
from omni.isaac.dynamic_control import _dynamic_control as dynamic_control

# ---------- CONFIG ----------
POLICIES_DIR   = "/home/alan/Desktop/policies"
POLICY_JIT     = "amo_jit.pt"
ADAPTER_JIT    = "adapter_jit.pt"
ADAPTER_STATS  = "adapter_norm_stats.pt"

ART_PATH         = "/World/g1/pelvis"
IMU_PATH         = "/World/g1/pelvis/imu_in_torso"

# Comandos base (si no hay gamepad)
CMD_VX   = 0.7
CMD_VY   = 0.0
CMD_YAW  = 0.0
CMD_DZ   = 0.0
TORSO_YAW = 0.0
TORSO_PIT = 0.0
TORSO_ROL = 0.0
CMD_VY_TRIM = 0.0   # corrige deriva lateral si la hay

USE_GAMEPAD = True           # requiere pygame instalado en el entorno de Isaac
GAMEPAD_DEADZONE = 0.15
GP_SCALE_FWD   = 1.2
GP_SCALE_YAW   = 1.2
GP_SCALE_STRAF = 0.5
GP_SCALE_DZ    = 0.5
GP_STEP_TORSO  = 0.05

CONTROL_HZ        = 50.0
_DT_CTRL          = 1.0 / CONTROL_HZ
TARGET_FILTER_TAU = 0.08

WARMUP_SECONDS    = 0.8
ACTION_RAMP_SEC   = 1.0

# postura inicial un poco más estable
START_POSE_OFFSETS = {
    "left_hip_roll":  +0.12,
    "right_hip_roll": -0.12,
    "left_knee":      +0.10,
    "right_knee":     +0.10,
    "waist_pitch":    -0.10,
}

# PD / Limits / Default pose (AMO MuJoCo)
G1_STIFFNESS = np.array([
    150, 150, 150, 300,  80,  20,
    150, 150, 150, 300,  80,  20,
    400, 400, 400,
     80,  80,  40,  60,
     80,  80,  40,  60,
], dtype=np.float32)
G1_DAMPING = np.array([
    2, 2, 2, 4, 2, 1,
    2, 2, 2, 4, 2, 1,
    15, 15, 15,
    2, 2, 1, 1,
    2, 2, 1, 1,
], dtype=np.float32)
G1_TORQUE_LIMS = np.array([
    88, 139, 88, 139, 50, 50,
    88, 139, 88, 139, 50, 50,
    88,  50,  50,
    25,  25,  25,  25,
    25,  25,  25,  25,
], dtype=np.float32)
G1_DEFAULT_DOF_POS = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     0.0, 0.0, 0.0,
     0.5, 0.0, 0.2, 0.3,
     0.5, 0.0,-0.2, 0.3,
], dtype=np.float32)

ACTION_SCALE   = 0.25
SCALE_ANG_VEL  = 0.25
SCALE_DOF_VEL  = 0.05
HIST_LEN       = 10
EXTRA_HIST_LEN = 25

AMO_DOF_ORDER = [
    "left_hip_pitch","left_hip_roll","left_hip_yaw","left_knee","left_ankle_pitch","left_ankle_roll",
    "right_hip_pitch","right_hip_roll","right_hip_yaw","right_knee","right_ankle_pitch","right_ankle_roll",
    "waist_yaw","waist_roll","waist_pitch",
    "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow",
    "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow",
]

# ---------- HELPERS ----------
def _log(m): carb.log_info(f"[amo-isaac] {m}")
def _warn(m): carb.log_warn(f"[amo-isaac] {m}")
def _err(m): carb.log_error(f"[amo-isaac] {m}")
def _norm(nm: str) -> str:
    s = (nm or "").lower()
    for suf in ("_joint","joint","_dof","_dof0","_dof1","_dof2"):
        if s.endswith(suf): s = s[:-len(suf)]
    return s

# ---------- CONTROLLER OBJECT (evita GC) ----------
class AmoController:
    def __init__(self):
        # Kit / Isaac
        self.app = get_app()
        self.timeline = omni.timeline.get_timeline_interface()
        self.usd_ctx = omni.usd.get_context()
        self.dc = dynamic_control.acquire_dynamic_control_interface()

        # estado de runtime
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._policy = None
        self._adapter = None
        self._in_mean=self._in_std=self._out_mean=self._out_std = None

        self._ART = 0
        self._DOF_HANDLES = []
        self._DOF_NAMES = []
        self._IDX_ISAAC_FROM_AMO = []

        self._nj = 23
        self._last_action = np.zeros(self._nj, dtype=np.float32)
        self._gait_cycle = np.array([0.25, 0.75], dtype=np.float32)
        self._gait_freq  = 1.3

        self._n_priv = 3
        self._n_proprio = 3 + 2 + 2 + 23 + 23 + 23 + 2 + 15
        self._prop_hist = [np.zeros(self._n_proprio, dtype=np.float32) for _ in range(HIST_LEN)]
        self._extra_hist= [np.zeros(self._n_proprio, dtype=np.float32) for _ in range(EXTRA_HIST_LEN)]
        self._q_cmd_prev = G1_DEFAULT_DOF_POS.copy()

        self._cmd_vec = np.array(
            [CMD_VX, CMD_YAW, CMD_VY, CMD_DZ, TORSO_YAW, TORSO_PIT, TORSO_ROL, 0.0],
            dtype=np.float32
        )
        self._cmd_lock = threading.Lock()

        self._built = False
        self._start_t = 0.0
        self._last_t = 0.0
        self._accum  = 0.0

        # Diag
        self._tick_diag_t = 0.0

        # Gamepad
        self._gp_thread = None
        self._gp_stop   = False

        # Subs (usar PUSH para mayor compat FP 4.5)
        self.upd_sub = self.app.get_update_event_stream().create_subscription_to_push(
            self._on_update, name="amo_update_push"
        )
        self.tl_sub  = self.timeline.get_timeline_event_stream().create_subscription_to_push(
            self._on_timeline, name="amo_timeline_push"
        )

        _log("Inicializado (espera PLAY).")
        # Si ya estaba en PLAY antes de ejecutar, construye inmediatamente:
        try:
            if self.timeline.is_playing():
                _log("PLAY ya activo → build inmediato")
                self._build()
        except Exception:
            pass

    # ---------- low-level ----------
    def _check_paths(self):
        ok = True
        for f in (POLICY_JIT, ADAPTER_JIT, ADAPTER_STATS):
            p = os.path.join(POLICIES_DIR, f)
            if not os.path.isfile(p):
                _err(f"Falta archivo de política/adapter: {p}")
                ok = False
        if not ok:
            _err("Corrige rutas/archivos en POLICIES_DIR.")
        return ok

    def _load_nets(self):
        ppth = os.path.join(POLICIES_DIR, POLICY_JIT)
        apth = os.path.join(POLICIES_DIR, ADAPTER_JIT)
        spth = os.path.join(POLICIES_DIR, ADAPTER_STATS)
        self._policy  = torch.jit.load(ppth, map_location=self._device)
        self._adapter = torch.jit.load(apth, map_location=self._device)
        self._policy.eval(); self._adapter.eval()
        for m in (self._policy, self._adapter):
            for p in m.parameters(): p.requires_grad=False
        stats = torch.load(spth, weights_only=False, map_location="cpu")
        self._in_mean  = torch.tensor(stats["input_mean"],  dtype=torch.float32, device=self._device)
        self._in_std   = torch.tensor(stats["input_std"],   dtype=torch.float32, device=self._device)
        self._out_mean = torch.tensor(stats["output_mean"], dtype=torch.float32, device=self._device)
        self._out_std  = torch.tensor(stats["output_std"],  dtype=torch.float32, device=self._device)
        _log(f"Políticas cargadas en {self._device}.")

    def _get_mapping_and_handles(self):
        self._ART = self.dc.get_articulation(ART_PATH)
        if not self._ART:
            _err(f"No hay articulation en {ART_PATH}")
            return False
        ndof = self.dc.get_articulation_dof_count(self._ART)
        self._DOF_HANDLES = [self.dc.get_articulation_dof(self._ART,i) for i in range(ndof)]
        self._DOF_NAMES=[]
        for h in self._DOF_HANDLES:
            try: self._DOF_NAMES.append((self.dc.get_dof_name(h) or "dof"))
            except: self._DOF_NAMES.append("dof")

        name_to_idx = {_norm(n): i for i,n in enumerate(self._DOF_NAMES)}
        # aliases por si tu USD usa pelvis_* o torso_* en vez de waist_*
        name_to_idx.setdefault("waist_pitch", name_to_idx.get("pelvis_pitch", name_to_idx.get("torso_pitch")))
        name_to_idx.setdefault("waist_roll",  name_to_idx.get("pelvis_roll",  name_to_idx.get("torso_roll")))
        name_to_idx.setdefault("waist_yaw",   name_to_idx.get("pelvis_yaw",   name_to_idx.get("torso_yaw")))

        self._IDX_ISAAC_FROM_AMO = []
        miss=[]
        for base in AMO_DOF_ORDER:
            i = name_to_idx.get(base)
            if i is None: miss.append(base)
            self._IDX_ISAAC_FROM_AMO.append(i if i is not None else -1)

        mapped = sum(1 for k in self._IDX_ISAAC_FROM_AMO if k is not None and k>=0)
        _log(f"DOFs Isaac({ndof}): {self._DOF_NAMES}")
        if miss:
            _warn(f"No mapeé {len(miss)} DOFs: {miss}")
        _log(f"Mapeados {mapped}/{len(AMO_DOF_ORDER)} DOFs.")
        return mapped >= 15

    def _quat_to_euler_wxyz(self, q):
        qw,qx,qy,qz = q
        sinr = 2*(qw*qx + qy*qz); cosr = 1 - 2*(qx*qx + qy*qy)
        roll = math.atan2(sinr, cosr)
        sinp = 2*(qw*qy - qz*qx)
        pitch = math.copysign(math.pi/2, sinp) if abs(sinp)>=1 else math.asin(sinp)
        siny = 2*(qw*qz + qx*qy); cosy = 1 - 2*(qy*qy + qz*qz)
        yaw = math.atan2(siny, cosy)
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _read_imu(self):
        try:
            rb = self.dc.get_rigid_body(IMU_PATH)
            if rb:
                pose = self.dc.get_rigid_body_pose(rb)
                if hasattr(pose, "r") and hasattr(pose.r, "w"):
                    q = [pose.r.w, pose.r.x, pose.r.y, pose.r.z]
                elif hasattr(pose, "r") and len(pose.r)==4:
                    q = [pose.r[3], pose.r[0], pose.r[1], pose.r[2]]
                else:
                    q = [1.0,0.0,0.0,0.0]
                av = self.dc.get_rigid_body_angular_velocity(rb)
                avel = np.array([av.x, av.y, av.z], dtype=np.float32) if hasattr(av,"x") else np.array(av, dtype=np.float32)
                return np.array(q, dtype=np.float32), avel
        except Exception:
            pass
        return np.array([1.0,0.0,0.0,0.0], dtype=np.float32), np.zeros(3, dtype=np.float32)

    def _get_qdq_in_amo_order(self):
        q_raw = []; dq_raw=[]
        for h in self._DOF_HANDLES:
            try:
                q_raw.append(float(self.dc.get_dof_position(h)))
                dq_raw.append(float(self.dc.get_dof_velocity(h)))
            except:
                q_raw.append(0.0); dq_raw.append(0.0)
        q_raw = np.array(q_raw, dtype=np.float32)
        dq_raw= np.array(dq_raw, dtype=np.float32)
        q = np.zeros(self._nj, dtype=np.float32)
        dq= np.zeros(self._nj, dtype=np.float32)
        for amo_i, ix in enumerate(self._IDX_ISAAC_FROM_AMO):
            if ix is None or ix<0 or ix>=len(q_raw): continue
            q[amo_i]  = q_raw[ix]
            dq[amo_i] = dq_raw[ix]
        return q, dq

    def _filter_targets(self, q_des, q_prev, dt):
        if dt <= 0.0: return q_des
        alpha = 1.0 - math.exp(-dt / max(1e-4, TARGET_FILTER_TAU))
        return (1.0 - alpha) * q_prev + alpha * q_des

    # ---------- policy ----------
    def _compute_observation(self):
        quat_wxyz, ang_vel = self._read_imu()
        rpy = self._quat_to_euler_wxyz(quat_wxyz)
        q, dq = self._get_qdq_in_amo_order()

        with self._cmd_lock:
            commands = self._cmd_vec.copy()
        commands[2] += CMD_VY_TRIM

        target_yaw = commands[1]
        dyaw = rpy[2] - target_yaw
        dyaw = (dyaw + math.pi) % (2*math.pi) - math.pi
        in_place = (abs(commands[0]) < 0.1)
        if in_place: dyaw = 0.0

        gait_obs = np.sin(self._gait_cycle * 2 * math.pi).astype(np.float32)

        adapter_in = np.zeros(12, dtype=np.float32)
        adapter_in[0] = 0.75 + commands[3]
        adapter_in[1] = commands[4]; adapter_in[2] = commands[5]; adapter_in[3] = commands[6]
        adapter_in[4:] = q[15:]

        try:
            ain = torch.tensor(adapter_in, device=self._device, dtype=torch.float32).unsqueeze(0)
            ain_n = (ain - self._in_mean) / (self._in_std + 1e-8)
            aout = self._adapter(ain_n.view(1,-1))
            aout = aout * self._out_std + self._out_mean
            aout_np = aout.detach().cpu().numpy().reshape(-1).astype(np.float32)
        except Exception:
            aout_np = np.zeros(15, dtype=np.float32)
            _warn("adapter forward falló; usando ceros.")

        obs_prop = np.concatenate([
            ang_vel * SCALE_ANG_VEL,
            rpy[:2],
            np.array([math.sin(dyaw), math.cos(dyaw)], dtype=np.float32),
            (q - G1_DEFAULT_DOF_POS),
            dq * SCALE_DOF_VEL,
            self._last_action,
            gait_obs,
            aout_np,
        ]).astype(np.float32)

        obs_demo = np.zeros(8 + 3 + 3 + 3, dtype=np.float32)
        obs_demo[:8] = q[15:]
        obs_demo[8]  = commands[0]
        obs_demo[9]  = commands[2]
        obs_demo[11] = commands[4]; obs_demo[12] = commands[5]; obs_demo[13] = commands[6]
        obs_demo[14:17] = 0.75 + commands[3]

        obs_priv = np.zeros(self._n_priv, dtype=np.float32)
        self._prop_hist.pop(0); self._prop_hist.append(obs_prop.copy())
        self._extra_hist.pop(0); self._extra_hist.append(obs_prop.copy())
        obs_hist = np.array(self._prop_hist, dtype=np.float32).flatten()

        obs = np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist), dtype=np.float32)

        self._gait_cycle[:] = np.remainder(self._gait_cycle + _DT_CTRL * self._gait_freq, 1.0).astype(np.float32)
        if in_place and ((abs(self._gait_cycle[0]-0.25)<0.05) or (abs(self._gait_cycle[1]-0.25)<0.05)):
            self._gait_cycle[:] = 0.25
        if (not in_place) and (abs(self._gait_cycle[0]-0.25)<0.05) and (abs(self._gait_cycle[1]-0.25)<0.05):
            self._gait_cycle[:] = [0.25, 0.75]

        if not hasattr(self, "_obs_logged"):
            _log(f"obs_dim={obs.shape[0]} (prop={self._n_proprio}, hist={HIST_LEN}x{self._n_proprio})")
            self._obs_logged = True
        return obs

    def _policy_step(self, age_sec: float):
        obs = self._compute_observation()
        extra_hist = np.array(self._extra_hist, dtype=np.float32).flatten()

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(self._device).float().unsqueeze(0)
            eh_t  = torch.from_numpy(extra_hist).to(self._device).float().unsqueeze(0)
            raw = self._policy(obs_t, eh_t).detach().cpu().numpy().reshape(-1).astype(np.float32)

        raw = np.clip(raw, -40., 40.).astype(np.float32)
        scaled = raw * ACTION_SCALE

        ramp = max(0.0, min(1.0, age_sec / max(1e-3, ACTION_RAMP_SEC)))
        scaled *= ramp

        q, _ = self._get_qdq_in_amo_order()
        self._last_action = np.concatenate([raw.copy(), (q - G1_DEFAULT_DOF_POS)[15:] / ACTION_SCALE]).astype(np.float32)

        pd_target = G1_DEFAULT_DOF_POS.copy()
        pd_target[:15] += scaled

        if age_sec < WARMUP_SECONDS:
            blend = 1.0 - (age_sec / max(1e-3, WARMUP_SECONDS))
            for jname, off in START_POSE_OFFSETS.items():
                try:
                    amo_idx = AMO_DOF_ORDER.index(jname)
                except ValueError:
                    continue
                pd_target[amo_idx] = (1.0 - blend) * pd_target[amo_idx] + blend * (G1_DEFAULT_DOF_POS[amo_idx] + off)

        return pd_target

    def _apply_targets(self, q_target, dt):
        q_cmd = self._filter_targets(q_target, self._q_cmd_prev, dt)
        self._q_cmd_prev = q_cmd.copy()

        for amo_i, ix in enumerate(self._IDX_ISAAC_FROM_AMO):
            if ix is None or ix < 0: continue
            h = self._DOF_HANDLES[ix]
            try:
                q  = float(self.dc.get_dof_position(h))
                dq = float(self.dc.get_dof_velocity(h))
                kp = float(G1_STIFFNESS[amo_i]) if amo_i < len(G1_STIFFNESS) else 100.0
                kd = float(G1_DAMPING[amo_i])   if amo_i < len(G1_DAMPING) else 2.0
                lim= float(G1_TORQUE_LIMS[amo_i]) if amo_i < len(G1_TORQUE_LIMS) else 50.0
                tau = kp*(q_cmd[amo_i] - q) - kd*dq
                if tau >  lim: tau =  lim
                if tau < -lim: tau = -lim
                self.dc.apply_dof_effort(h, tau)
            except Exception:
                pass

    # ---------- gamepad ----------
    def _start_gamepad(self):
        if not USE_GAMEPAD:
            _warn("gamepad OFF (config).")
            return
        try:
            import pygame
            pygame.init(); pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                _warn("gamepad: no hay joystick; sigo sin mando.")
                return
            js = pygame.joystick.Joystick(0); js.init()
            _log(f"gamepad: {js.get_name()}")

            def _poll():
                last_hat = (0,0)
                last_btn = {}
                while not self._gp_stop:
                    try:
                        pygame.event.pump()
                        with self._cmd_lock:
                            ly = -js.get_axis(1)
                            lx = -js.get_axis(0)
                            rx = -js.get_axis(2)
                            ry = -js.get_axis(3)
                            self._cmd_vec[0] = ly*GP_SCALE_FWD   if abs(ly) > GAMEPAD_DEADZONE else 0.0
                            self._cmd_vec[1] = lx*GP_SCALE_YAW   if abs(lx) > GAMEPAD_DEADZONE else 0.0
                            self._cmd_vec[2] = rx*GP_SCALE_STRAF if abs(rx) > GAMEPAD_DEADZONE else 0.0
                            self._cmd_vec[3] = ry*GP_SCALE_DZ    if abs(ry) > GAMEPAD_DEADZONE else 0.0
                            hat = js.get_hat(0) if js.get_numhats() > 0 else (0,0)
                            btn = {i: js.get_button(i) for i in range(js.get_numbuttons())}
                            if hat[0] == -1 and last_hat[0] != -1: self._cmd_vec[4] += GP_STEP_TORSO
                            if hat[0] == +1 and last_hat[0] != +1: self._cmd_vec[4] -= GP_STEP_TORSO
                            if btn.get(6,0) and not last_btn.get(6,0): self._cmd_vec[5] += GP_STEP_TORSO
                            if btn.get(7,0) and not last_btn.get(7,0): self._cmd_vec[5] -= GP_STEP_TORSO
                            if btn.get(4,0) and not last_btn.get(4,0): self._cmd_vec[6] -= GP_STEP_TORSO
                            if btn.get(5,0) and not last_btn.get(5,0): self._cmd_vec[6] += GP_STEP_TORSO
                            last_hat = hat; last_btn = btn
                        time.sleep(0.01)
                    except Exception as e:
                        _warn(f"gamepad thread: {e}")
                        time.sleep(0.2)
            t = threading.Thread(target=_poll, daemon=True)
            t.start()
            return t
        except Exception as e:
            _warn(f"gamepad init falló: {e}")

    # ---------- lifecycle ----------
    def _build(self):
        if not self._check_paths(): return False
        if not self._get_mapping_and_handles(): return False
        self._load_nets()
        self._start_t = float(self.timeline.get_current_time())
        self._last_t  = self._start_t
        self._accum   = 0.0
        if self._gp_thread is None:
            self._gp_stop = False
            self._gp_thread = self._start_gamepad()
        self._built = True
        _log("BUILD OK (torque PD).")
        return True

    def _on_timeline(self, ev):
        try:
            if self.timeline.is_playing():
                _log("PLAY detectado.")
                if not self._built:
                    self._built = self._build()
            else:
                _log("PAUSE/STOP detectado.")
        except Exception as e:
            _warn(f"timeline cb: {e}")

    def _on_update(self, ev):
        try:
            playing = bool(self.timeline.is_playing())
            now = float(self.timeline.get_current_time())
            if not playing:
                self._last_t = now
                return

            if not self._built:
                if not self._build():
                    return

            dt = max(1e-4, now - self._last_t)
            self._last_t = now
            self._accum += dt

            # diag cada ~1 s
            self._tick_diag_t += dt
            if self._tick_diag_t >= 1.0:
                self._tick_diag_t = 0.0
                _log(f"update alive: playing={playing}, built={self._built}, dt={dt:.4f}")

            while self._accum >= _DT_CTRL:
                self._accum -= _DT_CTRL
                age = now - self._start_t
                q_des = self._policy_step(age)
                self._apply_targets(q_des, _DT_CTRL)
        except Exception:
            _err("Excepción en update:\n"+traceback.format_exc())

    def shutdown(self):
        try:
            self._gp_stop = True
        except: pass
        _log("Cleanup done.")

# ---------- RUN ----------
_controller_singleton = AmoController()
atexit.register(_controller_singleton.shutdown)

