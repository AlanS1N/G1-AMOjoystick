# === AMO policy on Isaac Sim (G1) + Joystick (pygame) ===
# Extra stability: action ramp at startup + widened base pose + low-pass filtering
import math, os, atexit, traceback, types, time, threading
import numpy as np
import torch
import carb
from pxr import Usd, UsdPhysics

import omni.usd, omni.timeline
from omni.kit.app import get_app
from omni.isaac.dynamic_control import _dynamic_control as dynamic_control

# ---------------- CONFIG ----------------
# Folder with amo_jit.pt, adapter_jit.pt, adapter_norm_stats.pt
POLICIES_DIR = "/home/alan/Desktop/policies"
POLICY_JIT   = "amo_jit.pt"
ADAPTER_JIT  = "adapter_jit.pt"
ADAPTER_STATS= "adapter_norm_stats.pt"

# Scene paths
ART_PATH         = "/World/g1/pelvis"          # Articulation root
JOINTS_SCAN_ROOT = "/World/g1"                 # Where joints are located
IMU_PATH         = "/World/g1/pelvis/imu_in_torso"  # IMU body (fallback to zeros if missing)

# High-level commands (overridden by gamepad if present)
CMD = {
    "vx": 0.7,   # forward velocity (standing if |vx| < 0.1)
    "vy": 0.0,   # lateral velocity (not used by AMO policy, kept for completeness)
    "yaw": 0.0,  # desired yaw rate
    "dz":  0.0,  # height offset (policy uses 0.75 + dz)
    "tyaw":  0.0,  # torso yaw offset
    "tpit":  0.0,  # torso pitch offset
    "trol":  0.0,  # torso roll offset
    "arms":  0.0,  # arm toggle (not used here)
}

# Filtering and control timing
TARGET_FILTER_TAU = 0.09     # larger = smoother (less twitch)
CONTROL_HZ        = 75.0     # control loop frequency

# Startup stability ramps
WIDEN_SECS        = 1.25     # blend from widened stance -> policy pose
ACTION_RAMP_SECS  = 1.25     # scale actions 0->1 at startup
# Temporary offsets over default pose to improve initial stability
WIDEN_OFFSETS = {
    "left_hip_roll":   +0.12,
    "right_hip_roll":  -0.12,
    "left_ankle_roll": -0.08,
    "right_ankle_roll":+0.08,
    "waist_pitch":     -0.10,
    "left_knee":       +0.05,
    "right_knee":      +0.05,
}

# G1 gains/limits (from AMO viewer defaults)
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

# USD Drive defaults (applied per joint; can be overridden by name)
USD_DRIVE_DEFAULT = dict(stiffness=160.0, damping=10.0, max_force=220.0)
USD_DRIVE_OVERRIDES = {"waist_yaw_joint": dict(stiffness=200.0, damping=12.0, max_force=260.0)}

# AMO observation/scale parameters (match play_amo)
ACTION_SCALE   = 0.25
SCALE_ANG_VEL  = 0.25
SCALE_DOF_VEL  = 0.05
HIST_LEN       = 10
EXTRA_HIST_LEN = 25
# ----------------------------------------

# Global state
dc = dynamic_control.acquire_dynamic_control_interface()
usd_ctx = omni.usd.get_context()
timeline = omni.timeline.get_timeline_interface()

_SETTINGS_KEY = "/amo_isaac_bridge_active"
_SUB = _TL_SUB = None
_BUILT = False
_LAST_PLAY = False
_LAST_T = 0.0
_ACCUM = 0.0
_DT_CTRL = 1.0 / CONTROL_HZ

_ART = 0
_DOF_HANDLES = []
_DOF_NAMES = []
_IDX_ISAAC_FROM_AMO = []
_USE_TORQUE_FALLBACK = False

# Networks and stats
_device = "cuda" if torch.cuda.is_available() else "cpu"
_policy = None
_adapter = None
_in_mean=_in_std=_out_mean=_out_std = None

# AMO history buffers
_nj = 23
_last_action = np.zeros(_nj, dtype=np.float32)
_gait_cycle = np.array([0.25, 0.75], dtype=np.float32)
_gait_freq  = 1.3
_in_place   = (abs(CMD["vx"]) < 0.1)

_n_priv = 3
# proprio length: 3(ang vel) +2(rpy[:2]) +2(sin/cos dyaw) + 23(q - default) + 23(dq) + 23(last_action) + 2(gait) + 15(adapter out)
_n_proprio = 3 + 2 + 2 + 23 + 23 + 23 + 2 + 15
_prop_hist  = [np.zeros(_n_proprio, dtype=np.float32) for _ in range(HIST_LEN)]
_extra_hist = [np.zeros(_n_proprio, dtype=np.float32) for _ in range(EXTRA_HIST_LEN)]

# AMO DOF base names (without "_joint")
AMO_DOF_ORDER = [
    "left_hip_pitch","left_hip_roll","left_hip_yaw","left_knee","left_ankle_pitch","left_ankle_roll",
    "right_hip_pitch","right_hip_roll","right_hip_yaw","right_knee","right_ankle_pitch","right_ankle_roll",
    "waist_yaw","waist_roll","waist_pitch",
    "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow",
    "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow",
]

def _stage(): return usd_ctx.get_stage()
def _is_playing():
    try: return bool(timeline.is_playing())
    except: return False

def _norm(nm: str) -> str:
    """Normalize USD joint names to base (strip common suffixes)."""
    s = (nm or "").lower()
    for suf in ("_joint","joint","_dof","_dof0","_dof1","_dof2"):
        if s.endswith(suf): s = s[:-len(suf)]
    return s

def _force_drives(scan_root: str):
    """Ensure DriveAPI (stiffness/damping/max_force) is present on all joints under scan_root."""
    st = _stage()
    root = st.GetPrimAtPath(scan_root)
    if not root or not root.IsValid():
        carb.log_warn("[amo-isaac] Couldn't find scan_root to apply Drives.")
        return 0
    try: st.SetEditTarget(st.GetRootLayer())
    except: pass
    seen=ok=0
    for p in Usd.PrimRange(root):
        kind=None
        if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.SphericalJoint): kind="angular"
        elif p.IsA(UsdPhysics.PrismaticJoint): kind="linear"
        if not kind: continue
        seen+=1
        name = p.GetName()
        cfg = USD_DRIVE_OVERRIDES.get(name, USD_DRIVE_DEFAULT)
        try:
            drv = UsdPhysics.DriveAPI.Get(p, kind)
            if not drv or not drv.GetPrim().IsValid(): drv = UsdPhysics.DriveAPI.Apply(p, kind)
            drv.CreateStiffnessAttr().Set(float(cfg["stiffness"]))
            drv.CreateDampingAttr().Set(float(cfg["damping"]))
            drv.CreateMaxForceAttr().Set(float(cfg["max_force"]))
            drv.CreateTargetPositionAttr().Set(0.0)
            drv.CreateTargetVelocityAttr().Set(0.0)
            ok += 1
        except Exception as e:
            carb.log_warn(f"[amo-isaac] Drive on {name}: {e}")
    if ok: carb.log_info(f"[amo-isaac] PD Drives applied on {ok}/{seen} joints.")
    else:  carb.log_warn(f"[amo-isaac] Found {seen} joints but couldn't apply Drives.")
    return ok

def _get_mapping_and_handles():
    """Build Isaac->AMO joint index mapping and cache DOF handles."""
    global _ART, _DOF_HANDLES, _DOF_NAMES, _IDX_ISAAC_FROM_AMO
    _ART = dc.get_articulation(ART_PATH)
    if not _ART:
        carb.log_error(f"[amo-isaac] No articulation at {ART_PATH}")
        return False
    ndof = dc.get_articulation_dof_count(_ART)
    _DOF_HANDLES = [dc.get_articulation_dof(_ART,i) for i in range(ndof)]
    _DOF_NAMES=[]
    for h in _DOF_HANDLES:
        try: _DOF_NAMES.append((dc.get_dof_name(h) or "dof"))
        except: _DOF_NAMES.append("dof")
    carb.log_info(f"[amo-isaac] Isaac DOFs({ndof}): {_DOF_NAMES}")

    name_to_idx = {_norm(n): i for i,n in enumerate(_DOF_NAMES)}
    _IDX_ISAAC_FROM_AMO = []
    miss=[]
    for base in AMO_DOF_ORDER:
        i = name_to_idx.get(base) or name_to_idx.get(base.replace("waist","pelvis"))
        if i is None: miss.append(base)
        _IDX_ISAAC_FROM_AMO.append(i if i is not None else -1)
    if miss:
        carb.log_warn(f"[amo-isaac] Unmapped {len(miss)} DOFs (name mismatch): {miss}")
    return True

def _load_nets():
    """Load TorchScript policy + adapter and normalization stats."""
    global _policy, _adapter, _in_mean,_in_std,_out_mean,_out_std
    ppth = os.path.join(POLICIES_DIR, POLICY_JIT)
    apth = os.path.join(POLICIES_DIR, ADAPTER_JIT)
    spth = os.path.join(POLICIES_DIR, ADAPTER_STATS)
    _policy  = torch.jit.load(ppth, map_location=_device)
    _adapter = torch.jit.load(apth, map_location=_device)
    _policy.eval(); _adapter.eval()
    for m in (_policy, _adapter):
        for p in m.parameters(): p.requires_grad=False
    stats = torch.load(spth, weights_only=False, map_location="cpu")
    _in_mean  = torch.tensor(stats["input_mean"],  dtype=torch.float32, device=_device)
    _in_std   = torch.tensor(stats["input_std"],   dtype=torch.float32, device=_device)
    _out_mean = torch.tensor(stats["output_mean"], dtype=torch.float32, device=_device)
    _out_std  = torch.tensor(stats["output_std"],  dtype=torch.float32, device=_device)
    carb.log_info("[amo-isaac] TorchScript policies loaded.")

def _quat_to_euler_wxyz(q):
    """Quaternion (w,x,y,z) to Euler (roll,pitch,yaw)."""
    qw,qx,qy,qz = q
    sinr = 2*(qw*qx + qy*qz); cosr = 1 - 2*(qx*qx + qy*qy)
    roll = math.atan2(sinr, cosr)
    sinp = 2*(qw*qy - qz*qx)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp)>=1 else math.asin(sinp)
    siny = 2*(qw*qz + qx*qy); cosy = 1 - 2*(qy*qy + qz*qz)
    yaw = math.atan2(siny, cosy)
    return np.array([roll, pitch, yaw], dtype=np.float32)

def _read_imu_quat_wxyz_and_avel():
    """Read IMU orientation (wxyz) and angular velocity; fallback to zeros."""
    try:
        rb = dc.get_rigid_body(IMU_PATH)
        if rb:
            pose = dc.get_rigid_body_pose(rb)
            if hasattr(pose, "r") and hasattr(pose.r, "w"):
                q = [pose.r.w, pose.r.x, pose.r.y, pose.r.z]
            elif hasattr(pose, "r") and len(pose.r)==4:
                q = [pose.r[3], pose.r[0], pose.r[1], pose.r[2]]
            else:
                q = [1.0,0.0,0.0,0.0]
            av = dc.get_rigid_body_angular_velocity(rb)
            avel = np.array([av.x, av.y, av.z], dtype=np.float32) if hasattr(av,"x") else np.array(av, dtype=np.float32)
            return np.array(q, dtype=np.float32), avel
    except Exception:
        pass
    return np.array([1.0,0.0,0.0,0.0], dtype=np.float32), np.zeros(3, dtype=np.float32)

def _get_qdq_in_amo_order():
    """Read Isaac DOF positions/velocities and reorder into AMO joint order."""
    q_raw = []; dq_raw=[]
    for h in _DOF_HANDLES:
        try:
            q_raw.append(float(dc.get_dof_position(h)))
            dq_raw.append(float(dc.get_dof_velocity(h)))
        except:
            q_raw.append(0.0); dq_raw.append(0.0)
    q_raw = np.array(q_raw, dtype=np.float32)
    dq_raw= np.array(dq_raw, dtype=np.float32)
    q = np.zeros(_nj, dtype=np.float32); dq= np.zeros(_nj, dtype=np.float32)
    for amo_i, ix in enumerate(_IDX_ISAAC_FROM_AMO):
        if ix is None or ix<0 or ix>=len(q_raw): continue
        q[amo_i]  = q_raw[ix]; dq[amo_i] = dq_raw[ix]
    return q, dq

def _make_widen_stance(base):
    """Apply temporary widened-stance offsets on top of base pose."""
    out = base.copy()
    name_to_idx = {
        "left_hip_roll": 1,  "right_hip_roll": 7,
        "left_ankle_roll": 5, "right_ankle_roll": 11,
        "waist_pitch": 14, "left_knee": 3, "right_knee": 9,
    }
    for k, v in WIDEN_OFFSETS.items():
        i = name_to_idx.get(k, None)
        if i is not None: out[i] += v
    return out

def _ramp_factor(now, T):
    """0→1 ramp factor from _START_TIME to _START_TIME+T."""
    if T <= 0.0: return 1.0
    a = (now - _START_TIME) / T
    if a <= 0.0: return 0.0
    if a >= 1.0: return 1.0
    return a

# Control state
_q_cmd_prev = G1_DEFAULT_DOF_POS.copy()
_START_TIME = 0.0

def _build():
    """One-time setup on PLAY: drives, mapping, networks, ramp baselines."""
    global _USE_TORQUE_FALLBACK, _LAST_T, _ACCUM, _q_cmd_prev
    _force_drives(JOINTS_SCAN_ROOT)
    if not _get_mapping_and_handles(): return False
    _load_nets()
    _USE_TORQUE_FALLBACK = False
    try:
        if _DOF_HANDLES: dc.set_dof_position_target(_DOF_HANDLES[0], 0.0)
    except Exception:
        _USE_TORQUE_FALLBACK = True
        carb.log_warn("[amo-isaac] Runtime doesn't support position targets; falling back to torque PD.")
    _LAST_T = float(timeline.get_current_time()); _ACCUM = 0.0
    # initialize filter from widened stance to avoid a jump
    _q_cmd_prev = _make_widen_stance(G1_DEFAULT_DOF_POS).copy()
    carb.log_info("[amo-isaac] Build OK.")
    return True

def _filter_targets(q_des, q_prev, dt):
    """Per-component exponential low-pass filter on target positions."""
    if dt <= 0.0: return q_des
    alpha = 1.0 - math.exp(-dt / max(1e-4, TARGET_FILTER_TAU))
    return (1.0 - alpha) * q_prev + alpha * q_des

def _compute_observation():
    """Assemble AMO observation vector (matches play_amo)."""
    global _gait_cycle, _in_place, _last_action
    quat_wxyz, ang_vel = _read_imu_quat_wxyz_and_avel()
    rpy = _quat_to_euler_wxyz(quat_wxyz)
    q, dq = _get_qdq_in_amo_order()

    # command vector (viewer.commands equivalent)
    commands = np.zeros(8, dtype=np.float32)
    commands[:] = [CMD["vx"], CMD["yaw"], CMD["vy"], CMD["dz"], CMD["tyaw"], CMD["tpit"], CMD["trol"], CMD["arms"]]
    target_yaw = commands[1]
    dyaw = rpy[2] - target_yaw
    dyaw = (dyaw + math.pi) % (2*math.pi) - math.pi
    _in_place = (abs(commands[0]) < 0.1)
    if _in_place: dyaw = 0.0

    gait_obs = np.sin(_gait_cycle * 2 * math.pi).astype(np.float32)

    # adapter input: [height_cmd, torso yaw/pitch/roll, 8 arm DOFs]
    adapter_in = np.zeros(12, dtype=np.float32)
    adapter_in[0] = 0.75 + commands[3]
    adapter_in[1] = commands[4]; adapter_in[2] = commands[5]; adapter_in[3] = commands[6]
    adapter_in[4:] = q[15:]  # arm DOFs
    ain = torch.tensor(adapter_in, device=_device, dtype=torch.float32).unsqueeze(0)
    ain_n = (ain - _in_mean) / (_in_std + 1e-8)
    aout = _adapter(ain_n.view(1,-1))
    aout = aout * _out_std + _out_mean
    aout_np = aout.detach().cpu().numpy().reshape(-1).astype(np.float32)

    obs_prop = np.concatenate([
        ang_vel * SCALE_ANG_VEL,
        rpy[:2],
        np.array([math.sin(dyaw), math.cos(dyaw)], dtype=np.float32),
        (q - G1_DEFAULT_DOF_POS),
        dq * SCALE_DOF_VEL,
        _last_action,
        gait_obs,
        aout_np,
    ]).astype(np.float32)

    # demo/priv/history (kept for compatibility with play_amo signature)
    obs_demo = np.zeros(8 + 3 + 3 + 3, dtype=np.float32)
    obs_demo[:8] = q[15:]
    obs_demo[8]  = commands[0]  # vx
    obs_demo[9]  = commands[2]  # vy
    obs_demo[11] = commands[4]; obs_demo[12] = commands[5]; obs_demo[13] = commands[6]
    obs_demo[14:17] = 0.75 + commands[3]

    obs_priv = np.zeros(_n_priv, dtype=np.float32)

    _prop_hist.pop(0); _prop_hist.append(obs_prop.copy())
    _extra_hist.pop(0); _extra_hist.append(obs_prop.copy())
    obs_hist = np.array(_prop_hist, dtype=np.float32).flatten()

    obs = np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist), dtype=np.float32)

    # gait phase update
    _gait_cycle = np.remainder(_gait_cycle + _DT_CTRL * _gait_freq, 1.0).astype(np.float32)
    if _in_place and ((abs(_gait_cycle[0]-0.25)<0.05) or (abs(_gait_cycle[1]-0.25)<0.05)):
        _gait_cycle[:] = 0.25
    if (not _in_place) and (abs(_gait_cycle[0]-0.25)<0.05) and (abs(_gait_cycle[1]-0.25)<0.05):
        _gait_cycle[:] = [0.25, 0.75]

    return obs

def _policy_step():
    """Run policy once, apply startup ramps, return 23-DOF position targets."""
    global _last_action
    obs = _compute_observation()
    extra_hist = np.array(_extra_hist, dtype=np.float32).flatten()

    with torch.no_grad():
        obs_t = torch.from_numpy(obs).to(_device).float().unsqueeze(0)
        eh_t  = torch.from_numpy(extra_hist).to(_device).float().unsqueeze(0)
        raw = _policy(obs_t, eh_t).detach().cpu().numpy().reshape(-1).astype(np.float32)

    raw = np.clip(raw, -40., 40.).astype(np.float32)

    tnow = float(timeline.get_current_time())
    r_act   = _ramp_factor(tnow, ACTION_RAMP_SECS)  # action ramp
    r_pose  = _ramp_factor(tnow, WIDEN_SECS)        # widened stance blend

    scaled = raw * (ACTION_SCALE * r_act)

    q, _ = _get_qdq_in_amo_order()
    _last_action = np.concatenate([raw.copy(), (q - G1_DEFAULT_DOF_POS)[15:] / ACTION_SCALE]).astype(np.float32)

    pd_target = G1_DEFAULT_DOF_POS.copy()
    pd_target[:15] += scaled

    # blend widened stance -> policy
    widen = _make_widen_stance(G1_DEFAULT_DOF_POS)
    pd_target = (1.0 - r_pose) * widen + r_pose * pd_target
    return pd_target

def _apply_targets(q_target, dt):
    """Send position targets (or torque PD fallback) to Isaac joints."""
    global _q_cmd_prev, _USE_TORQUE_FALLBACK
    q_cmd = _filter_targets(q_target, _q_cmd_prev, dt)
    _q_cmd_prev = q_cmd.copy()

    # Preferred: use position targets if supported
    if not _USE_TORQUE_FALLBACK:
        try:
            for amo_i, ix in enumerate(_IDX_ISAAC_FROM_AMO):
                if ix is None or ix < 0: continue
                h = _DOF_HANDLES[ix]
                dc.set_dof_position_target(h, float(q_cmd[amo_i]))
            return
        except Exception:
            _USE_TORQUE_FALLBACK = True
            carb.log_warn("[amo-isaac] Switching to torque PD (Drives not operational).")

    # Fallback: explicit torque PD per-joint
    for amo_i, ix in enumerate(_IDX_ISAAC_FROM_AMO):
        if ix is None or ix < 0: continue
        h = _DOF_HANDLES[ix]
        try:
            q  = float(dc.get_dof_position(h))
            dq = float(dc.get_dof_velocity(h))
            kp = float(G1_STIFFNESS[amo_i]) if amo_i < len(G1_STIFFNESS) else 100.0
            kd = float(G1_DAMPING[amo_i])   if amo_i < len(G1_DAMPING) else 2.0
            lim= float(G1_TORQUE_LIMS[amo_i]) if amo_i < len(G1_TORQUE_LIMS) else 50.0
            tau = kp*(q_cmd[amo_i] - q) - kd*dq
            tau = max(-lim, min(lim, tau))
            dc.apply_dof_effort(h, tau)
        except Exception:
            pass

# ---------- Joystick (pygame, optional) ----------
_gamepad_thread = None
_gamepad_alive  = False
def _start_gamepad_thread():
    """Start background thread to read a gamepad via pygame (Switch Pro mapping tested)."""
    global _gamepad_thread, _gamepad_alive
    try:
        import pygame
        pygame.init(); pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            carb.log_warn("[amo-isaac] gamepad: no joystick detected. Continuing without gamepad.")
            return
        js = pygame.joystick.Joystick(0); js.init()
        carb.log_info(f"[amo-isaac] gamepad detected: {js.get_name()}")
        deadzone = 0.15

        def loop():
            global _gamepad_alive
            _gamepad_alive = True
            last_hat = (0,0)
            last_btn = {}
            while _gamepad_alive:
                try:
                    pygame.event.pump()
                    # Linux mapping commonly used by Switch Pro Controller
                    AX_LX, AX_LY, AX_RX, AX_RY = 0, 1, 2, 3
                    # Sticks (invert vertical axes)
                    ly = -js.get_axis(AX_LY)
                    lx = -js.get_axis(AX_LX)
                    rx = -js.get_axis(AX_RX)
                    ry = -js.get_axis(AX_RY)
                    CMD["vx"]  = ly*1.2 if abs(ly)>deadzone else 0.0
                    CMD["yaw"] = lx*1.2 if abs(lx)>deadzone else 0.0
                    CMD["vy"]  = rx*0.5 if abs(rx)>deadzone else 0.0
                    CMD["dz"]  = ry*0.5 if abs(ry)>deadzone else 0.0

                    # Buttons: ZL/ZR (6/7), L/R (4/5)
                    cur_btn = {b: js.get_button(b) for b in range(js.get_numbuttons())}
                    hat = js.get_hat(0) if js.get_numhats()>0 else (0,0)

                    # D-Pad -> torso yaw nudges
                    if hat[0] == -1 and last_hat[0] != -1: CMD["tyaw"] += 0.1
                    elif hat[0] ==  1 and last_hat[0] !=  1: CMD["tyaw"] -= 0.1

                    # ZL/ZR -> torso pitch nudges
                    if cur_btn.get(6,0) and not last_btn.get(6,0): CMD["tpit"] += 0.05
                    if cur_btn.get(7,0) and not last_btn.get(7,0): CMD["tpit"] -= 0.05

                    # L/R -> torso roll nudges
                    if cur_btn.get(4,0) and not last_btn.get(4,0): CMD["trol"] -= 0.05
                    if cur_btn.get(5,0) and not last_btn.get(5,0): CMD["trol"] += 0.05

                    last_hat = hat; last_btn = cur_btn
                    time.sleep(0.01)
                except Exception as e:
                    carb.log_warn(f"[amo-isaac] gamepad error: {e}")
                    time.sleep(0.5)
            try:
                js.quit()
            except: pass

        _gamepad_thread = threading.Thread(target=loop, daemon=True)
        _gamepad_thread.start()
    except Exception as e:
        carb.log_warn(f"[amo-isaac] gamepad unavailable ({e}). Running without gamepad.")

def _stop_gamepad_thread():
    """Stop gamepad thread on shutdown."""
    global _gamepad_alive
    _gamepad_alive = False
# ----------------------------------------------------

def _on_timeline(ev):
    """Handle PLAY toggles: rebuild on transition to PLAY and start timers."""
    playing = _is_playing()
    global _LAST_PLAY, _BUILT, _ACCUM, _LAST_T, _START_TIME
    if playing and not _LAST_PLAY:
        carb.log_info("[amo-isaac] PLAY → rebuild")
        _BUILT=False; _ACCUM = 0.0
        _LAST_T = float(timeline.get_current_time())
        _START_TIME = _LAST_T             # mark ramp start
    _LAST_PLAY = playing

def _on_update(ev):
    """Main update loop (Kit update stream): run control at CONTROL_HZ while playing."""
    try:
        if not _is_playing(): return
        global _BUILT, _ACCUM, _LAST_T
        if not _BUILT:
            ok = _build()
            if not ok: return
            _BUILT = True

        tnow = float(timeline.get_current_time())
        dt = max(1e-4, tnow - _LAST_T)
        _LAST_T = tnow
        _ACCUM += dt

        while _ACCUM >= _DT_CTRL:
            _ACCUM -= _DT_CTRL
            q_des = _policy_step()
            _apply_targets(q_des, _DT_CTRL)
    except Exception:
        carb.log_error("[amo-isaac] Exception in update:\n"+traceback.format_exc())

def _cleanup():
    """Unsubscribe, release single-instance flag, stop gamepad thread."""
    try:
        if _TL_SUB is not None: _TL_SUB.unsubscribe()
    except: pass
    try:
        if _SUB is not None: _SUB.unsubscribe()
    except: pass
    try:
        import carb.settings
        carb.settings.get_settings().set(_SETTINGS_KEY, False)
    except: pass
    _stop_gamepad_thread()

def _init():
    """Register update/timeline subscriptions and start gamepad (single instance)."""
    global _SUB, _TL_SUB
    try:
        import carb.settings
        s = carb.settings.get_settings()
        if s.get(_SETTINGS_KEY):
            carb.log_warn("[amo-isaac] An instance is already active.")
            return
        s.set(_SETTINGS_KEY, True)
    except: pass

    app = get_app()
    _SUB = app.get_update_event_stream().create_subscription_to_pop(_on_update, name="amo_isaac_update")
    _TL_SUB = timeline.get_timeline_event_stream().create_subscription_to_pop(_on_timeline, name="amo_isaac_timeline")
    _start_gamepad_thread()
    carb.log_info("[amo-isaac] Initialized.")

atexit.register(_cleanup)
_init()