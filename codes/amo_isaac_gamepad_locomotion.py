# -----------------------------------------------------------------------------
# Copyright [2025] [Jialong Li, Xuxin Cheng, Tianshu Huang, Xiaolong Wang]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script is based on an initial draft generously provided by Zixuan Chen.
# -----------------------------------------------------------------------------

# === AMO policy on Isaac Sim (G1) — Gamepad + Locomotion Modes (walk/sprint/crouch/jump) ===
# Paste into Isaac Sim 4.5.0 Script Editor and press PLAY.
# Keeps AMO-compatible observations/history/adapter and adds gamepad-driven locomotion modes.
import math, os, atexit, traceback, types, threading, time
import carb
from pxr import Usd, UsdPhysics
import numpy as np
import torch

import omni.usd, omni.timeline
from omni.kit.app import get_app
from omni.isaac.dynamic_control import _dynamic_control as dynamic_control

# ---------------- CONFIG ----------------
POLICIES_DIR = "/home/alan/Desktop/policies"   # <- adjust to your policy directory
POLICY_JIT   = "amo_jit.pt"
ADAPTER_JIT  = "adapter_jit.pt"
ADAPTER_STATS= "adapter_norm_stats.pt"

ART_PATH         = "/World/g1/pelvis"          # Articulation root prim
JOINTS_SCAN_ROOT = "/World/g1"                 # Root to scan for joints (for USD drives)
IMU_PATH         = "/World/g1/pelvis/imu_in_torso"

# Base command (will be overwritten at runtime by the gamepad thread)
# [vx, yaw, vy, dz, torso_yaw, torso_pitch, torso_roll, arms_toggle]
_cmd = np.zeros(8, dtype=np.float32)

# Filters and control rate
TARGET_FILTER_TAU = 0.075    # target low-pass time constant (s) for smoothness
CONTROL_HZ        = 50.0     # AMO update cadence (sim_dt=0.002, decimation=10)
ACTION_RAMP_TAU   = 0.15     # action ramp to soften mode transitions / start-up impulses

# PD gains / torque limits (taken from AMO reference)
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

# USD Drives (fallback PD via torque if position targets are not supported)
USD_DRIVE_DEFAULT = dict(stiffness=160.0, damping=10.0, max_force=220.0)
USD_DRIVE_OVERRIDES = {
    "waist_yaw_joint": dict(stiffness=200.0, damping=12.0, max_force=260.0),
}

# AMO observation scales/history lengths
ACTION_SCALE   = 0.25
SCALE_ANG_VEL  = 0.25
SCALE_DOF_VEL  = 0.05
HIST_LEN       = 10
EXTRA_HIST_LEN = 25

# ---------------- Isaac/NN global state ----------------
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
_IDX_ISAAC_FROM_AMO = []   # map: AMO index -> Isaac DOF index
_USE_TORQUE_FALLBACK = False

# NN handles/statistics
_device = "cuda" if torch.cuda.is_available() else "cpu"
_policy = None
_adapter = None
_in_mean=_in_std=_out_mean=_out_std = None

# AMO history / gait
_nj = 23
_last_action = np.zeros(_nj, dtype=np.float32)
_gait_cycle = np.array([0.25, 0.75], dtype=np.float32)
_gait_freq  = 1.3
_in_place   = True

_n_priv = 3
# 3(ang vel) +2(rpy[:2]) +2(sin/cos dyaw) + 23(pos- default) + 23(vel) + 23(last_action) + 2(gait) + 15(adapter out)
_n_proprio = 3 + 2 + 2 + 23 + 23 + 23 + 2 + 15
_prop_hist = [np.zeros(_n_proprio, dtype=np.float32) for _ in range(HIST_LEN)]
_extra_hist= [np.zeros(_n_proprio, dtype=np.float32) for _ in range(EXTRA_HIST_LEN)]

# AMO joint order (base names without "_joint" suffix)
AMO_DOF_ORDER = [
    "left_hip_pitch","left_hip_roll","left_hip_yaw","left_knee","left_ankle_pitch","left_ankle_roll",
    "right_hip_pitch","right_hip_roll","right_hip_yaw","right_knee","right_ankle_pitch","right_ankle_roll",
    "waist_yaw","waist_roll","waist_pitch",
    "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow",
    "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow",
]

# ---------------- Locomotion modes ----------------
# Button layout for Nintendo Switch Pro via pygame (typical mapping):
#   A = jump (sequence), B = toggle sprint, Y = toggle crouch, Minus = toggle arms
class _Locomotion:
    mode = "walk"            # descriptive only; we use booleans below
    sprint = False
    crouch = False
    jump_active = False
    jump_step = 0
    jump_t = 0.0
    last_mode_change = 0.0

    # mode parameters
    speed_walk   = 1.2
    speed_sprint = 2.4
    crouch_dz    = -0.18     # lower COM for stability
    crouch_speed = 0.6
    jump_crouch_t = 0.35
    jump_impulse_t= 0.25
    jump_recover_t=0.45
    jump_impulse  = 1.6

_loco = _Locomotion()
_last_scaled = np.zeros(15, dtype=np.float32)  # action ramp state (first 15 actuated joints)

# ---------------- Utilities ----------------
def _stage():
    return usd_ctx.get_stage()

def _is_playing():
    try: return bool(timeline.is_playing())
    except: return False

def _norm(nm: str) -> str:
    """Normalize DOF names to match AMO base naming (strip common suffixes)."""
    s = (nm or "").lower()
    for suf in ("_joint","joint","_dof","_dof0","_dof1","_dof2"):
        if s.endswith(suf):
            s = s[:-len(suf)]
    return s

def _force_drives(scan_root: str):
    """Ensure USD Physics DriveAPI is applied to all joints with reasonable defaults."""
    st = _stage()
    root = st.GetPrimAtPath(scan_root)
    if not root or not root.IsValid():
        carb.log_warn("[amo-isaac] Can't find scan_root for Drives.")
        return 0
    try: st.SetEditTarget(st.GetRootLayer())
    except: pass

    seen=ok=0
    for p in Usd.PrimRange(root):
        kind=None
        if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.SphericalJoint):
            kind="angular"
        elif p.IsA(UsdPhysics.PrismaticJoint):
            kind="linear"
        if not kind: continue
        seen+=1
        name = p.GetName()
        cfg = USD_DRIVE_OVERRIDES.get(name, USD_DRIVE_DEFAULT)
        try:
            drv = UsdPhysics.DriveAPI.Get(p, kind)
            if not drv or not drv.GetPrim().IsValid():
                drv = UsdPhysics.DriveAPI.Apply(p, kind)
            drv.CreateStiffnessAttr().Set(float(cfg["stiffness"]))
            drv.CreateDampingAttr().Set(float(cfg["damping"]))
            drv.CreateMaxForceAttr().Set(float(cfg["max_force"]))
            drv.CreateTargetPositionAttr().Set(0.0)
            drv.CreateTargetVelocityAttr().Set(0.0)
            ok += 1
        except Exception as e:
            carb.log_warn(f"[amo-isaac] Drive on {name}: {e}")
    if ok:
        carb.log_info(f"[amo-isaac] PD Drives applied on {ok}/{seen} joints.")
    else:
        carb.log_warn(f"[amo-isaac] Found {seen} joints but couldn't apply Drives.")
    return ok

def _get_mapping_and_handles():
    """Acquire articulation, DOF handles, and build AMO->Isaac DOF index map."""
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
        i = name_to_idx.get(base)
        if i is None:
            # allow pelvis_* aliases instead of waist_*
            i = name_to_idx.get(base.replace("waist","pelvis"))
        if i is None: miss.append(base)
        _IDX_ISAAC_FROM_AMO.append(i if i is not None else -1)
    if miss:
        carb.log_warn(f"[amo-isaac] Unmapped DOFs ({len(miss)} names mismatch): {miss}")
    return True

def _load_nets():
    """Load torchscript policy and adapter + normalization stats."""
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
    carb.log_info("[amo-isaac] TorchScript nets loaded.")

def _quat_to_euler_wxyz(q):
    """wxyz quaternion -> roll/pitch/yaw (radians)."""
    qw,qx,qy,qz = q
    # roll
    sinr = 2*(qw*qx + qy*qz); cosr = 1 - 2*(qx*qx + qy*qy)
    roll = math.atan2(sinr, cosr)
    # pitch
    sinp = 2*(qw*qy - qz*qx)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp)>=1 else math.asin(sinp)
    # yaw
    siny = 2*(qw*qz + qx*qy); cosy = 1 - 2*(qy*qy + qz*qz)
    yaw = math.atan2(siny, cosy)
    return np.array([roll, pitch, yaw], dtype=np.float32)

def _read_imu_quat_wxyz_and_avel():
    """Read IMU rigid body pose and angular velocity; fallback to identity if not available."""
    try:
        rb = dc.get_rigid_body(IMU_PATH)
        if rb:
            pose = dc.get_rigid_body_pose(rb)
            if hasattr(pose, "r") and hasattr(pose.r, "w"):
                q = [pose.r.w, pose.r.x, pose.r.y, pose.r.z]
            elif hasattr(pose, "r") and len(pose.r)==4:
                # assume (x,y,z,w)
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
    """Read Isaac q/dq and reorder to AMO indexing."""
    q_raw = []; dq_raw=[]
    for h in _DOF_HANDLES:
        try:
            q_raw.append(float(dc.get_dof_position(h)))
            dq_raw.append(float(dc.get_dof_velocity(h)))
        except:
            q_raw.append(0.0); dq_raw.append(0.0)
    q_raw = np.array(q_raw, dtype=np.float32)
    dq_raw= np.array(dq_raw, dtype=np.float32)

    q = np.zeros(_nj, dtype=np.float32)
    dq= np.zeros(_nj, dtype=np.float32)
    for amo_i, ix in enumerate(_IDX_ISAAC_FROM_AMO):
        if ix is None or ix<0 or ix>=len(q_raw): continue
        q[amo_i]  = q_raw[ix]
        dq[amo_i] = dq_raw[ix]
    return q, dq

def _apply_locomotion_modes(base_cmd):
    """
    Adjust base gamepad command with locomotion mode effects.
    base_cmd: np.array([vx, yaw, vy, dz, torso_yaw, torso_pitch, torso_roll, arms_toggle])
    Returns: modified cmd
    """
    cmd = base_cmd.copy()

    # sprint toggles speed multiplier (vx,yaw); crouch clamps speed and lowers body height
    spd = _loco.speed_sprint if _loco.sprint else _loco.speed_walk
    if _loco.crouch:
        spd = min(spd, _loco.crouch_speed)
        cmd[3] += _loco.crouch_dz  # lower COM

    cmd[0] *= spd
    cmd[1] *= spd
    cmd[2] *= 0.5  # strafe softer than forward

    # jump sequence (A button): crouch → upward impulse → recovery
    if _loco.jump_active:
        _loco.jump_t += _DT_CTRL
        if _loco.jump_step == 0:  # crouch pre-load
            cmd[3] += -0.15
            cmd[0] *= 0.6; cmd[1] *= 0.6
            if _loco.jump_t >= _loco.jump_crouch_t:
                _loco.jump_step = 1; _loco.jump_t = 0.0
        elif _loco.jump_step == 1:  # upward impulse
            phase = min(1.0, _loco.jump_t/_loco.jump_impulse_t)
            cmd[3] += _loco.jump_impulse * (1.0 - phase)  # linear decay
            cmd[0] *= 0.8; cmd[1] *= 0.8
            if _loco.jump_t >= _loco.jump_impulse_t:
                _loco.jump_step = 2; _loco.jump_t = 0.0
        elif _loco.jump_step == 2:  # recovery damping
            cmd[3] += -0.08
            if _loco.jump_t >= _loco.jump_recover_t:
                _loco.jump_active = False
                _loco.jump_step = 0; _loco.jump_t = 0.0
    return cmd

def _build():
    """One-time setup when PLAY starts: drives, mapping, load networks, test position targets."""
    global _BUILT, _USE_TORQUE_FALLBACK, _LAST_T, _ACCUM
    _force_drives(JOINTS_SCAN_ROOT)
    if not _get_mapping_and_handles():
        return False
    _load_nets()

    _USE_TORQUE_FALLBACK = False
    try:
        if _DOF_HANDLES:
            dc.set_dof_position_target(_DOF_HANDLES[0], 0.0)
    except Exception:
        _USE_TORQUE_FALLBACK = True
        carb.log_warn("[amo-isaac] Runtime doesn't accept position targets; falling back to PD torque.")

    _LAST_T = float(timeline.get_current_time())
    _ACCUM = 0.0
    carb.log_info("[amo-isaac] Build OK.")
    return True

def _filter_targets(q_des, q_prev, dt):
    """Exponential smoothing per-DOF on desired position targets."""
    if dt <= 0.0: return q_des
    alpha = 1.0 - math.exp(-dt / max(1e-4, TARGET_FILTER_TAU))
    return (1.0 - alpha) * q_prev + alpha * q_des

_q_cmd_prev = G1_DEFAULT_DOF_POS.copy()

def _compute_observation():
    """Assemble AMO observation (proprio + demo + priv + history) including adapter output."""
    global _gait_cycle, _in_place, _last_action
    # IMU and kinematics
    quat_wxyz, ang_vel = _read_imu_quat_wxyz_and_avel()
    rpy = _quat_to_euler_wxyz(quat_wxyz)
    q, dq = _get_qdq_in_amo_order()

    # Gamepad command with locomotion modes applied
    cmd = _apply_locomotion_modes(_cmd)

    target_yaw = cmd[1]
    dyaw = rpy[2] - target_yaw
    dyaw = (dyaw + math.pi) % (2*math.pi) - math.pi
    _in_place = (abs(cmd[0]) < 0.1)
    if _in_place: dyaw = 0.0

    # Gait features
    gait_obs = np.sin(_gait_cycle * 2 * math.pi).astype(np.float32)

    # Adapter input: [height_cmd, torso yaw/pitch/roll, 8 arm DOFs]
    adapter_in = np.zeros(12, dtype=np.float32)
    adapter_in[0] = 0.75 + cmd[3]
    adapter_in[1] = cmd[4]; adapter_in[2] = cmd[5]; adapter_in[3] = cmd[6]
    adapter_in[4:] = q[15:]
    ain = torch.tensor(adapter_in, device=_device, dtype=torch.float32).unsqueeze(0)
    ain_n = (ain - _in_mean) / (_in_std + 1e-8)
    aout = _adapter(ain_n.view(1,-1))
    aout = aout * _out_std + _out_mean
    aout_np = aout.detach().cpu().numpy().reshape(-1).astype(np.float32)

    # Proprio vector as used by AMO
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

    # Demo/priv/history blocks
    obs_demo = np.zeros(8 + 3 + 3 + 3, dtype=np.float32)
    obs_demo[:8] = q[15:]          # arm DOFs
    obs_demo[8]  = cmd[0]          # vx
    obs_demo[9]  = cmd[2]          # vy (strafe)
    obs_demo[11] = cmd[4]; obs_demo[12] = cmd[5]; obs_demo[13] = cmd[6]
    obs_demo[14:17] = 0.75 + cmd[3]

    obs_priv = np.zeros(_n_priv, dtype=np.float32)

    _prop_hist.pop(0); _prop_hist.append(obs_prop.copy())
    _extra_hist.pop(0); _extra_hist.append(obs_prop.copy())
    obs_hist = np.array(_prop_hist, dtype=np.float32).flatten()

    # Update gait cycle
    _gait_cycle = np.remainder(_gait_cycle + _DT_CTRL * _gait_freq, 1.0).astype(np.float32)
    if _in_place and ((abs(_gait_cycle[0]-0.25)<0.05) or (abs(_gait_cycle[1]-0.25)<0.05)):
        _gait_cycle[:] = 0.25
    if (not _in_place) and (abs(_gait_cycle[0]-0.25)<0.05) and (abs(_gait_cycle[1]-0.25)<0.05):
        _gait_cycle[:] = [0.25, 0.75]

    return np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist), dtype=np.float32)

def _policy_step():
    """Run policy with AMO observation + extra history; ramp actions; write last_action buffer."""
    global _last_action, _last_scaled
    obs = _compute_observation()
    extra_hist = np.array(_extra_hist, dtype=np.float32).flatten()

    with torch.no_grad():
        obs_t = torch.from_numpy(obs).to(_device).float().unsqueeze(0)
        eh_t  = torch.from_numpy(extra_hist).to(_device).float().unsqueeze(0)
        raw = _policy(obs_t, eh_t).detach().cpu().numpy().reshape(-1).astype(np.float32)

    raw = np.clip(raw, -40., 40.).astype(np.float32)
    scaled = raw * ACTION_SCALE

    # Action ramp (soft start / mode transition smoothing)
    if ACTION_RAMP_TAU > 1e-5:
        alpha = 1.0 - math.exp(-_DT_CTRL / ACTION_RAMP_TAU)
        _last_scaled = (1.0 - alpha)*_last_scaled + alpha*scaled
        scaled = _last_scaled

    # last_action (23): [raw_action(15), (q - default)[15:]/ACTION_SCALE]
    q, _ = _get_qdq_in_amo_order()
    _last_action = np.concatenate([raw.copy(), (q - G1_DEFAULT_DOF_POS)[15:] / ACTION_SCALE]).astype(np.float32)

    # pd_target(23): first 15 = legs+waist, arms = default
    pd_target = G1_DEFAULT_DOF_POS.copy()
    pd_target[:15] += scaled
    return pd_target

def _apply_targets(q_target, dt):
    """Apply desired joint positions (Isaac Drive targets) or PD torque fallback per joint."""
    global _q_cmd_prev, _USE_TORQUE_FALLBACK
    q_cmd = _filter_targets(q_target, _q_cmd_prev, dt)
    _q_cmd_prev = q_cmd.copy()

    # Drive position targets if available
    if not _USE_TORQUE_FALLBACK:
        try:
            for amo_i, ix in enumerate(_IDX_ISAAC_FROM_AMO):
                if ix is None or ix < 0: continue
                h = _DOF_HANDLES[ix]
                dc.set_dof_position_target(h, float(q_cmd[amo_i]))
            return
        except Exception:
            _USE_TORQUE_FALLBACK = True
            carb.log_warn("[amo-isaac] Switching to PD torque fallback (Drives inoperative).")

    # PD torque fallback
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

def _on_timeline(ev):
    """Rebuild on PLAY start; pause stops updates automatically."""
    playing = _is_playing()
    global _LAST_PLAY, _BUILT, _ACCUM, _LAST_T
    if playing and not _LAST_PLAY:
        carb.log_warn("[amo-isaac] PLAY → controller starting (gamepad/modes active).")
        _BUILT=False
        _ACCUM = 0.0
        _LAST_T = float(timeline.get_current_time())
    _LAST_PLAY = playing

def _on_update(ev):
    """Kit app update: step control at ~CONTROL_HZ while timeline is playing."""
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
    """Unsubscribe and clear the active-instance guard on exit."""
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

# ---------------- Gamepad (pygame) ----------------
def _start_gamepad_thread():
    """
    Start a daemon thread that reads pygame joystick and fills `_cmd`.
    Typical Nintendo Switch Pro mapping in pygame:
      - Left stick: axis 0 (X), 1 (Y)
      - Right stick: axis 2 (X), 3 (Y)
      - Buttons: A=0, B=1, X=2, Y=3, L=4, R=5, ZL=6, ZR=7, Minus=8, Plus=9
    If your mapping differs, adjust the constants below.
    """
    try:
        import pygame
    except Exception:
        carb.log_warn("[amo-isaac] pygame not available: running without gamepad.")
        return

    def _thread():
        try:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                carb.log_warn("[amo-isaac] gamepad: no joystick detected. Running without controller.")
                return
            js = pygame.joystick.Joystick(0); js.init()
            carb.log_info(f"[amo-isaac] Gamepad detected: {js.get_name()}")

            # Axis mapping (adjust if your device differs)
            AXIS_LX = 0   # yaw (turn left/right)
            AXIS_LY = 1   # walk (forward/backward)
            AXIS_RX = 2   # strafe (left/right)
            AXIS_RY = 3   # height (up/down)

            # Buttons (typical Switch Pro in pygame)
            BTN_A = 0
            BTN_B = 1
            BTN_X = 2
            BTN_Y = 3
            BTN_L = 4
            BTN_R = 5
            BTN_ZL = 6
            BTN_ZR = 7
            BTN_MINUS = 8
            BTN_PLUS  = 9

            last_btn = {}
            deadzone = 0.15

            while True:
                pygame.event.pump()
                # Read sticks (invert to match intuitive directions)
                ly = -js.get_axis(AXIS_LY) if js.get_numaxes()>AXIS_LY else 0.0
                lx = -js.get_axis(AXIS_LX) if js.get_numaxes()>AXIS_LX else 0.0
                rx = -js.get_axis(AXIS_RX) if js.get_numaxes()>AXIS_RX else 0.0
                ry = -js.get_axis(AXIS_RY) if js.get_numaxes()>AXIS_RY else 0.0

                # Base command from sticks (speed scaling also occurs later in mode logic)
                _cmd[0] = ly * (_loco.speed_sprint if _loco.sprint else _loco.speed_walk) if abs(ly)>deadzone else 0.0
                _cmd[1] = lx * (_loco.speed_sprint if _loco.sprint else _loco.speed_walk) if abs(lx)>deadzone else 0.0
                _cmd[2] = rx * 0.5 if abs(rx)>deadzone else 0.0
                if not _loco.jump_active:
                    _cmd[3] = ry * 0.5 if abs(ry)>deadzone else 0.0

                # Buttons
                cur = {b: js.get_button(b) for b in range(js.get_numbuttons())}

                # Toggle sprint (B)
                if cur.get(BTN_B,0) and not last_btn.get(BTN_B,0):
                    _loco.sprint = not _loco.sprint
                    carb.log_warn(f"[amo-isaac] Sprint {'ON' if _loco.sprint else 'OFF'}")

                # Toggle crouch (Y)
                if cur.get(BTN_Y,0) and not last_btn.get(BTN_Y,0):
                    _loco.crouch = not _loco.crouch
                    carb.log_warn(f"[amo-isaac] Crouch {'ON' if _loco.crouch else 'OFF'}")

                # Jump sequence (A)
                if cur.get(BTN_A,0) and not last_btn.get(BTN_A,0) and not _loco.jump_active:
                    _loco.jump_active = True; _loco.jump_step = 0; _loco.jump_t = 0.0
                    carb.log_warn("[amo-isaac] Jump!")

                # Torso pitch with ZL/ZR (discrete steps)
                if cur.get(BTN_ZL,0) and not last_btn.get(BTN_ZL,0):
                    _cmd[5] += 0.05
                if cur.get(BTN_ZR,0) and not last_btn.get(BTN_ZR,0):
                    _cmd[5] -= 0.05

                # Torso roll with L/R (discrete steps)
                if cur.get(BTN_L,0) and not last_btn.get(BTN_L,0):
                    _cmd[6] -= 0.05
                if cur.get(BTN_R,0) and not last_btn.get(BTN_R,0):
                    _cmd[6] += 0.05

                # Minus toggles arm movement flag
                if cur.get(BTN_MINUS,0) and not last_btn.get(BTN_MINUS,0):
                    _cmd[7] = 0.0 if _cmd[7] else 1.0
                    carb.log_warn(f"[amo-isaac] Arm movement {'ENABLED' if _cmd[7]>0.5 else 'DISABLED'}")

                last_btn = cur
                time.sleep(0.01)  # ~100 Hz
        except Exception as e:
            carb.log_error(f"[amo-isaac] Gamepad thread error: {e}")

    th = threading.Thread(target=_thread, daemon=True)
    th.start()

# ---------------- Init/quit ----------------
def _init():
    """Subscribe to Kit update/timeline streams and start gamepad reader."""
    global _SUB, _TL_SUB
    try:
        import carb.settings
        s = carb.settings.get_settings()
        if s.get(_SETTINGS_KEY):
            carb.log_warn("[amo-isaac] Another instance is already active.")
            return
        s.set(_SETTINGS_KEY, True)
    except: pass

    app = get_app()
    _SUB = app.get_update_event_stream().create_subscription_to_pop(_on_update, name="amo_isaac_update")
    _TL_SUB = timeline.get_timeline_event_stream().create_subscription_to_pop(_on_timeline, name="amo_isaac_timeline")
    carb.log_warn("[amo-isaac] Press PLAY to start control.")
    _start_gamepad_thread()

atexit.register(_cleanup)
_init()

