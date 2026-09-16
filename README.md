# G1-AMOjoystick
>[!WARNING]
>This is Work in Progress Project from the <a href="https://www.uvs-robotarium-lab.ca"> Robotarium Lab</a> at the University of Calgary.

![banner-UnitreeG1](https://github.com/user-attachments/assets/9e225866-34e4-4fd8-9327-32fe1c5febcc)

![GitHub last commit](https://img.shields.io/github/last-commit/AlanS1N/G1-AMOjoystick?style=for-the-badge)
![GitHub](https://img.shields.io/github/license/AlanS1N/G1-AMOjoystick?style=for-the-badge)

Table of Contents
 
- [Project Overview](#project-overview)
- [Screenshots and Demos](#screenshots-and-demos)
- [System Architecture](#system-architecture)
- [Environment & Tools](#environment--tools)
- [Future Goals](#future-goals)
- [Acknowledgments](#acknowledgments)
---

# Project Overview

The objective of this project is to have a controller option, which should be intuitive and user friendly, to control and move a humanoid robot. Initially this project was aimed to be used with a 37-DoF robot with a joystick package as a controller in ROS. It was later said to use a G1 humanoid robot with the <a href="https://amo-humanoid.github.io"> AMO</a>, which is a framework developed by the UC San Diego (UCSD), released on May 10th 2025. It was tested on a 29-DoF Unitree G1 humanoid robot. 

AMO provides body movement optimization on humanoid robots by using RL trained neural network policies to define how it should act based on environment observations. It is built upon Python 3.9+ and can communicate with real hardware using controllers such as ROS. Also note that these policies were trained using Proximal Policy Optimization (PPO) with reward shaping.

The pipeline is first validated in MuJoCo, then migrated to Isaac Sim/Isaac Lab for higher-fidelity physics.

**AMO resources and references are listed below:**

🌐 Click here to <a href="https://github.com/OpenTeleVision/AMO"> VIEW the official AMO repository</a> in Github.

📜 Click here to<a href="https://amo-humanoid.github.io/resources/amo.pdf"> VIEW the official AMO paper published</a> by the AMO team.



**Additional resources and references (Isaac Lab / Unitree):**

🦿 [Isaac Lab Joint Drive Documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.schemas.html?utm_source=#joint-drive)

🧠 [Isaac Sim **(4.5.0)** Policy Deployment Documentation & Example](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/isaac_lab_tutorials/tutorial_policy_deployment.html)

📋 [Unitree Isaac Lab Official Tasks Test Repository](https://github.com/unitreerobotics/unitree_sim_isaaclab?tab=readme-ov-file)

💻 [Unitree Isaac Lab Official RL Environment](https://github.com/unitreerobotics/unitree_rl_lab)

**Key Features:**
  1. Gamepad teleoperation of the full-body humanoid via a Nintendo Switch Pro Controller (Bluetooth, handled with Pygame) — forward/lateral velocity, yaw, torso pitch/roll/yaw and height.
  2. Whole-body stability powered by AMO: a PPO-trained, TorchScript neural network policy that keeps the 29-DoF Unitree G1 balanced while following the operator's commands.
  3. Real-time control validation in the MuJoCo viewer before moving to a higher-fidelity simulator (Isaac Sim).
  4. A migration pipeline that ports the AMO policy from MuJoCo into NVIDIA Isaac Sim / Isaac Lab, including USD scene setup, articulation drives, IMU integration and AMO-Isaac joint-order mapping.
  5. Gamepad-triggered locomotion modes: toggled Sprint (B, 2× speed multiplier), toggled Crouch (Y, lowers the center of mass and clamps speed for stability), and a scripted 3-phase Jump (A: crouch → upward impulse → recovery [NOT STABLE]) — plus torso pitch/roll trim (ZL/ZR, L/R) and an arm-movement toggle (Minus).
 
# Screenshots and Demos
<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/4bd4447f-bafe-4cc1-bc16-2779ed6da940" width="500"/>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/715569bb-0b04-476d-a99e-8ed4fe5f1029" width="500"/>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/4c1e5dcb-9fd6-44d7-97a2-c50e4252d4d6" width="500"/>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/1036b618-597f-4e62-acb6-15054ec9e86b" width="500"/>
    </td>
  </tr>
</table>

Demo videos referenced in the manual:
- 🎮 [Gamepad control demo (MuJoCo)](https://youtu.be/7HSVvK0hRzQ)
- ⚛️ [Gamepad control demo (IsaacSim)](https://youtu.be/8PjEr26Z97I)
 
# System Architecture
 
Controller communication and Simulator:
 
```
GAMEPAD (Switch Pro Controller, Bluetooth)
        │  Pygame (axes, buttons, D-pad)
        ▼
 LOCOMOTION INPUT ──walk / sprint / crouch / jump──
        │                               
        ▼                                   
   ENVIRONMENT  ◀────────────────────  NAVIGATION
 (MuJoCo Viewer  /  NVIDIA Isaac Sim & Isaac Lab)
        │
        ▼
 AMO POLICY (PPO-trained MLP, TorchScript)
        │
        ▼
 USD POSITION DRIVES  ──▶  Unitree G1 (29-DoF)
   (auto PD-torque fallback if drives are unavailable)
```
 
The gamepad commands are converted into a command vector (velocity, yaw, torso pose, height) that is modified by the active locomotion mode (sprint/crouch/jump) and fed into the AMO observation vector alongside the robot's proprioception (orientation, joint positions/velocities). 

The AMO policy outputs target joint positions, which are smoothed and applied to the G1 model through Isaac Sim's USD articulation drives — falling back automatically to a per-joint PD torque controller (using AMO's stiffness/damping/torque-limit gains) if the runtime doesn't support position targets. Full block-by-block detail is in the [User Manual](https://docs.google.com/document/d/1VXFfM9Yygdm7M20YvoX3aLMf8np9Tc78E8D04FuDVUY/edit?usp=sharing).
 
# Environment & Tools
 
The software/hardware stack used is listed below:
 
|            Component             |                      Role                       |
|-----------------------------------|--------------------------------------------------|
| Unitree G1 (29-DoF)                | Target humanoid robot platform                   |
| AMO Framework (UCSD)               | RL whole-body control policies (PPO, TorchScript)|
| MuJoCo / MuJoCo Viewer             | Real-time physics simulator for initial testing  |
| NVIDIA Isaac Sim 4.5.0             | High-fidelity physics simulator for deployment   |
| NVIDIA Isaac Lab                   | Robot-learning framework built on Isaac Sim      |
| Python 3.10 (Miniconda)            | Runtime environment                              |
| PyTorch                            | Neural network inference (TorchScript policies)  |
| Pygame                             | Gamepad input handling                           |
| Nintendo Switch Pro Controller     | Teleoperation input device (Bluetooth)           |
| Ubuntu 20.04 / 22.04               | Operating system                                 |
 
# Future Goals
 
This project was a project worked on July - September 2025 during the Mitacs GRI 2025 program at the University of Calgary, in the Robotarium Lab, under the supervision of Dr. Alex Ramirez-Serrano.
 
The current future goals involve:
- ⚖️ Improving balance and stability of the AMO policy once ported to Isaac Sim's more realistic physics engine — the jump routine in particular is still unstable and falls.
- 🔁 Testing transitions between different neural network policies (e.g. switching the AMO locomotion policy for a dedicated crawling policy).
- ✋ Extending teleoperation to hand control, using the 5-finger hand or the Dex 3-1 hands, this last ones being the option that the G1 of the Robotarium Lab uses.

# Acknowledgments
 
This project's Isaac Sim control bridge is built directly on top of the AMO framework and reuses its trained policies, PD gains and joint conventions:
 
**AMO** — © 2025 Jialong Li, Xuxin Cheng, Tianshu Huang, Xiaolong Wang (UC San Diego), licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0). See the [official AMO repository](https://github.com/OpenTeleVision/AMO).
