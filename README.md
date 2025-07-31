# Spot Arm Grasp & WASD Control

### Introduction

This project provides an interactive interface for Boston Dynamics Spot’s arm, enabling both autonomous object grasping and manual arm manipulation via keyboard. It integrates real-time vision-based object selection, advanced grasp constraints, and live feedback, making it ideal for research and prototyping in robotics.

---

### Demo

- Click on an object in the camera feed to command Spot to grasp it.
- After grasping, use WASD and other keys to manually control the arm and gripper in real time.

---

### :rocket: Quick Start

#### Setting Up the Python Environment

```bash
python3 -m venv my_spot_env
source my_spot_env/bin/activate
pip install bosdyn-client bosdyn-mission bosdyn-choreography-client bosdyn-orbit opencv-python numpy
```

#### Running the Program

```bash
python3 arm_grasp_wasd.py --hostname <ROBOT_IP> [options]
```

**Common options:**
- `-i`, `--image-source` : Camera image source (default: `frontleft_fisheye_image`)
- `-t`, `--force-top-down-grasp` : Force top-down grasp
- `-f`, `--force-horizontal-grasp` : Force horizontal grasp
- `-r`, `--force-45-angle-grasp` : Force 45-degree angled grasp
- `-s`, `--force-squeeze-grasp` : Force squeeze grasp

**Example:**
```bash
python3 arm_grasp_wasd.py --hostname 192.168.80.3 -t
```

---

### System Implementation

#### Interactive Grasping

- Select objects to grasp by clicking on live camera images.
- The robot computes and executes a grasp using Spot’s manipulation API.

#### Manual Arm Control

- After grasping, control the arm in real time using intuitive WASD keyboard commands:
  - `w/s`: move out/in
  - `a/d`: rotate ccw/cw
  - `r/f`: up/down
  - `u/o`: +rx/-rx
  - `i/k`: +ry/-ry
  - `j/l`: +rz/-rz
  - `m`: close gripper
  - `n`: open gripper
  - `z`: stow arm
  - `x`: unstow arm
  - `g`: grasp again
  - `q`: quit

#### Grasp Constraints

- Supports top-down, horizontal, angled, and squeeze grasps (set via command-line flags).
- Only one constraint can be active at a time.

#### Visual Feedback

- Live OpenCV windows display camera images for object selection and provide immediate feedback on grasping actions.

---

### Technologies Used

- **Languages:** Python 3
- **Libraries:** Boston Dynamics Spot SDK, OpenCV, NumPy

---

### Safety & Best Practices

- Ensure Spot is in a safe, open area before running.
- Monitor the robot during operation, especially in manual mode.
- Use E-Stop if any unsafe behavior is observed.

---

### License

This project is based on the Boston Dynamics SDK and is subject to the Boston Dynamics Software Development Kit License.

---
