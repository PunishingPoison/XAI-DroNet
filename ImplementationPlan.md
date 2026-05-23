# XAI-DroNet: Explainable Autonomous Drone Navigation Across Unseen Simulated Environments

## Project Type
Research-oriented AI + Computer Vision + Robotics Simulation Project

## Core Research Paper
DroNet: Learning to Fly by Driving  
https://rpg.ifi.uzh.ch/dronet.html

Paper:
https://arxiv.org/abs/1804.02308

Official Repository:
https://github.com/uzh-rpg/rpg_public_dronet

---

# 1. Project Objective

Build an autonomous drone navigation system entirely in simulation using a DroNet-inspired CNN model.

The system should:
- Navigate autonomously using monocular vision
- Predict steering angle
- Predict collision probability
- Avoid obstacles
- Generalize to unseen environments
- Explain navigation decisions using Grad-CAM visualization

The project must remain faithful to the DroNet paper.

---

# 2. Final Deliverables

## Functional Deliverables
- Autonomous drone simulation
- Real-time navigation
- Collision avoidance
- Explainable AI visualization
- Environment generalization testing
- Evaluation metrics dashboard

## Research Deliverables
- Generalization analysis
- Explainability analysis
- Performance benchmarking
- Failure case analysis

---

# 3. High-Level Architecture

```text
AirSim Environment
        ↓
Monocular Camera Feed
        ↓
DroNet CNN
 ┌─────────────────────┐
 │ Steering Prediction │
 │ Collision Probability │
 └─────────────────────┘
        ↓
Drone Control Logic
        ↓
Autonomous Navigation
        ↓
Grad-CAM Explainability
        ↓
Evaluation + Dashboard
```

---

# 4. Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Simulator | AirSim |
| Graphics Engine | Unreal Engine 4.27 |
| Vision | OpenCV |
| Explainability | Grad-CAM |
| Visualization | Matplotlib |
| Dashboard | Streamlit |
| Version Control | Git |

---

# 5. Hardware Requirements

## Current Hardware
- 16 GB RAM
- RTX 3050 Laptop GPU

This is sufficient for:
- AirSim
- PyTorch
- Grad-CAM
- Real-time inference
- Fine-tuning lightweight CNNs

---

# 6. Software Installation

## 6.1 Install Python

Recommended Version:
Python 3.10

Download:
https://www.python.org/downloads/

---

## 6.2 Install Git

Download:
https://git-scm.com/downloads

---

## 6.3 Install VS Code

Download:
https://code.visualstudio.com/

Recommended Extensions:
- Python
- Pylance
- Jupyter
- GitLens

---

## 6.4 Install Unreal Engine

Recommended Version:
UE 4.27

Download:
https://www.unrealengine.com/en-US/download

---

## 6.5 Install AirSim

Documentation:
https://microsoft.github.io/AirSim/

GitHub:
https://github.com/microsoft/AirSim

Windows Build Guide:
https://microsoft.github.io/AirSim/build_windows/

---

# 7. Initial Project Setup

## Create Project Folder

```bash
mkdir XAI-DroNet
cd XAI-DroNet
```

---

## Create Virtual Environment

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install numpy
pip install matplotlib
pip install pandas
pip install airsim
pip install streamlit
pip install grad-cam
pip install pillow
pip install scikit-learn
```

---

# 8. Recommended Folder Structure

```text
XAI-DroNet/
│
├── datasets/
├── models/
├── simulator/
├── explainability/
├── evaluation/
├── dashboard/
├── notebooks/
├── outputs/
├── configs/
├── utils/
├── main.py
└── requirements.txt
```

---

# 9. Project Phases

## PHASE 1 — Understand and Reproduce DroNet

### Goal
Run the original DroNet successfully.

### Tasks
- Clone DroNet repository
- Install dependencies
- Load pretrained model
- Run inference on sample images
- Understand:
  - architecture
  - outputs
  - preprocessing

---

## PHASE 2 — AirSim Integration

### Goal
Connect DroNet to drone simulation.

### Tasks
- Launch AirSim environment
- Capture monocular camera frames
- Feed frames into DroNet
- Retrieve:
  - steering prediction
  - collision probability

### Expected Output
Drone receives AI-based control commands.

---

## PHASE 3 — Navigation Controller

### Goal
Implement drone control logic.

### Based On Paper
Use:
- steering angle
- collision probability
- smooth velocity filtering

### Control Logic

#### Steering
Convert:
```text
[-1, 1] → yaw angle
```

#### Speed
Use:
```text
speed = (1 - collision_probability)
```

#### Smoothing
Apply low-pass filtering.

---

## PHASE 4 — Environment Generalization

### Goal
Test the model on unseen environments.

### Training Environment
- Urban roads

### Testing Environments
- Corridors
- Parking lots
- Warehouses
- Tunnels
- Indoor halls

### Metrics
- Collision rate
- Navigation success rate
- Travel distance
- Steering smoothness

---

## PHASE 5 — Explainable AI (Grad-CAM)

### Goal
Visualize model attention.

### Why
Show:
- what the drone focuses on
- why it turns
- why it slows down

### Library
https://github.com/jacobgil/pytorch-grad-cam

### Tasks
- Hook into last convolution layer
- Generate Grad-CAM heatmaps
- Overlay heatmaps onto camera frames

### Expected Output
Real-time explainability visualization.

---

## PHASE 6 — Evaluation Framework

### Goal
Scientifically evaluate navigation.

### Metrics

| Metric | Purpose |
|---|---|
| Collision Rate | Safety |
| Avg Travel Distance | Navigation Quality |
| Steering Variance | Smoothness |
| FPS | Real-Time Performance |
| Attention Consistency | Explainability Stability |

---

## PHASE 7 — Dashboard

### Goal
Build visual monitoring interface.

### Tool
Streamlit

### Features
- Live camera feed
- Collision probability graph
- Steering angle graph
- Grad-CAM overlay
- Environment selector
- Metrics display

---

# 10. Simulation Environments

## Recommended Starter Environment
Blocks Environment

## Advanced Environment
City Environment

## Indoor Testing
- Warehouse
- Parking garage
- Corridor layouts

---

# 11. Input Specifications

## Camera Input
- Monocular RGB
- Convert to grayscale
- Resize to:
```text
200 × 200
```

This follows the original paper.

---

# 12. Core Model Design

## Shared CNN Backbone
ResNet-inspired architecture.

## Outputs
1. Steering regression
2. Collision probability classification

---

# 13. Important Research Constraints

## MUST FOLLOW
- Monocular vision only
- Lightweight architecture
- End-to-end learning
- Steering + collision prediction

## MUST NOT
- Add LiDAR
- Use SLAM systems
- Use external path planning
- Convert project into generic RL navigation

This project must stay faithful to the DroNet paper.

---

# 14. Recommended Development Order

1. Reproduce DroNet
2. Connect AirSim
3. Make drone fly
4. Add Grad-CAM
5. Add evaluation metrics
6. Add dashboard
7. Optimize later

---

# 15. Key Python Modules to Build

## model.py
DroNet architecture

## inference.py
Prediction pipeline

## airsim_controller.py
Drone control

## gradcam.py
Explainability generation

## evaluator.py
Metrics and testing

## dashboard.py
Visualization interface

---

# 16. Timeline

| Week | Goal |
|---|---|
| 1 | Setup tools |
| 2 | Reproduce DroNet |
| 3 | Setup AirSim |
| 4 | Drone control |
| 5 | Autonomous navigation |
| 6 | Generalization testing |
| 7 | Grad-CAM integration |
| 8 | Evaluation framework |
| 9 | Dashboard |
| 10 | Documentation + presentation |

---

# 17. Best Learning Resources

## DroNet
https://rpg.ifi.uzh.ch/dronet.html

## AirSim
https://microsoft.github.io/AirSim/

## PyTorch Tutorials
https://docs.pytorch.org/tutorials/

## Grad-CAM
https://github.com/jacobgil/pytorch-grad-cam

## OpenCV
https://docs.opencv.org/

---

# 18. Final Research Goal

The final system should demonstrate:
- Autonomous drone navigation
- End-to-end CNN control
- Collision avoidance
- Environment generalization
- Explainable AI reasoning

while remaining strongly aligned with the original DroNet paper.
