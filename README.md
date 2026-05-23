# XAI-DroNet: Explainable Autonomous Drone Navigation

Welcome to the **XAI-DroNet** project! This repository contains a complete, research-oriented implementation of an Explainable Autonomous Drone Navigation system, based on the principles of the **DroNet** paper (*Learning to Fly by Driving*), enhanced with Explainable AI (XAI) capabilities through **Grad-CAM** and integrated with **Unreal Engine 5** via **Colosseum/AirSim**.

This project provides a robust pipeline for simulating, evaluating, and visualizing deep neural network decisions for autonomous drone flight in unseen, simulated environments.

---

## 📖 Table of Contents
- [Project Overview](#project-overview)
- [Key Concepts](#key-concepts)
- [Tech Stack](#tech-stack)
- [Directory Structure](#directory-structure)
- [Installation & Setup](#installation--setup)
- [Usage & Execution](#usage--execution)
- [Dashboard & Explainability](#dashboard--explainability)
- [License & Acknowledgements](#license--acknowledgements)

---

## 🚀 Project Overview

The core objective of XAI-DroNet is to enable a quadcopter drone to autonomously navigate complex environments (like city streets, hallways, or obstacle courses) using only **monocular vision** (a single front-facing camera). 

Unlike standard black-box AI models, this project integrates **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate real-time heatmaps, allowing researchers and developers to "see what the AI sees" and understand *why* the drone makes specific steering and braking decisions.

### Features
1. **End-to-End Navigation**: Uses a lightweight ResNet-8 architecture to predict steering angles and collision probabilities directly from 200x200 grayscale images.
2. **Explainable AI (XAI)**: Generates live heatmaps showing which parts of the image influence the drone's collision avoidance and steering.
3. **Simulation Integration**: Fully integrated with Colosseum (an actively maintained fork of Microsoft AirSim) running on Unreal Engine 5 for high-fidelity physics and rendering.
4. **Interactive Dashboard**: A real-time Streamlit dashboard to monitor flight metrics, camera feeds, and Grad-CAM overlays.

---

## 🧠 Key Concepts

- **DroNet Architecture**: A residual convolutional neural network (CNN) that processes images to simultaneously predict a continuous steering angle (regression) and a binary collision probability (classification).
- **Grad-CAM**: An explainability technique that uses the gradients of the target concept (e.g., collision probability) flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image.
- **Continuous Control**: A low-pass filtered controller that maps the neural network's predictions to smooth physical velocity and yaw commands for the drone. If a collision is imminent (high collision probability), the forward velocity drops, and an emergency panic-turn reflex is triggered.

---

## 💻 Tech Stack

- **Machine Learning**: PyTorch, Torchvision
- **Simulation Environment**: Unreal Engine 5, Colosseum (AirSim)
- **Computer Vision**: OpenCV, NumPy
- **Dashboard & Visualization**: Streamlit, Matplotlib
- **Language**: Python 3.10+

---

## 🛠️ Installation & Setup

### Prerequisites
1. **Python 3.10**: Highly recommended for best PyTorch and AirSim compatibility.
2. **Colosseum / Unreal Engine**: You must have Unreal Engine (5.4+) and a compiled Colosseum environment running.

### 1. Clone the Repository
```powershell
git clone https://github.com/PunishingPoison/XAI-DroNet.git
cd XAI-DroNet
```

### 2. Create a Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
pip install -r requirements-airsim.txt
```
*(Note: If you are using a custom compiled version of Colosseum, install the PythonClient directly from your Colosseum source folder instead of the default PyPI `airsim` package).*

### 4. Download Official Weights
To use the pre-trained Keras weights converted to PyTorch:
```powershell
python .\scripts\download_official_dronet_model.py
python .\scripts\convert_keras_weights.py --keras-weights .\checkpoints\official\dronet\best_weights.h5 --output .\checkpoints\dronet_official.pt
```

---

## 🎮 Usage & Execution

### Running the Simulator
1. Open your Colosseum/Unreal Engine environment.
2. Press **Play** in the editor. Ensure your drone spawns in an open area.

### Autonomous Flight
Open a PowerShell window, activate your environment, and execute the evaluation script:
```powershell
python scripts\run_evaluation.py --environment-name "Blocks" --execute-control --takeoff --num-steps 1000
```
- `--execute-control`: Sends actual velocity commands to the simulator.
- `--takeoff`: Commands the drone to take off before engaging the AI.
- `--num-steps`: Number of inference loops to execute.

---

## 📊 Dashboard & Explainability

To visualize the AI's decision-making process in real-time, launch the Streamlit dashboard in a separate PowerShell window:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run src\xai_dronet\dashboard\app.py
```
This will open a browser window displaying:
- The live camera feed from the drone.
- The Grad-CAM heatmap overlay (highlighting obstacles).
- Live telemetry including steering angle and collision probability.

---

## 📁 Directory Structure
- `checkpoints/` - Model weights and conversions.
- `docs/` - Development notes and phase documentation.
- `outputs/` - Generated logs, evaluation metrics, and image frames.
- `scripts/` - Execution scripts for inference, evaluation, and utilities.
- `src/xai_dronet/` - Core Python module containing control, inference, simulator bridges, and XAI code.

---

## 📜 License & Acknowledgements
Based on the original [DroNet paper](http://rpg.ifi.uzh.ch/dronet.html) by Antonio Loquercio et al. (University of Zurich). 
Built using [Colosseum](https://github.com/CodexLabsLLC/Colosseum), the active fork of Microsoft AirSim.
