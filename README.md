# XAI-DroNet: Explainable Autonomous Drone Navigation

<img width="1918" height="1021" alt="SS1" src="https://github.com/user-attachments/assets/1fc91504-85c7-43d1-9320-fdc41419ffdb" />


This repository contains a research-oriented implementation of an Explainable Autonomous Drone Navigation system. It is based on the principles of the DroNet paper ("Learning to Fly by Driving"), enhanced with Explainable AI (XAI) capabilities through Grad-CAM, and integrated with Unreal Engine 5 via the Colosseum (AirSim) simulator.

This project provides a robust pipeline for simulating, evaluating, and visualizing deep neural network decisions for autonomous drone flight in unseen, simulated environments.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Concepts](#key-concepts)
- [Technology Stack](#technology-stack)
- [Directory Structure](#directory-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage and Execution](#usage-and-execution)
- [Dashboard and Explainability](#dashboard-and-explainability)
- [License and Acknowledgements](#license-and-acknowledgements)

---

## Project Overview

The core objective of XAI-DroNet is to enable a quadcopter drone to autonomously navigate complex environments (such as city streets, hallways, or obstacle courses) using only monocular vision (a single front-facing camera). 

Unlike standard black-box AI models, this project integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to generate real-time heatmaps. This allows researchers and developers to visualize the regions of the image that influence the drone's collision avoidance and steering decisions.

### Features
1. **End-to-End Navigation**: Utilizes a lightweight ResNet-8 architecture to predict steering angles and collision probabilities directly from 200x200 grayscale images.
2. **Explainable AI (XAI)**: Generates live heatmaps showing which parts of the image influence the drone's behavior.
3. **Simulation Integration**: Fully integrated with Colosseum (an actively maintained fork of Microsoft AirSim) running on Unreal Engine 5 for high-fidelity physics and rendering.
4. **Interactive Dashboard**: A real-time Streamlit dashboard to monitor flight metrics, camera feeds, and Grad-CAM overlays.

---

## Key Concepts

- **DroNet Architecture**: A residual convolutional neural network (CNN) that processes images to simultaneously predict a continuous steering angle (regression) and a binary collision probability (classification).

<img width="1918" height="1018" alt="SS4" src="https://github.com/user-attachments/assets/84f900f0-1a24-4c54-8a58-c2ada42f6cb1" />

- **Grad-CAM**: An explainability technique that uses the gradients of the target concept (e.g., collision probability) flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image.

<img width="1702" height="830" alt="SS9" src="https://github.com/user-attachments/assets/98d1ad58-e5c8-41ed-8488-107cc373dd45" />

- **Continuous Control**: A low-pass filtered controller that maps the neural network's predictions to smooth physical velocity and yaw commands for the drone. If a collision is imminent (high collision probability), the forward velocity drops, and an emergency panic-turn reflex is triggered to avoid the obstacle.

<img width="1482" height="761" alt="SS3" src="https://github.com/user-attachments/assets/4cfc441a-c767-4cc0-9d62-bd991ba602f6" />


---

## Technology Stack

- **Machine Learning**: PyTorch, Torchvision
- **Simulation Environment**: Unreal Engine 5, Colosseum (AirSim)
- **Computer Vision**: OpenCV, NumPy
- **Dashboard and Visualization**: Streamlit, Matplotlib
- **Language**: Python 3.10+

---

## Installation and Setup

The following instructions detail the setup process for the Python environment and the simulation dependencies.

### Prerequisites
1. **Python 3.10 or higher**: Required for compatibility with PyTorch and the AirSim client.
2. **Git**: Required for cloning the repository.
3. **Colosseum and Unreal Engine 5**: You must have Unreal Engine (version 5.4 or newer) installed, along with a compiled Colosseum environment.

### 1. Clone the Repository
Open a terminal or command prompt and run:
```bash
git clone https://github.com/PunishingPoison/XAI-DroNet.git
cd XAI-DroNet
```

### 2. Python Environment Setup

It is highly recommended to use an isolated Python environment. You can choose either the standard `venv` module (Method A) or Anaconda/Miniconda (Method B).

#### Method A: Using Python venv (Recommended)
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

#### Method B: Using Conda (Alternative)
If you prefer Conda for environment management:
```bash
conda create -n xai-dronet python=3.10
conda activate xai-dronet
```

### 3. Install Dependencies

Install the core machine learning and dashboard requirements:
```bash
pip install -r requirements.txt
```

Next, install the AirSim simulation requirements:
```bash
pip install -r requirements-airsim.txt
```

**Troubleshooting AirSim Installation:**
If you are using a custom compiled version of Colosseum and encounter a `bad cast` exception during simulation, the default PyPI `airsim` package may be outdated. As an alternative, install the Python client directly from your Colosseum source directory:
```bash
pip uninstall -y airsim
pip install --no-build-isolation -e <Path-To-Your-Colosseum-Directory>\PythonClient
```

### 4. Download Official Weights
To use the pre-trained Keras weights provided by the original authors, run the following scripts to download and convert them into a PyTorch checkpoint:
```bash
python scripts/download_official_dronet_model.py
python scripts/convert_keras_weights.py --keras-weights checkpoints/official/dronet/best_weights.h5 --output checkpoints/dronet_official.pt
```

---

## Usage and Execution

### Running the Simulator
1. Open your Colosseum environment in the Unreal Engine 5 Editor.
2. Ensure the "GameMode Override" in your World Settings is set to `AirSimGameMode`.
3. Press **Play** in the editor. Ensure your drone spawns successfully in the environment.

### Autonomous Flight
Open a terminal, activate your Python environment, and execute the evaluation script:
```bash
python scripts/run_evaluation.py --environment-name "Blocks" --execute-control --takeoff --num-steps 1000
```
**Command Line Arguments:**
- `--execute-control`: Sends actual velocity commands to the simulator. Without this flag, the script will perform a dry-run.
- `--takeoff`: Commands the drone to take off before engaging the AI.
- `--num-steps`: The number of inference loops to execute.

---

## Dashboard and Explainability

To visualize the AI's decision-making process in real-time, launch the Streamlit dashboard in a separate terminal window:

```bash
python -m streamlit run src/xai_dronet/dashboard/app.py
```
This will open a browser window displaying:
- The live camera feed from the drone.
- The Grad-CAM heatmap overlay highlighting identified obstacles.
- Live telemetry, including the predicted steering angle and collision probability.

---

## Directory Structure
- `checkpoints/` - Model weights and conversions.
- `docs/` - Development notes and phase documentation.
- `outputs/` - Generated logs, evaluation metrics, and image frames.
- `scripts/` - Execution scripts for inference, evaluation, and utilities.
- `src/xai_dronet/` - Core Python module containing control, inference, simulator bridges, and XAI code.

---

## License and Acknowledgements
This project is based on the original [DroNet paper](http://rpg.ifi.uzh.ch/dronet.html) by Antonio Loquercio et al. (University of Zurich). 
The simulation framework utilizes [Colosseum](https://github.com/CodexLabsLLC/Colosseum), an actively maintained fork of Microsoft AirSim.
