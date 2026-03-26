# 🚗 Autonomous-vehicle-pipeline

---

## 📌 Overview
Learning and implementing core algorithms used in autonomous vehicles.

---

## 🎯 Objectives
- Understand how key algorithms work in real world scenarios  
- Implement and visualize each component  
- Build a modular autonomous driving pipeline step by step  
- Compare different approaches *(e.g., Kalman vs Particle Filter)*  

---

## 🧠 System Architecture

The learning is structured around a real autonomous vehicle pipeline:

**Sensors → Perception → Localization → Planning → Control → Decision Making**

Each module will be:
- Implemented  
- Tested  
- Later integrated  

---


## Repository Structure
```text
.
├── perception/
├── localization/
├── planning/
├── control/
├── decision_making/
├── learning/
├── simulations/
└── README.md

```
Each folder contains:
- Implementations  
- Visualizations  
- Notes and explanations  

---

## 🛣️ Learning Roadmap

### 1. Perception
- CNN basics  
- Object Detection *(R-CNN, Fast R-CNN, Faster R-CNN)*  
- YOLO / SSD  
- Semantic Segmentation  
- Stereo Vision  

---

### 2. Localization & State Estimation
- Kalman Filter  
- Extended Kalman Filter  
- Unscented Kalman Filter  
- Particle Filter  
- SLAM  

---

### 3. Planning
- A*  
- Dijkstra  
- D*  
- RRT  
- PRM  

---

### 4. Control
- PID Controller  
- Trajectory tracking  

---

### 5. Decision Making
- Finite State Machines  
- MDPs  
- Reinforcement Learning *(Q-learning, SARSA, Actor-Critic)*  

---

### 6. Learning Methods
- Supervised / Unsupervised Learning  
- Imitation Learning  
- DAgger  

---

## 🛠️ Tools & Technologies
- Python  
- ROS  
- CARLA Simulator  
- OpenCV  
- TensorFlow  
- NumPy / Matplotlib  

---

## NOTE 
- #The accuracy of the genrated graphs r low -In side the logic its not mathematically perfect as I produced the graphs to get an understanding about what happens inside the algorithms and filters .
