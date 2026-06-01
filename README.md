<h2 align="center">
  <br>
    <img width="1280" height="320" alt="banner" src="https://github.com/user-attachments/assets/2997131d-db53-418c-a8d2-741263ee8f52">
</h2>

<h3 align="center"><i>Turning pixels into precision:</i> Democratizing elite gymnastics judging with Computer Vision</h4>

<h4 align="center"> <i>Bachelor's Thesis (TFG) - Universitat Autònoma de Barcelona (UAB)</i></h6>
<p align="center">
  <img src="https://img.shields.io/badge/uab-tfg-blue?style=for-the-badge" alt="uab tfg">

  <img src="https://img.shields.io/badge/status-wip-blue?style=for-the-badge" alt="status wip">
  
</p>

<p align="center">
  <a href="#About-The-Project">ABOUT THE PROJECT</a> •
  <a href="#Key-Features">KEY FEATURES</a> •
  <a href="#System Architecture">SYSTEM ARCHITECTURE</a> •
  <a href="#Repository Structure">REPOSITORY STRUCTURE</a> •
  <a href="#Tech Stack">TECH STACK</a> •
  <a href="#Results-Performance">RESULTS & PERFORMANCE</a> •
  <a href="#How-To-Use">HOW TO USE</a> •
  <a href="#License">LICENSE</a> •
  <a href="#Credits">CREDITS</a>
</p>



## ABOUT THE PROJECT

Traditional gymnastics judging is a highly demanding task. **Officials must track hyper-fast, complex joint movements in fractions of a second**, which makes the evaluation of the Execution Score *(E-Score)* naturally prone to **subjectivity** and cognitive overload.

This project is an **AI-assisted judging system** designed to **automate**, **standardize** and bring **transparency** to artistic gymnastics evaluation. Developed as a Bachelor’s Thesis *(TFG)* at the *Universitat Autònoma de Barcelona (UAB)*, this project bridges the gap between **sports biomechanics** and **Computer Vision**.

#### *HOW DOES IT WORK?*

Instead of relying on rigid heuristics, the system processes raw video inputs through a modular pipeline:
1. **_Keypoint Extraction:_** Tracks the athlete's full-body joint topology frame-by-frame.
2. **_Temporal Classification:_** Uses a Recurrent Neural Network (LSTM) to predict the specific acrobatic element (*Tuck, Pike, Split, or Straddle*) over time sequences.
3. **_Biomechanical Audit:_** Automatically calculates angular vectors and joint displacements, cross-referencing them with official regulations to flag faults like bent knees OR insufficient flexion.

#### *BROUGHT TO THE REAL WORLD*

The goal of this project is not to replace human judges, but to act as a **precise assistant**. The core of the application is an **interactive desktop dashboard** where the AI proposes timestamped deductions and kinematic breakdowns, while the human official retains ultimate control, seamlessly accepting or rejecting individual proposals via intuitive hotkeys. _**The result is a collaborative, objective, and auditable final E-Score.**_

---
  
## KEY FEATURES

To bridge the gap between high-performance technology and sports, the application is designed not just as a technical tool, but as a solution to real-world judging challenges. The core functionalities are structured around three main pillars: **the democratization of elite technology**, **mathematical objectivity**, and **human-centered efficiency**.

#### *DEMOCRATIZING ELITE TECHNOLOGY*
Brings **elite judging to local clubs, schools, and low-budget competitions**. By requiring only standard video inputs (like smartphone recordings), it eliminates the need for expensive multi-camera infrastructure or specialized biomechanical sensors.

#### *ELIMINATION OF SUBJECTIVITY*
Standardizes _E-Score_ through **mathematical analysis**. The system cross-references the athlete’s movements with official regulations, ensuring fair and unbiased scoring that protects gymnasts from human fatigue or split-second oversights.

#### *WORKFLOW EFFICIENCY*
Drastically reduces the time required to audit and review complex routines. With specialized **keyboard shortcuts**, judges can instantly navigate frame-by-frame and jump directly to critical execution peaks without breaking their workflow rhythm.

#### *IMMEDIATE ACTIONABLE FEEDBACK*
Enhances the training cycle by **exporting instant, structured chronological reports**. Coaches and athletes no longer just receive a single cold final number; they get a timestamped breakdown of exactly where and why points were deducted, **turning the evaluation into a powerful learning tool.**

---

## SYSTEM ARCHITECTURE

The project is designed following a **modular software architecture** that ensures a strict separation of concerns. The system isolates deep learning inference, temporal sequence classification, and geometric biomechanical auditing from the graphical user interface.

A complete diagram of the modular architecture and package interaction can be found below.

<img width="800" height="800" alt="Block Diagram" src="https://github.com/user-attachments/assets/4e8b7be1-a867-4381-aa8d-10004d1eec4c" />

---
## REPOSITORY STRUCTURE

The software ecosystem is strictly structured into the following specialized directories and modules:

```text
gymnastics-error-detection/
├── app.py
├── backend/
│   ├── hpe/
│   │   └── pose/
│   ├── rnn/
│   │   ├── model.py
│   │   ├── predict.py
│   │   └── train.py
│   └── scoring/
│       ├── rules.py
│       ├── score.py
│       └── evaluator.py
└── gui/
    ├── assets/
    ├── components.py
    ├── interface.py
    ├── logic.py
    └── style.py
```
The software ecosystem is strictly structured into the following specialized directories and modules:

##### *MAIN ENTRY POINT*
* **`app.py`:** The main executable script that initializes the system, setting up the application environment and launching the graphical dashboard.

##### *BACKEND MODULES (`backend/`)*
* **`hpe/pose/`:** Acts as the data ingestion layer. It manages the integration with Meta's Sapiens-2B foundation model repository to process raw video inputs (`.mp4`/`.avi`) and extract frame-by-frame joint coordinates into normalized `.json` structures. It also handles spatial interpolation and Savitzky-Golay filtering via `utils/` to guarantee skeleton stability.
* **`rnn/` (`model.py`, `predict.py`, `train.py`):** The temporal processing unit. It takes the sequential `.json` coordinates, structures them into fixed temporal blocks, and feeds them into a trained Long Short-Term Memory (LSTM) network to classify active acrobatic elements (*Tuck, Pike, Split, Straddle*) with an associated confidence percentage.
* **`scoring/`:** The biomechanical audit engine. 
  * `rules.py` maps the physical regulations of the official FIG *Code of Points* into geometric constraints.
  * `score.py` builds the *Virtual Pelvis* anchor point and calculates vectorial angles at execution peaks.
  * `evaluator.py` (`AcrobaticEvaluator`) processes these values to calculate specific penalty weights (*Minor, Medium, Severe*) and update the final *E-Score*.

##### *GRAPHICAL USER INTERFACE LAYER (`gui/`)*
* **`interface.py`:** The core window wrapper built with the PyQt6 framework. It coordinates the asynchronous desktop event loop, structures the *Dual Viewport* display layout, and captures hotkey events.
* **`components.py`:** Contains custom, reusable PyQt6 GUI elements and widgets (such as individual deduction logs, media buttons, or customized timelines) to keep `interface.py` clean and maintainable.
* **`logic.py`:** Acts as the controller or "glue code" between the frontend state and the backend evaluation engine. It handles loading data stacks, processes undo/redo requests natively, and formats execution updates for the UI.
* **`style.py`:** Centralizes all graphical themes and QSS (Qt Style Sheets) configurations, managing borders, padding, border-radius, and interactive hover states for UI elements.
* **`assets/`:** A dedicated directory that stores static graphical elements, user icons, and application media assets.

---

## TECH STACK

The project is built entirely on Python, leveraging industry-standard libraries for Deep Learning inference, advanced computer vision, mathematical signal processing, and high-performance desktop engineering.

| TECHNOLOGY | CATEGORY | ROLE |
| :--- | :--- | :--- |
| <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"> | Core Language | The foundational programming language chosen for its extensive scientific ecosystem and native integration with AI frameworks. |
| <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"> | Deep Learning | Powers the temporal deep learning execution layer, handling sequence dimensions and modern deep inference for the LSTM classifier. |
| <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"> | Computer Vision | Acts as the primary video processing engine. Handles frame decoding, coordinates pixel operations, and renders the vectorized skeleton structures dynamically. |
| <img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy"> | Signal Processing | Provides the mathematical implementation for the *Savitzky-Golay* filter to smooth joint positions and eliminate skeleton jitter. |
| <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">  <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"> | Data Manipulation | High-performance multi-dimensional array structures and data frames required to manipulate and parse structural coordinate timelines efficiently. |
| <img src="https://img.shields.io/badge/Qt-%2341CD52.svg?style=for-the-badge&logo=Qt&logoColor=white" alt="Qt"> | Graphical UI | Desktop dashboard framework (PyQt6) chosen to build the referee auditing interface, handling high-speed desktop events and native hotkeys asynchronously. |

---

## RESULTS & PERFORMANCE

The system bridges neural network predictions with real-time geometric rule checking. Below is a visual analysis of the pipeline's performance, showcasing both optimal executions and critical edge cases processed by the application.

---

#### VISUAL AUDITING & CASE ANALYSIS

##### *Optimal Case: Element Detection & Deduction Mapping*
The LSTM correctly classifies the acrobatic leap, isolates the exact geometric execution peak frame, and highlights joint infractions dynamically on the canvas. 

[add image]

> _**Biomechanical Result:**_ The system automatically maps the official FIG *Code of Points* thresholds, triggering the appropriate penalty colors (Green for Minor, Orange for Medium, Red for Severe).

##### *Critical Case: Transition False Positive*
Due to the temporal boundaries of the dataset, a transition landing frame is falsely flagged as an active leap, forcing an artificial severe joint penalty due to the ground impact compression[cite: 1].

[add image]

>_**Human-in-the-Loop Solution:**_ This technical limitation highlights the vital necessity of our collaborative architecture. The user interface allows judges to instantly bypass and override these localized temporal anomalies with a single click, keeping human expertise at the center of the scoring process.

---

#### *QUANTITATIVE METRICS & TRAINING REPORT*

All raw mathematical data, comparative benchmarks between Human Pose Estimation architectures (Sapiens-2B vs. YOLO vs. MediaPipe), and the LSTM confusion matrix have been offloaded to maintain a clean main workflow.

>**_Looking for the numbers?_** For an extensive breakdown of the accuracy scores, data augmentation techniques, and loss functions, please check the report [Metrics and Model Evaluation](./metrics/README.md).


#### *COMPUTATIONAL PERFORMANCE & LATENCY*
* **Processing Cost:** Processing a continuous 1-minute and 20-second video routine (approx. 2400 frames) requires **3 to 4 hours** of execution utilizing an *NVIDIA GeForce RTX 2080 Ti* GPU.
* **Engineering Trade-off:** This processing latency is a direct consequence of deploying a foundational model with 2 billion parameters Sapiens-2B. However, this trade-off is highly justified, as it provides the extreme anatomical robustness required to accurately track multi-axial athletic movements from low-cost, monocular standard camera devices, democratizing sports analytics for grassroots clubs.


---

# How To Use
wip

---

# License
wip

---

# Credits
wip

