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
  <a href="#About-The-Project">About The Project</a> •
  <a href="#Key-Features">Key Features</a> •
  <a href="#System Architecture">System Architecture</a> •
  <a href="#Repository Structure">Repository Structure</a> •
  <a href="#Tech Stack">Tech Stack</a> •
  <a href="#Dataset-Training">Dataset & Training</a> •
  <a href="#Results-Performance">Results & Performance</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#License">License</a> •
  <a href="#Credits">Credits</a>
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

# Tech Stack
wip

---

# Dataset & Training
wip 

---

# Results & Performance
wip

---

# How To Use
wip

---

# License
wip

---

# Credits
wip

