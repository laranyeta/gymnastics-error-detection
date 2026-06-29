<h2 align="center">
  <br>
    <img width="1280" height="320" alt="Image" src="https://github.com/user-attachments/assets/ae2f7424-bb72-4c26-85f8-02ebff4c15d2" />
</h2>

<h3 align="center"><samp><i>Turning pixels into precision:</i> Democratizing elite gymnastics judging with Computer Vision</samp></h3>

<h4 align="center"> <samp><i>Bachelor's Thesis (TFG) - Universitat Autònoma de Barcelona (UAB)</i></h6>
<p align="center"></samp>
  <img src="https://img.shields.io/badge/status-done-green?style=for-the-badge" alt="status done">

  <a href="https://raw.githubusercontent.com/laranyeta/gymnastics-error-detection/main/docs/TFG_InformeFinal.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Read Thesis">
  </a>
  <a href="https://raw.githubusercontent.com/laranyeta/gymnastics-error-detection/main/docs/TFG_Poster.pdf">
    <img src="https://img.shields.io/badge/Poster-PDF-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Read Poster">
  </a>
</p>

<p align="center">
  <sub>
    <a href="#About-The-Project"><samp>ABOUT THE PROJECT</samp></a> •
    <a href="#Key-Features"><samp>KEY FEATURES</samp></a> •
    <a href="#System-Architecture"><samp>SYSTEM ARCHITECTURE</samp></a> •
    <a href="#Repository-Structure"><samp>REPOSITORY STRUCTURE</samp></a> •
    <a href="#Tech-Stack"><samp>TECH STACK</samp></a> •
    <a href="#Results-Performance"><samp>RESULTS & PERFORMANCE</samp></a> •
    <a href="#How-To-Use"><samp>HOW TO USE</samp></a> •
    <a href="#License"><samp>LICENSE</samp></a> •
    <a href="#Credits-Acknowledgements"><samp>CREDITS & ACKNOWLEDGEMENTS</samp></a>
  </sub>
</p>

## <samp>ABOUT THE PROJECT</samp>

Traditional gymnastics judging is a highly demanding task. **Officials must track hyper-fast, complex joint movements in fractions of a second**, which makes the evaluation of the Execution Score *(E-Score)* naturally prone to **subjectivity** and cognitive overload.

This project is an **AI-assisted judging system** designed to **automate**, **standardize** and bring **transparency** to artistic gymnastics evaluation. Developed as a Bachelor’s Thesis *(TFG)* at the *Universitat Autònoma de Barcelona (UAB)*, this project bridges the gap between **sports biomechanics** and **Computer Vision**.

### <samp>> *HOW DOES IT WORK?*</samp>

Instead of relying on rigid heuristics, the system processes raw video inputs through a modular pipeline:</br>
<samp>1 **Keypoint Extraction:**</samp> Tracks the athlete's full-body joint topology frame-by-frame.</br>
<samp>2 **Temporal Classification:**</samp> Uses a Recurrent Neural Network (LSTM) to predict the specific acrobatic element (*Tuck, Pike, Split, or Straddle*) over time sequences. </br>
<samp>3 **Biomechanical Audit:**</samp> Automatically calculates angular vectors and joint displacements, cross-referencing them with official regulations to flag faults like bent knees OR insufficient flexion.

### <samp>> *THE ACROBATIC ELEMENTS*</samp>

The system currently focuses on the **four foundational** aerial leap postures in gymnastics, analyzing the geometric execution of each at their peak frame:

<p align="center">
  <img width="902" height="199" alt="Image" src="https://github.com/user-attachments/assets/8c5d0be1-60f9-46da-b523-0125de9fe5a1" />
</p>

Sorted (left to right) acrobatics are:

* **<samp>TUCK:</samp>** The gymnast brings their knees tightly to their chest, folding the body. The system penalizes loose tucks or insufficient knee flexion (angles >100°).
* **<samp>PIKE:</samp>** The body is bent sharply at the hips with the legs kept perfectly straight. The AI monitors the hip angle for sufficient compression and penalizes bent knees.
* **<samp>SPLIT:</samp>** An aerial leap where the legs are extended in opposite directions (front and back). The algorithm checks for a perfect 180° vectorial opening and straight knees.
* **<samp>STRADDLE:</samp>** A leap where the legs are extended laterally (sideways) rather than front-to-back. Similar to the split, the system calculates the peak angular opening and knee straightness.

### <samp>> *BROUGHT TO THE REAL WORLD*</samp>

The goal of this project is not to replace human judges, but to act as a **precise assistant**. The core of the application is an **interactive desktop dashboard** where the AI proposes timestamped deductions and kinematic breakdowns, while the human official retains ultimate control, seamlessly accepting or rejecting individual proposals via intuitive hotkeys. _**The result is a collaborative, objective, and auditable final E-Score.**_
<h2 align="center">
  <img width="800" height="556" alt="Image" src="https://github.com/user-attachments/assets/a2058db4-9d84-498c-b951-873cd2d2df5c" />
</h2>

> _<samp>**READ THE FULL RESEARCH PAPER:**</samp> The complete mathematical methodology, model training logs, and full FIG Code of Points geometric mapping can be found in the **[Official Thesis Document (PDF)](https://raw.githubusercontent.com/laranyeta/gymnastics-error-detection/main/docs/TFG_InformeFinal.pdf)**._

---
  
## <samp>KEY FEATURES</samp>

To bridge the gap between high-performance technology and sports, the application is designed not just as a technical tool, but as a solution to real-world judging challenges. The core functionalities are structured around three main pillars: **the democratization of elite technology**, **mathematical objectivity**, and **human-centered efficiency**.

### <samp>> *DEMOCRATIZING ELITE TECHNOLOGY*</samp>
Brings **elite judging to local clubs, schools, and low-budget competitions**. By requiring only standard video inputs (like smartphone recordings), it eliminates the need for expensive multi-camera infrastructure or specialized biomechanical sensors.

### <samp>> *ELIMINATION OF SUBJECTIVITY*</samp>
Standardizes _E-Score_ through **mathematical analysis**. The system cross-references the athlete’s movements with official regulations, ensuring fair and unbiased scoring that protects gymnasts from human fatigue or split-second oversights.

### <samp>> *WORKFLOW EFFICIENCY*</samp>
Drastically reduces the time required to audit and review complex routines. With specialized **keyboard shortcuts**, judges can instantly navigate frame-by-frame and jump directly to critical execution peaks without breaking their workflow rhythm.

### <samp>> *IMMEDIATE ACTIONABLE FEEDBACK*</samp>
Enhances the training cycle by **exporting instant, structured chronological reports**. Coaches and athletes no longer just receive a single cold final number; they get a timestamped breakdown of exactly where and why points were deducted, **turning the evaluation into a powerful learning tool.**

---

## <samp>SYSTEM ARCHITECTURE</samp>

The project is designed following a **modular software architecture** that ensures a strict separation of concerns. The system isolates *Deep Learning* inference, temporal sequence classification, and geometric biomechanical auditing from the graphical user interface.

A complete diagram of the modular architecture and package interaction can be found below.

<img width="800" height="800" alt="Block Diagram" src="https://github.com/user-attachments/assets/4e8b7be1-a867-4381-aa8d-10004d1eec4c" />

---
## <samp>REPOSITORY STRUCTURE</samp>

The software ecosystem is strictly structured into the following specialized directories and modules:

```text
gymnastics-error-detection/
├── app.py
├── backend/
│   ├── hpe/
│   │   └── pose/
│   │       ├── main.py
│   │       └── utils/
│   │           ├── data.py
│   │           └── vision.py 
│   ├── rnn/
│   │   ├── model.py
│   │   ├── process.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── evaluate.py
│   │   └── score.py
│   └── scoring/
│       ├── rules.py
│       └── evaluator.py
└── gui/
    ├── assets/
    ├── components.py
    ├── interface.py
    ├── logic.py
    └── style.py
```
The software ecosystem is strictly structured into the following specialized directories and modules:

### <samp>> *MAIN ENTRY POINT*</samp>
* **`app.py`:** The main executable script that initializes the system, setting up the application environment and launching the graphical dashboard.

### <samp>> *BACKEND MODULES (`backend/`)*</samp>
* **`hpe/pose/`:** Acts as the data ingestion layer, managing the integration with *Meta*'s Sapiens-2B foundation model repository.
  * `main.py` processes raw video inputs and extract frame-by-frame joint coordinates into normalized JSON structures. 
  * `data.py` processes the raw spatial data and calculates joint angle for the acrobatic classificator.
  * `vision.py` handles spatial interpolation and *Savitzky-Golay* filtering to guarantee skeleton stability and *KNN Imputation* to assess *frame-dropping*.
* **`rnn/`:** The temporal processing unit. It takes the sequential JSON coordinates, structures them into fixed temporal blocks, and feeds them into a trained Long Short-Term Memory (LSTM) network to classify active acrobatic elements (*Tuck, Pike, Split, Straddle*) with an associated confidence percentage.
  * `model.py` calls the LSTM model into the `RNNAcrobaticClassificator` class with customized parameters.
  * `process.py` structures JSON data into fixed temporal blocks and data augmentation logic for the main video dataset.
  * `train.py` trains the LSTM model generating the `best.pth` checkpoint file.
  * `predict.py` runs an inference on a singular frame, returning predicted class and confidence.
  * `evaluate.py` creates confusion matrix and generates metrics to evaluate the trained LSTM model on our dataset
  * `score.py` builds the *Virtual Pelvis* anchor point and calculates vectorial angles at execution peaks.
* **`scoring/`:** The biomechanical audit engine. 
  * `rules.py` maps the physical regulations of the official FIG *Code of Points* into geometric constraints.
  * `evaluator.py` processes these values in the new generated class `AcrobaticEvaluator` to calculate specific penalty weights (*Minor, Medium, Severe*) and update the final *E-Score*.

### <samp>> *GRAPHICAL USER INTERFACE LAYER (`gui/`)*</samp>
* **`interface.py`:** The core window wrapper built with the PyQt6 framework. It coordinates the asynchronous desktop event loop, structures the *Dual Viewport* display layout, and captures hotkey events.
* **`components.py`:** Contains custom, reusable PyQt6 GUI elements and widgets (such as individual deduction logs, media buttons, or customized timelines) to keep `interface.py` clean and maintainable.
* **`logic.py`:** Acts as the controller or "glue code" between the frontend state and the backend evaluation engine. It handles loading data stacks, processes undo/redo requests natively, and formats execution updates for the UI.
* **`style.py`:** Centralizes all graphical themes and QSS (Qt Style Sheets) configurations, managing borders, padding, border-radius, and interactive hover states for UI elements.
* **`assets/`:** A dedicated directory that stores static graphical elements, user icons, and application media assets.

---

## <samp>TECH STACK</samp>

The project is built entirely on Python, leveraging industry-standard libraries for *Deep Learning* inference, advanced Computer Vision, mathematical signal processing, and high-performance desktop engineering.

| TECHNOLOGY | CATEGORY | ROLE |
| :--- | :--- | :--- |
| <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"> | Core Language | The foundational programming language chosen for its extensive scientific ecosystem and native integration with AI frameworks. |
| <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"> | Deep Learning | Powers the temporal deep learning execution layer, handling sequence dimensions and modern deep inference for the LSTM classifier. |
| <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"> | Computer Vision | Acts as the primary video processing engine. Handles frame decoding, coordinates pixel operations, and renders the vectorized skeleton structures dynamically. |
| <img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy"> | Signal Processing | Provides the mathematical implementation for the *Savitzky-Golay* filter to smooth joint positions and eliminate skeleton jitter. |
| <img src="https://img.shields.io/badge/Sklearn-%230C55A5.svg?style=for-the-badge&logo=Scikit-Learn&logoColor=white" alt="Sklearn"> | Data Imputation | Implements KNN Imputation to prevent frame dropping issues between frames. |
| <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">  <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"> | Data Manipulation | High-performance multi-dimensional array structures and data frames required to manipulate and parse structural coordinate timelines efficiently. |
| <img src="https://img.shields.io/badge/Qt-%2341CD52.svg?style=for-the-badge&logo=Qt&logoColor=white" alt="Qt"> | Graphical UI | Desktop dashboard framework (PyQt6) chosen to build the referee auditing interface, handling high-speed desktop events and native hotkeys asynchronously. |

---

## <samp>RESULTS & PERFORMANCE</samp>

The system bridges neural network predictions with real-time geometric rule checking. Below is a visual analysis of the pipeline's performance, showcasing both optimal executions and critical edge cases processed by the application.

### <samp>> *VISUAL AUDITING & CASE ANALYSIS*</samp>

#### <samp>*OPTIMAL CASE: Element Detection & Deduction Mapping*</samp>
The LSTM correctly classifies the acrobatic leap, isolates the exact geometric execution peak frame, and highlights joint infractions dynamically on the canvas. 

<img width="1274" height="452" alt="Correct Behaviour" src="https://github.com/user-attachments/assets/66655bac-215d-44c4-ac20-53c3e7822774" />

> _**<samp>BIOMECHANICAL RESULT:</samp>**_ The system automatically maps the official FIG *Code of Points* thresholds, triggering the appropriate penalty colors (Green for Minor, Orange for Medium, Red for Severe).

#### <samp>*CRITICAL CASE: Transition False Positive*</samp>
Due to the temporal boundaries of the dataset, a transition landing frame is falsely flagged as an active leap, forcing an artificial severe joint penalty due to the ground impact compression.
<img width="1284" height="434" alt="Bad Behaviour" src="https://github.com/user-attachments/assets/f5a0e50a-61a2-4b3f-8b29-73c3b3496b0f" />

>_**<samp>HUMAN-IN-THE-LOOP SOLUTION</samp>:**_ This technical limitation highlights the vital necessity of our collaborative architecture. The user interface allows judges to instantly bypass and override these localized temporal anomalies with a single click, keeping human expertise at the center of the scoring process.

---

### <samp>> *QUANTITATIVE METRICS & TRAINING REPORT*</samp>

All raw mathematical data, comparative benchmarks between Human Pose Estimation architectures (Sapiens-2B vs. YOLO vs. MediaPipe), and the LSTM confusion matrix have been offloaded to maintain a clean main workflow.

>**_<samp>LOOKING FOR THE NUMBERS?</samp>_** For an extensive breakdown of the accuracy scores, data augmentation techniques, and loss functions, please check the report [Metrics and Model Evaluation](./metrics/README.md).


### <samp>> *COMPUTATIONAL PERFORMANCE & LATENCY*</samp>
* **<samp>Processing Cost:</samp>** Processing a continuous 1-minute and 20-second video routine (approx. 2400 frames) requires **3 to 4 hours** of execution utilizing an *NVIDIA GeForce RTX 2080 Ti* GPU.
* **<samp>Engineering Trade-off:</samp>** This processing latency is a direct consequence of deploying a foundational model with 2 billion parameters Sapiens-2B. However, this trade-off is highly justified, as it provides the extreme anatomical robustness required to accurately track multi-axial athletic movements from low-cost, monocular standard camera devices, democratizing sports analytics for grassroots clubs.


---

## <samp>HOW TO USE</samp>
The project is designed to be accessible for both non-technical end-users (judges and coaches) and developers. You can either run the pre-packaged desktop application or clone the repository to run it from the source code.

### <samp> > HOW DO I RUN THE APP? </samp>

### <samp>Method A: Standalone Executable *(Recommended for End-Users)*</samp>
You do not need to install Python or any dependencies. 
1. Navigate to the **[Releases](../../releases)** tab on this GitHub repository.
2. Download the compressed file for your operating system (`macOS`, `Windows`, or `Linux`).
3. Extract the folder and double-click the `GymnasticsErrorDetector` executable to launch the application.

### <samp>Method B: Running from Source *(For Developers)*</samp>
If you prefer to run the application natively via Python, ensure you have Python 3.9+ installed and follow these steps:

```bash
#clone the repository
git clone [https://github.com/laranyeta/gymnastics-error-detection.git](https://github.com/laranyeta/gymnastics-error-detection.git)
cd gymnastics-error-detection

#create and activate a virtual environment (optional but recommended)
#macos/linux
python3.11 -m venv venv
source venv/bin/activate

#windows
py -3.11 -m venv venv
venv\Scripts\activate

#install the required dependencies
pip install -r requirements.txt

#launch the application
python3 -m app
```
### <samp> > HOW DO I PROCESS MY OWN VIDEOS? </samp>
To evaluate a brand new gymnastics routine, the raw video must first be processed by the *Deep Learning* extraction layer to generate the kinematic coordinate JSON file.

Because foundation models like Meta's Sapiens-2B require specific system architectures and heavy dependencies, we have isolated this process using Docker to prevent local dependency conflicts.

> **<samp>STEP-BY-STEP EXTRACTION GUIDE:</samp>** Please refer to the **[Human Pose Estimation (HPE) Module Documentation](backend/hpe/pose/checkpoints/README.md)** for detailed instructions on downloading the Sapiens model weights.

#### <samp> 1  *NAVIGATE TO THE HPE MODULE*</samp>
```
cd backend/hpe
```

#### <samp> 2  *BUILD THE DOCKER IMAGE*</samp>
We provide a ready-to-use Dockerfile that contains all necessary PyTorch and Computer Vision libraries.

```
docker build -t sapiens-extractor
```

#### <samp> 3  *RUN THE EXTRACTION*</samp>
Mount your local video folder to the container and run the inference script. The script will output a normalized .json file containing the frame-by-frame joint coordinates.
```
docker run --gpus '"device=0"' --shm-size=24g -it -v $(pwd):/workspace sapiens-extractor
```
> In parameter `device` you must put your own available device.

Then run the `main.py` specifying your own video path source (in case it's saved in the main workspace directory) and your desired output directory (optional).
```
python extract.py --input /workspace/my_video.mp4 --output /workspace/results/
```

#### <samp> 4  *LOAD IT TO THE APP*</samp>
Place the newly generated `my_video.json` file in the same directory as your `my_video_skeleton.avi` video file. When you load the video into the desktop application, the system will automatically detect and pair the kinematic data!

---

## <samp> LICENSE </samp>

This project is licensed under the **MIT License** *(see the **[LICENSE](LICENSE)** file for details)*

#### <samp>> **Third-Party Intellectual Property**</samp>
The Human Pose Estimation extraction module utilizes Meta's Sapiens-2B foundation model. The weights and architecture of Sapiens are subject to Meta's specific licensing terms and non-commercial research agreements. Please refer to their official repository for detailed compliance information before deploying this module in production.

---

## <samp>CREDITS & ACKNOWLEDGEMENTS</samp>

This thesis and software project would not have been possible without the support, resources, and encouragement provided by several key people and institutions:

> <samp>**Coen Antens *(Project Tutor)*:**</samp> For trusting the viability of this ambitious idea from day one, and for providing invaluable guidance, academic support, and technical feedback throughout the development process at the Universitat Autònoma de Barcelona (UAB).

> <samp>**CVC *(Internship Company)*:**</samp> For their continuous support and for facilitating the vital hardware resources, specifically the high-performance GPU infrastructure required to successfully run and test the heavy Deep Learning inference models.

> <samp>**Family, Partner & Friends:**</samp> For their unconditional patience, encouragement, and emotional support during the countless long hours of research, debugging, and coding.

