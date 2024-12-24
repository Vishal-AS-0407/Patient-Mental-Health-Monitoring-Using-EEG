---

# 🧠 **EEG Signal Analysis and Recommendation System**

## Project Overview

This project presents a comprehensive system that uses **EEG signals** to provide real-time health analysis and personalized recommendations. The system processes EEG data to assess various physiological and psychological states, including:
- Mental states
- Cognitive load
- Emotions

The signals are first subjected to **biomedical signal preprocessing**, then fed into deep learning models trained on specialized datasets targeting these states. The outputs from these models are processed by a **Large Language Model (LLM)** to generate a detailed health report. This report provides cumulative insights into a person's well-being, including recommendations for:

- Personalized activities
- Dietary plans
- Medical advice

By correlating multiple EEG signals, the system can identify potential health risks and offer tailored suggestions, such as:

- Rest or sleep recommendations
- Focus and cognitive load management
- Consultation with a healthcare provider

The system offers a holistic view of a person's health, enabling them to make informed decisions for a healthier lifestyle.

---

## 📊 **Datasets Used**

1. **DREAMER Dataset**:  
   - A dataset for emotion recognition based on EEG signals.  
   - Includes valence, arousal, and dominance ratings based on audiovisual stimuli.  
   - 🧠 **EEG Channels**: 14 channels with a sampling rate of 128 Hz.  

2. **INRIA BCI Challenge Dataset**:  
   - A dataset for brain-computer interface classification focused on error-related potentials (ERP).  
   - 🧠 **EEG Sensors**: 56 passive EEG sensors, recorded at 200 Hz.  

---

## 🗂️ **Files and Functions**

| File           | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `bcipre.py`    | Preprocessing script for the BCI dataset.                                  |
| `bcitrain.py`  | Training script for the BCI model.                                         |
| `bcitest.py`   | Testing script for the trained BCI model.                                  |
| `dreamerpre.py`| Preprocessing script for the DREAMER dataset.                              |
| `drtrain.py`   | Training script for the DREAMER model.                                     |
| `drtest.py`    | Testing script for the trained DREAMER model.                              |
| `biogpt.py`    | Inference script to get predictions from trained models.                   |
| `gemini.py`    | Generates the health report using a large language model (LLM).            |
| `htmlgen.py`   | Converts the health report into a visually appealing HTML page.            |

---

## ⚙️ **Requirements**
To run this project, you need the following dependencies. Install them via the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 🛠️ **Setup and Usage**

### 1️⃣ Clone the Repository:
```bash
git clone https://github.com/your-username/eeg-signal-analysis.git
cd eeg-signal-analysis
```

### 2️⃣ Install Dependencies:
```bash
pip install -r requirements.txt
```

### 3️⃣ Preprocess Data:
Run the preprocessing scripts for the respective datasets:
```bash
python bcipre.py
python dreamerpre.py
```

### 4️⃣ Train Models:
Train the models for both datasets:
```bash
python bcitrain.py
python drtrain.py
```

### 5️⃣ Generate Reports:
Use the `biogpt.py` and `gemini.py` scripts to process model outputs and generate detailed health reports:
```bash
python biogpt.py
python gemini.py
```

---


## 🚀 **Expected Benefits**

1. **Student Learning Enhancement**:  
   Tailor education based on cognitive load and mental readiness.  

2. **Workplace Productivity Optimization**:  
   Dynamically adjust workloads to improve efficiency and reduce stress.  

3. **Healthcare and Psychological Support**:  
   Assist doctors and psychologists in diagnosing mental states for better treatment plans.  

4. **Versatile Applications**:  
   Useful in sports, therapy, and performance coaching to optimize well-being and decision-making.  

---

### 🌟 **Show Your Support**
If you find this project helpful, give it a ⭐ on GitHub and share it with others! 😊

---
