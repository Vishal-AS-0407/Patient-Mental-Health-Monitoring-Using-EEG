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
   - [DREAMER Dataset](https://zenodo.org/record/546113)

2. **INRIA BCI Challenge Dataset**:  
   - A dataset for brain-computer interface classification focused on error-related potentials (ERP).  
   - 🧠 **EEG Sensors**: 56 passive EEG sensors, recorded at 200 Hz.  
   - [INRIA BCI Challenge Dataset](https://www.kaggle.com/c/inria-bci-challenge)

---

## 🗂️ **Files and Functions**

| File           | Description                                                                 | Outputs Report                                               |
|----------------|-----------------------------------------------------------------------------|-------------------------------------------------------------|
| `bcipre.py`    | Preprocessing script for the BCI dataset.                                  | Preprocessed EEG data for error-related potential analysis. |
| `bcitrain.py`  | Training script for the BCI model.                                         | Trained model for detecting ERP states.                     |
| `bcitest.py`   | Testing script for the trained BCI model.                                  | Test accuracy, confusion matrix, and predictions.           |
| `dreamerpre.py`| Preprocessing script for the DREAMER dataset.                              | Preprocessed EEG data for emotion recognition.              |
| `drtrain.py`   | Training script for the DREAMER model.                                     | Trained model for emotion state classification.             |
| `drtest.py`    | Testing script for the trained DREAMER model.                              | Test accuracy, confusion matrix, and predictions.           |
| `biogpt.py`    | Inference script to get predictions from trained models.                   | Psychological parameters: valence, arousal, and dominance.  |
| `gemini.py`    | Generates the health report using a large language model (LLM).            | Personalized health recommendations.                        |
| `htmlgen.py`   | Converts the health report into a visually appealing HTML page.            | A user-friendly HTML health report.                         |

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

## 🖼️ **Outputs and Pipeline**

### Overall Pipeline  
![Pipeline](https://github.com/user-attachments/assets/fd777c6b-6584-439e-9034-fce58b694226)

### Sample Reports  
Below are sample output reports.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1ade8f7d-4fb6-42ce-9566-9264c391ec64" alt="Report 1" />
  <img src="https://github.com/user-attachments/assets/8984b89a-6eda-484d-a180-f0ae60f6bd90" alt="Report 2" />
  <img src="https://github.com/user-attachments/assets/7dc6f1d3-354a-470e-9b2c-28726a657c84" alt="Report 3" />
</p>


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

