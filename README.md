# **Sleep Apnea Prediction Tool**  
🌙 A Machine Learning-based application to predict sleep apnea and classify its severity using Polysomnography (PSG) data in EDF format.

[👉 **Live Demo**](https://sleep-disorder-ml-project.streamlit.app/)

---

## **Overview**  
This tool utilizes advanced Machine Learning models to process respiratory and EEG signals, estimate sleep time, detect apnea/hypopnea events, and classify sleep apnea severity. It is designed to assist researchers and clinicians in analyzing sleep disorders efficiently.

---

## **Key Features**  

### 1. **Apnea-Hypopnea Index (AHI) Prediction**  
Predicts the Apnea-Hypopnea Index (AHI) based on PSG data.

### 2. **Sleep Apnea Classification**  
Classifies sleep apnea severity into:
- **No Sleep Apnea**  
- **Mild Sleep Apnea**  
- **Moderate Sleep Apnea**  
- **Severe Sleep Apnea**

### 3. **Event Detection**  
Detects respiratory events such as **apnea** and **hypopnea** from PSG signals.

### 4. **Sleep Time Estimation**  
Estimates total sleep time using EEG delta band power.

---

## **How It Works**  

### **Signals Processed**
- **Respiratory (oro-nasal airflow)**: Detects apnea and hypopnea events.  
- **EEG (Fpz-Cz)**: Estimates sleep time by analyzing delta band (0.5–4 Hz) power.

### **Models Used**
- **Random Forest Regressor**: Predicts AHI from respiratory and EEG features (`rf_regressor.pkl`).  
- **Support Vector Machine (SVM)**: Classifies sleep apnea severity (`best_svm.pkl`).  

---

## **Data Source**  
We use the [PhysioNet Sleep-EDF Expanded Dataset](https://physionet.org/content/sleep-edfx/1.0.0/), a publicly available collection of PSG recordings.

---

## **Performance Metrics**  
The SVM model achieves outstanding performance:  

- **Cross-validation Accuracy Scores**:  
  `[0.947, 0.789, 0.946, 0.946, 0.919]`  
- **Mean Accuracy**:  
  `90.95%`

---

## **Installation and Usage**  

### 1. **Clone the Repository**
```bash
git clone https://github.com/your-username/sleep-apnea-prediction-tool.git
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Run the Application**
```bash
streamlit run app.py
```

### 4. **Upload Your Data**
- Upload your Polysomnography (PSG) files in **EDF format** to get predictions.

---

## **Screenshots**  
![Tool Interface](gradient_Squad_Buaters_Screenshot (3).png)  
_A clean and intuitive interface for efficient analysis._

---

## **Future Improvements**
- Extend compatibility to include additional PSG signals.  
- Implement a deeper neural network for enhanced prediction accuracy.  
- Add features like visualization of detected events and raw signals.  

---

## **Contributions**
Contributions are welcome! Please open an issue or submit a pull request for any suggestions or improvements.

---

## **License**  
This project is licensed under the **MIT License**.  
