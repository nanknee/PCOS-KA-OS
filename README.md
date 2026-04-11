# PCOS Detection — Multi-Output SVM Classifier

A machine learning web application that detects **Polycystic Ovary Syndrome (PCOS)** and predicts associated symptoms using a Support Vector Machine (SVM) ClassifierChain model, served via a Flask API with an interactive browser UI.

> **Disclaimer:** This project is for research and educational purposes only. It is not a substitute for clinical diagnosis.

---

## Features

- Multi-output prediction: simultaneously predicts PCOS diagnosis + 4 associated symptoms
- SVM with RBF kernel wrapped in a `ClassifierChain` (label dependencies preserved)
- Interactive web UI with sliders for all clinical inputs
- REST API endpoint for programmatic access
- Confidence scores for every prediction

---

## Project Structure

```
PCOS-KA-OS/
│
├── pcosdetection.ipynb       # EDA, training, evaluation
├── app.py                    # Flask API + interactive UI
├── pcos_chain_svm.pkl        # Trained model (generated after running notebook)
├── scaler.pkl                # Fitted StandardScaler (generated after running notebook)
├── PCOS_data.csv             # Dataset
└── README.md
```

---

## Model Details

### Algorithm
- **Model:** `ClassifierChain` wrapping `SVC`
- **Kernel:** RBF (`rbf`)
- **C:** 10
- **Gamma:** `scale`
- **Class weight:** `balanced`

### Input Features (10 clinical variables)

| Feature | Description |
|---|---|
| Follicle No. (L) | Number of follicles — left ovary |
| Follicle No. (R) | Number of follicles — right ovary |
| AMH (ng/mL) | Anti-Müllerian hormone |
| FSH/LH | FSH to LH ratio |
| LH (mIU/mL) | Luteinizing hormone |
| Waist:Hip Ratio | Waist-to-hip measurement ratio |
| BMI | Body Mass Index |
| Cycle length (days) | Menstrual cycle length |
| Avg. F size L (mm) | Average follicle size — left ovary |
| Avg. F size R (mm) | Average follicle size — right ovary |

### Output Labels (5 predictions)

| Label | Description |
|---|---|
| PCOS (Y/N) | Primary PCOS diagnosis |
| Weight gain (Y/N) | Associated weight gain |
| Hair growth (Y/N) | Excess hair growth (hirsutism) |
| Skin darkening (Y/N) | Skin darkening symptom |
| Cycle (R/I) | Cycle regularity (regular/irregular) |

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- Conda or pip

### 1. Clone the repository
```bash
git clone https://github.com/nanknee/PCOS-KA-OS.git
cd PCOS-KA-OS
```

### 2. Install dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn flask jupyter
```

### 3. Train the model
Open and run all cells in `pcosdetection.ipynb`. This will:
- Load and clean the dataset
- Train the SVM ClassifierChain
- Save `pcos_chain_svm.pkl` and `scaler.pkl`

### 4. Run the web app
```bash
python app.py
```

Open your browser at `http://127.0.0.1:5001`

---

## API Usage

### `POST /predict`

**Request:**
```json
{
  "features": [8, 9, 3.2, 1.4, 5.1, 0.82, 23.5, 30, 11.0, 12.0]
}
```

Feature order must match: `[Follicle L, Follicle R, AMH, FSH/LH, LH, Waist:Hip, BMI, Cycle length, F size L, F size R]`

**Response:**
```json
{
  "predictions": [1, 1, 0, 1, 1],
  "probabilities": [0.91, 0.78, 0.34, 0.82, 0.87]
}
```

`predictions[0]` = PCOS (1 = positive, 0 = negative), followed by the 4 symptom labels in the same order.

**Test with curl:**
```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [8, 9, 3.2, 1.4, 5.1, 0.82, 23.5, 30, 11.0, 12.0]}'
```

---

## Dataset

- **Source:** PCOS Extended Dataset
- **Size:** 2000 patient records
- **Features:** 44 clinical variables
- **Target:** PCOS diagnosis + symptom flags

Key preprocessing steps:
- `AMH(ng/mL)` and `II beta-HCG(mIU/mL)` converted from object to numeric
- Missing values filled with column median
- Features standardised using `StandardScaler`
- 80/20 train/test split with stratification on primary PCOS label

---

## Tech Stack

| Component | Technology |
|---|---|
| ML model | scikit-learn SVC + ClassifierChain |
| Data processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Web framework | Flask |
| Frontend | HTML/CSS/JS (embedded in Flask) |
| Environment | GitHub Codespaces / Jupyter |

---

## Running in GitHub Codespaces

1. Open the repository in Codespaces
2. Run the notebook to generate the `.pkl` files
3. Run `python app.py` in the terminal
4. Go to the **PORTS** tab → right-click port `5001` → set visibility to **Public**
5. Click the globe icon to open the forwarded URL in your browser
