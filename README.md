# SkinTell 

A computer vision project that classifies facial skin conditions (acne types, rosacea, normal) using deep learning and transfer learning with MobileNetV2.

---

## Project Structure

```
skin-condition-classifier/
├── data/
│   ├── raw/              # Original downloaded dataset
│   └── processed/        # Cleaned, resized, split images
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── preprocess.py     # Image preprocessing utilities
│   ├── model.py          # Model definition
│   └── evaluate.py       # Evaluation utilities
├── models/               # Saved model weights
├── plots/                # Output visualizations
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses the **ACNE04 dataset** from Kaggle, downloaded automatically via `kagglehub`.

**Classes:**
- `acne`
- `redness`
- `dark_spots`
- `normal`

The dataset is downloaded automatically when running the first notebook. You do not need to download anything manually.

**Setup Kaggle API access:**
1. Go to kaggle.com → Settings → API → **Create New Token**
2. Run the following in your terminal:
```powershell
# Windows
[System.IO.File]::WriteAllText("$HOME\.kaggle\access_token", "your_token_here")

# Mac/Linux
mkdir -p ~/.kaggle && echo "your_token_here" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
```
3. Run notebook `01_data_exploration.ipynb` — the dataset downloads automatically.

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/skin-condition-classifier.git
cd skin-condition-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order (01 → 04)
```

---

## Approach

1. **Exploratory Analysis** — class distribution, sample images, image size consistency
2. **Preprocessing** — resize to 224x224, normalize, augment (flip, rotate, brightness)
3. **Modeling** — Transfer learning with MobileNetV2 pretrained on ImageNet, fine-tuned on skin data
4. **Evaluation** — Accuracy, confusion matrix, per-class precision/recall, sample predictions

---

## Key Findings

*(To be updated after training)*

---

## Sample Predictions

*(To be updated after training)*

---

## Limitations & Ethics

- This model is **not a medical diagnostic tool**. It is a learning project only.
- Performance depends heavily on dataset quality and diversity.
- Skin condition datasets may underrepresent certain skin tones — model predictions may be less accurate across all demographics.
- Always consult a dermatologist for medical advice.

---

## Tech Stack

- Python, NumPy, pandas
- TensorFlow / Keras
- scikit-learn
- Matplotlib, Seaborn
