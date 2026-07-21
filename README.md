# Katniss Transit Search

Deep learning pipeline for exoplanet detection in Kepler space telescope data using a hybrid CNN architecture with auxiliary stellar parameters.

## Overview

This project implements a neural network-based approach to identify exoplanet candidates from Kepler light curves. The model combines global and local views of folded transit signals with auxiliary stellar parameters to classify transiting signals as either confirmed planets or false positives.

## Architecture

The model (**KatnissNet**) uses a dual-branch CNN architecture:

- **Global Branch:** Processes the full folded light curve (2001 points) to capture overall transit shape and baseline characteristics.
- **Local Branch:** Processes a zoomed-in view (1001 points) focused on the transit window to extract fine-grained features.
- **Auxiliary Branch:** Incorporates 8 stellar and transit parameters including SDE, period, odd-even mismatch, SNR, impact parameter, depth, and planetary radius.

The extracted features are concatenated and passed through fully connected layers to produce a binary classification output.

---

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- SciPy
- Lightkurve
- Transit Least Squares (`transitleastsquares`)
- Scikit-learn

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/isabellylimals/katniss-transit-search.git
cd katniss-transit-search
```

### 2. Install dependencies

```bash
pip install torch numpy pandas scipy lightkurve transitleastsquares scikit-learn
```

---

## Data Processing Pipeline

The data pipeline processes Kepler light curves through the following stages:

1. **Download**
   - Retrieves light curves from the MAST archive using **Lightkurve**.

2. **Preprocessing**
   - Removes outliers.
   - Normalizes flux.
   - Bins observations.
   - Cleans missing values.

3. **Transit Search**
   - Uses the **Transit Least Squares (TLS)** algorithm to detect periodic transit signals.

4. **Feature Extraction**
   - Generates:
     - Global folded transit views.
     - Local zoomed transit views.

5. **Parameter Collection**
   - Extracts auxiliary stellar and planetary parameters from the KOI catalog.

---

## Training

Run the training script:

```bash
python src/training/train.py
```

### Training Features

- Focal Loss for class imbalance handling.
- Dropout regularization (0.3).
- Learning rate scheduling using `ReduceLROnPlateau`.
- Early stopping (patience = 12 epochs).
- Automatic saving of the top 3 best-performing models.

---

## Evaluation

Evaluate a trained model with:

```bash
python src/training/test.py
```

The evaluation script:

- Loads the test dataset.
- Computes the AUC-ROC score.
- Finds the optimal classification threshold using **Youden's J statistic**.
- Generates a classification report.
- Saves misclassified false positives for further analysis.

---

## Model Performance

The model is evaluated using confirmed Kepler planets and false positives.

Reported metrics include:

- AUC-ROC Score
- Precision
- Recall
- F1-Score
- Optimal classification threshold

---

## Key Features

- Robust preprocessing pipeline for noisy astronomical data.
- Automatic handling of missing values and outliers.
- Smart caching system to avoid redundant downloads.
- Hybrid dual-branch CNN architecture.
- Auxiliary stellar parameter integration.
- Focal Loss with alpha balancing and label smoothing.
- Automatic threshold optimization for deployment.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{katniss_transit_search,
  author = {Isabelly Lima},
  title = {Katniss Transit Search: Deep Learning for Exoplanet Detection},
  year = {2024},
  url = {https://github.com/isabellylimals/katniss-transit-search}
}
```

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Author

**Isabelly Lima**

GitHub: **@isabellylimals**

---

## Acknowledgments

- NASA Kepler Mission for providing the observational data.
- The **Lightkurve** development team.
- The developers of the **Transit Least Squares (TLS)** algorithm.
