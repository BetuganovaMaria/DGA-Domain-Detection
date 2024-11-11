# DGA Domain Detection

---

Model for classification of legitimate and dga domains.

## Dataset

--- 

The dataset for training and testing the model is generated in the program in `creating_dataset.py`.
It splits for train and test folds.

Legitimate domains are taken from this [site](https://majestic.com/reports/majestic-million).

DGA domains are generated using an algorithm in `dga.py`.

It also removes domains that repeat domains in `validation.csv`.

## Data preprocessing

---

Data preprocessing includes tld and duplicates removing and empty field handling.

## Model

---

Uses `tensorflow.keras` for neural network creating: `keras.Sequential()` with `LSTM` layer for neural network.

## Results:

---
Results for test dataset

- **True positive:** 1681
- **True negative:** 109
- **False positive**: 85
- **False negative:** 1892
- **Accuracy:** 0.9485
- **Precision:** 0.9455
- **Recall:** 0.9570
- **F1:** 0.9512
- **AUC:** 0.9894
