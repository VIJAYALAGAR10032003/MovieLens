"""Evaluation helpers — RMSE, MAE, Accuracy, P/R/F1."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from app.config import LIKE_THRESHOLD, ACC_TOLERANCE


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def regression_accuracy(y_true, y_pred, tol=ACC_TOLERANCE):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tol))


def classification_metrics(y_true, y_pred, threshold=LIKE_THRESHOLD):
    y_true_bin = (np.asarray(y_true) >= threshold).astype(int)
    y_pred_bin = (np.asarray(y_pred) >= threshold).astype(int)
    return {
        "accuracy":  float(accuracy_score(y_true_bin, y_pred_bin)),
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall":    float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1":        float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
    }