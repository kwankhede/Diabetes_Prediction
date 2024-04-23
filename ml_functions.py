# ml_functions.py

# Import necessary libraries
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib


# Function to split data into training, validation, and test sets
# Function to split data into training and test sets
def split_data(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=1
    )
    print("Training Dataset:", X_train.shape, y_train.shape)
    print("Testing Dataset:", X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test


# Function to train models and evaluate their performance
def train_and_evaluate_models(classifiers_dict, X_train, X_test, y_train, y_test):
    results = pd.DataFrame(
        columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC", "Time"]
    )
    for name, classifier in classifiers_dict.items():
        start = time.time()
        clf = classifier.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        end = time.time()
        roc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        t = end - start
        model_results = pd.DataFrame(
            [[name, acc, prec, rec, f1, roc, t]],
            columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC", "Time"],
        )
        results = results.append(model_results, ignore_index=True)
    print(results)
    return results


# Function to plot ROC curves for all classifiers
def plot_roc_curves(classifiers_dict, X_test, y_test):
    plt.figure(figsize=(15, 10))
    for name, classifier in classifiers_dict.items():
        fit = classifier.fit(X_, y_train)
        y_pred = classifier.predict_proba(X_test)[:, 1]
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        fpr = false_positive_rate
        tpr = true_positive_rate
        plt.plot(fpr, tpr, lw=2, label=name)

    plt.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve of all classifiers")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.show()


# Function to save the trained model
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")


# Function to load the pre-trained model
def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model
