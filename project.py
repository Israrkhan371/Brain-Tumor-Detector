from utils import extract_features

import os
import h5py
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from skimage.feature import (
    graycomatrix,
    graycoprops,
    hog,
    local_binary_pattern
)

# =========================================================
# CONFIG
# =========================================================
dataset_path = "data"
split_folder = "dataset_split"

train_dir = os.path.join(split_folder, "train")
test_dir = os.path.join(split_folder, "test")

train_csv = "train_files.csv"
test_csv = "test_files.csv"

CACHE_TRAIN_X = "cache_train_X.npy"
CACHE_TRAIN_Y = "cache_train_y.npy"
CACHE_TEST_X = "cache_test_X.npy"
CACHE_TEST_Y = "cache_test_y.npy"

# =========================================================
# CREATE FOLDERS
# =========================================================
os.makedirs(split_folder, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# =========================================================
# TRAIN / TEST SPLIT CACHE CHECK
# =========================================================
if os.path.exists(train_csv) and os.path.exists(test_csv):
    print("===================================")
    print("Using Existing Train/Test Split")
    print("===================================")

    train_files = pd.read_csv(train_csv)["train_files"].tolist()
    test_files = pd.read_csv(test_csv)["test_files"].tolist()

else:
    print("===================================")
    print("Creating New Train/Test Split")
    print("===================================")

    files = [f for f in os.listdir(dataset_path) if f.endswith(".mat")]

    train_files, test_files = train_test_split(
        files, test_size=0.2, random_state=42
    )

    pd.DataFrame(train_files, columns=["train_files"]).to_csv(train_csv, index=False)
    pd.DataFrame(test_files, columns=["test_files"]).to_csv(test_csv, index=False)

    for f in train_files:
        src = os.path.join(dataset_path, f)
        dst = os.path.join(train_dir, f)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    for f in test_files:
        src = os.path.join(dataset_path, f)
        dst = os.path.join(test_dir, f)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    print("Train/Test folders created successfully!")
    print("\nTrain Size:", len(train_files))
    print("Test Size:", len(test_files))

# =========================================================
# LOAD OR EXTRACT TRAIN FEATURES
# =========================================================
if os.path.exists(CACHE_TRAIN_X) and os.path.exists(CACHE_TRAIN_Y):
    print("\nLoading Cached TRAIN Features")

    X_train = np.load(CACHE_TRAIN_X)
    y_train = np.load(CACHE_TRAIN_Y)

else:
    print("\nExtracting TRAIN Features")

    X_train, y_train = [], []

    for f in train_files:
        file_path = os.path.join(dataset_path, f)

        mat = h5py.File(file_path, 'r')
        cjdata = mat['cjdata']

        image = np.array(cjdata['image'])
        label = int(np.array(cjdata['label'])[0][0])

        features = extract_features(image)

        X_train.append(features)
        y_train.append(label - 1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    np.save(CACHE_TRAIN_X, X_train)
    np.save(CACHE_TRAIN_Y, y_train)

    print("TRAIN feature cache saved!")

# =========================================================
# LOAD OR EXTRACT TEST FEATURES
# =========================================================
if os.path.exists(CACHE_TEST_X) and os.path.exists(CACHE_TEST_Y):
    print("\nLoading Cached TEST Features")

    X_test = np.load(CACHE_TEST_X)
    y_test = np.load(CACHE_TEST_Y)

else:
    print("\nExtracting TEST Features")

    X_test, y_test = [], []

    for f in test_files:
        file_path = os.path.join(dataset_path, f)

        mat = h5py.File(file_path, 'r')
        cjdata = mat['cjdata']

        image = np.array(cjdata['image'])
        label = int(np.array(cjdata['label'])[0][0])

        features = extract_features(image)

        X_test.append(features)
        y_test.append(label - 1)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    np.save(CACHE_TEST_X, X_test)
    np.save(CACHE_TEST_Y, y_test)

    print("TEST feature cache saved!")

# =========================================================
# EDA SECTION
# =========================================================

# =========================================================
# IMPORT FOR EDA
# =========================================================

import seaborn as sns

# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Meningioma",
    "Glioma",
    "Pituitary"
]

# =========================================================
# CLASS DISTRIBUTION
# =========================================================

print("\n===================================")
print("EDA : CLASS DISTRIBUTION")
print("===================================")

unique, counts = np.unique(y_train, return_counts=True)

for i in range(len(unique)):

    print(class_names[i], ":", counts[i])

plt.figure(figsize=(6, 4))

plt.bar(class_names, counts)

plt.title("Training Dataset Class Distribution")

plt.xlabel("Tumor Classes")

plt.ylabel("Number of Images")

plt.show()

# =========================================================
# SAMPLE MRI IMAGES
# =========================================================

print("\n===================================")
print("EDA : SAMPLE MRI IMAGES")
print("===================================")

plt.figure(figsize=(12, 4))

shown = [False, False, False]

position = 1

for f in train_files:

    file_path = os.path.join(dataset_path, f)

    mat = h5py.File(file_path, 'r')

    cjdata = mat['cjdata']

    image = np.array(cjdata['image'])

    label = int(np.array(cjdata['label'])[0][0]) - 1

    if not shown[label]:

        plt.subplot(1, 3, position)

        plt.imshow(image, cmap='gray')

        plt.title(class_names[label])

        plt.axis('off')

        shown[label] = True

        position += 1

    if all(shown):
        break

plt.suptitle("Sample MRI Images")

plt.show()

# =========================================================
# PIXEL INTENSITY HISTOGRAM
# =========================================================

print("\n===================================")
print("EDA : PIXEL INTENSITY HISTOGRAM")
print("===================================")

sample_image = image.flatten()

plt.figure(figsize=(6, 4))

plt.hist(sample_image, bins=50)

plt.title("Pixel Intensity Distribution")

plt.xlabel("Pixel Intensity")

plt.ylabel("Frequency")

plt.show()

# =========================================================
# FEATURE DISTRIBUTION
# =========================================================

print("\n===================================")
print("EDA : FEATURE DISTRIBUTION")
print("===================================")

feature_means = X_train.mean(axis=1)

plt.figure(figsize=(6, 4))

plt.hist(feature_means, bins=30)

plt.title("Feature Mean Distribution")

plt.xlabel("Mean Feature Value")

plt.ylabel("Frequency")

plt.show()

# =========================================================
# FEATURE CORRELATION HEATMAP
# =========================================================

print("\n===================================")
print("EDA : FEATURE CORRELATION")
print("===================================")

subset = pd.DataFrame(X_train[:, :20])

plt.figure(figsize=(10, 8))

sns.heatmap(subset.corr())

plt.title("Feature Correlation Heatmap")

plt.show()

# =========================================================
# FEATURE STATISTICS
# =========================================================

print("\n===================================")
print("EDA : FEATURE STATISTICS")
print("===================================")

print("\nMean Feature Value :")
print(np.mean(X_train))

print("\nStandard Deviation :")
print(np.std(X_train))

print("\nMaximum Feature Value :")
print(np.max(X_train))

print("\nMinimum Feature Value :")
print(np.min(X_train))

# =========================================================
# CLASS-WISE SAMPLE COUNTS
# =========================================================

print("\n===================================")
print("EDA : CLASS COUNTS")
print("===================================")

for i in range(len(class_names)):

    count = np.sum(y_train == i)

    print(class_names[i], "=", count)

# =========================================================
# FEATURE SHAPE INFO
# =========================================================

print("\n===================================")
print("EDA : DATASET INFORMATION")
print("===================================")

print("Training Features Shape :", X_train.shape)

print("Testing Features Shape :", X_test.shape)

print("Total Features Extracted :", X_train.shape[1])

print("Training Samples :", X_train.shape[0])

print("Testing Samples :", X_test.shape[0])

# =========================================================
# FEATURE SCALING
# =========================================================
if os.path.exists("scaler.pkl"):
    scaler = joblib.load("scaler.pkl")

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nScaler loaded successfully!")

else:
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, "scaler.pkl")

    print("\nScaler saved successfully!")

# =========================================================
# MODELS
# =========================================================
models = {
    "SVM": SVC(kernel='rbf', C=10, gamma='scale', probability=True),
    "Decision_Tree": DecisionTreeClassifier(max_depth=10),
    "Random_Forest": RandomForestClassifier(n_estimators=300),
    "KNN": KNeighborsClassifier(),
    "Naive_Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        eval_metric='mlogloss'
    ),
    "ANN": MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1000)
}

# =========================================================
# TRAIN / LOAD MODELS
# =========================================================
results = []

for name, model in models.items():

    print("\n===================================")
    print("MODEL:", name)
    print("===================================")

    model_path = f"{name}.pkl"

    if os.path.exists(model_path):
        print("Loading Saved Model...")
        model = joblib.load(model_path)

    else:
        print("Training Model...")
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print("Model Saved!")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # =========================================================
    # CLASSIFICATION REPORT DICTIONARY
    # =========================================================

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    weighted = report["weighted avg"]

    precision = weighted["precision"]

    recall = weighted["recall"]

    f1 = weighted["f1-score"]

    # =========================================================
    # STORE RESULTS
    # =========================================================

    results.append([

        name,

        accuracy,

        precision,

        recall,

        f1
    ])

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

# =========================================================
# SAVE RESULTS
# =========================================================
results_df = pd.DataFrame(results, columns=[

    "Model",

    "Accuracy",

    "Precision",

    "Recall",

    "F1"

])
results_df = results_df.sort_values(by="Accuracy", ascending=False)

results_df.to_csv("results.csv", index=False)

print("\n===================================")
print("FINAL RESULTS")
print("===================================")

print(results_df)
print("\nResults CSV saved successfully!")

# =========================================================
# PLOT RESULTS
# =========================================================
plt.figure(figsize=(10, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.xticks(rotation=20)
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()