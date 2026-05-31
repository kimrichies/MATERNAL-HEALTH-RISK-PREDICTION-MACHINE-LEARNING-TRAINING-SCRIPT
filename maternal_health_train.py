#!/usr/bin/env python3
"""
================================================================================
  MATERNAL HEALTH RISK PREDICTION — MACHINE LEARNING TRAINING SCRIPT
  Mbarara University of Science and Technology (MUST)
  MSc Health Informatics — HIN 7102
  Author  : Dr. Richard Kimera
  Version : 1.0  |  2026
================================================================================

WHAT THIS SCRIPT DOES
─────────────────────
This script trains a neural network to predict maternal health risk from six
clinical vitals — Age, Systolic Blood Pressure, Diastolic Blood Pressure,
Blood Glucose, Body Temperature, and Heart Rate — and classifies each patient
into one of three risk categories:
    • Class 0 — Low Risk
    • Class 1 — Moderate Risk
    • Class 2 — High Risk

The trained model is exported as a TensorFlow Lite (.tflite) file for embedding
directly inside the Android application without any internet connection.

HOW TO RUN THIS SCRIPT
──────────────────────
1.  Make sure you have Python 3.8 or newer installed.
        Check:  python --version

2.  Install the required libraries (one-time setup):
        pip install tensorflow scikit-learn matplotlib seaborn pandas numpy

3.  Run the script from your terminal:
        python maternal_health_train.py

4.  When it finishes you will find these new files in the same folder:
        maternal_health.tflite      ← the model for Android
        scaler_params.json          ← normalisation values for Android
        plots/                      ← folder containing all graphs

READING THE OUTPUT
──────────────────
• "Epoch X/60" lines — each epoch is one full pass through the training data.
  Watch the "accuracy" number climb toward 1.0 and the "loss" fall toward 0.
• "Test accuracy" — how well the model performs on data it has never seen.
  Anything above 0.85 (85%) is good for this type of clinical screening tool.
• The confusion matrix plot shows where the model makes mistakes.
• The ROC curves show the trade-off between true positives and false positives.

GLOSSARY FOR BEGINNERS
──────────────────────
• Neural Network  : A mathematical model loosely inspired by the brain, made of
                    layers of numbers that learn patterns from examples.
• Epoch           : One complete training cycle through all training examples.
• Loss            : A number measuring how wrong the model's predictions are.
                    Lower loss = better model.
• Accuracy        : Proportion of correct predictions. 1.0 = 100% correct.
• Normalisation   : Rescaling numbers so they are all on a similar scale,
                    preventing large numbers (like blood pressure ~120) from
                    drowning out small ones (like temperature ~37).
• TFLite          : A compressed version of the model designed to run on a
                    smartphone without needing a powerful computer or internet.
================================================================================
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 0 — IMPORT LIBRARIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Think of libraries as toolboxes. Before you can use a hammer, you pick it up.
# Each import below picks up a specific toolbox:

import os            # Interact with the file system (create folders, paths)
import json          # Save data in JSON format (human-readable text files)
import warnings      # Suppress unimportant warning messages
warnings.filterwarnings("ignore")         # Keep the terminal output clean
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Hide TensorFlow's internal messages

import numpy as np                        # The core maths library — arrays, stats
import pandas as pd                       # Table-like data manipulation
import matplotlib.pyplot as plt           # Plotting and graph drawing
import matplotlib.gridspec as gridspec    # Advanced subplot layout control
import seaborn as sns                     # High-level, beautiful statistical plots

from sklearn.model_selection import train_test_split   # Split data into train/test
from sklearn.preprocessing import StandardScaler       # Normalise features to mean=0
from sklearn.metrics import (
    classification_report,     # Per-class precision, recall, F1-score table
    confusion_matrix,          # Count of correct vs incorrect predictions
    roc_curve,                 # Receiver Operating Characteristic curve data
    auc,                       # Area Under the Curve — single-number model quality
    ConfusionMatrixDisplay,    # Ready-made confusion matrix visualiser
)
from sklearn.utils.class_weight import compute_class_weight  # Handle imbalanced data

import tensorflow as tf                   # The machine learning framework
from tensorflow import keras              # High-level neural network building API

# ── Reproducibility ─────────────────────────────────────────────────────────
# Setting seeds means that every time you run this script you get the same
# random numbers — and therefore the same results. This is essential for
# scientific reproducibility. Without this, two runs could give different
# accuracy values even with identical code and data.
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 72)
print("  MATERNAL HEALTH RISK PREDICTION — MODEL TRAINING")
print("  Mbarara University of Science and Technology | MSc Health Informatics")
print("=" * 72)
print(f"\n  TensorFlow version : {tf.__version__}")
print(f"  NumPy version      : {np.__version__}")
print(f"  Random seed        : {SEED}  (ensures reproducible results)")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — CREATE THE OUTPUT FOLDER FOR ALL PLOTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)   # Create the folder; don't crash if it exists
print(f"[STEP 1]  Output folder ready → '{PLOTS_DIR}/'")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 — GENERATE THE DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# WHY SYNTHETIC DATA?
# The UCI Maternal Health Risk dataset (Ahmed et al., 2020) contains 1,014
# real patient records from IoT sensors in rural Bangladeshi hospitals.
# We generate synthetic data based on its statistical properties so that:
#  1. Student laptops don't need a separate data file to run this script.
#  2. The class distribution is balanced (equal numbers per risk level),
#     preventing the model from ignoring the rare "high risk" class.
#  3. The generation process is fully transparent and auditable.
#
# HOW SYNTHETIC GENERATION WORKS:
# For each risk class we define realistic clinical ranges, then sample random
# values from those ranges using numpy's random number generators.
# np.random.randint(a, b) → a random whole number between a and b
# np.random.uniform(a, b) → a random decimal number between a and b

print("\n[STEP 2]  Generating synthetic maternal health dataset...")
print("          (Based on UCI Maternal Health Risk Dataset distributions)")
### Access the dataset here: https://archive.ics.uci.edu/dataset/863/maternal+health+risk

N_PER_CLASS = 400   # Number of patient records to generate per risk class
                    # Total dataset = 400 × 3 classes = 1,200 records

rows = []   # We will collect each patient's data as a list and append here

for _ in range(N_PER_CLASS):
    # ── Low Risk (Class 0) ────────────────────────────────────────────────────
    # Younger mothers, normal BP, healthy glucose, normal temperature, calm HR.
    # These ranges reflect WHO guidelines for uncomplicated pregnancy.
    rows.append([
        np.random.randint(18, 35),          # Age: 18–34 years
        np.random.randint(90, 120),         # Systolic BP: 90–119 mmHg (normal)
        np.random.randint(60, 80),          # Diastolic BP: 60–79 mmHg (normal)
        round(np.random.uniform(6.0, 7.5), 1),   # Blood Glucose: 6–7.5 mmol/L
        round(np.random.uniform(36.5, 37.5), 1), # Body Temp: 36.5–37.5 °C
        np.random.randint(60, 80),          # Heart Rate: 60–79 bpm
        0                                   # LABEL: 0 = Low Risk
    ])

for _ in range(N_PER_CLASS):
    # ── Moderate Risk (Class 1) ───────────────────────────────────────────────
    # Older mothers, pre-hypertensive BP, elevated glucose (gestational diabetes
    # territory), slightly elevated temperature, faster heart rate.
    rows.append([
        np.random.randint(28, 45),
        np.random.randint(120, 140),        # Systolic: 120–139 mmHg (elevated)
        np.random.randint(80, 90),          # Diastolic: 80–89 mmHg (elevated)
        round(np.random.uniform(7.5, 10.0), 1),
        round(np.random.uniform(37.5, 38.5), 1),
        np.random.randint(75, 95),
        1                                   # LABEL: 1 = Moderate Risk
    ])

for _ in range(N_PER_CLASS):
    # ── High Risk (Class 2) ────────────────────────────────────────────────────
    # Older mothers, hypertensive crisis range (preeclampsia territory),
    # severely elevated glucose (diabetic range), fever, tachycardia.
    rows.append([
        np.random.randint(35, 65),
        np.random.randint(140, 180),        # Systolic ≥ 140 = hypertensive crisis
        np.random.randint(90, 110),         # Diastolic ≥ 90
        round(np.random.uniform(10.0, 15.0), 1),
        round(np.random.uniform(38.5, 40.0), 1),
        np.random.randint(90, 120),
        2                                   # LABEL: 2 = High Risk
    ])

# Column names — these must match exactly what the Android app expects
FEATURE_NAMES = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BloodGlucose",
    "BodyTemp",
    "HeartRate",
]
TARGET_NAME = "RiskLevel"
CLASS_NAMES = ["Low Risk", "Moderate Risk", "High Risk"]

# Convert the list of rows into a DataFrame (think of it as a spreadsheet in Python)
df = pd.DataFrame(rows, columns=FEATURE_NAMES + [TARGET_NAME])

# Shuffle the rows so Low/Moderate/High records are not grouped together.
# random_state=SEED keeps the shuffle reproducible.
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"\n          Dataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"          Features      : {FEATURE_NAMES}")
print(f"          Target        : {TARGET_NAME}")
print("\n          Class distribution (should be perfectly balanced):")
for cls, count in df[TARGET_NAME].value_counts().sort_index().items():
    print(f"            Class {cls} — {CLASS_NAMES[cls]:<16} : {count} records")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 — EXPLORATORY DATA ANALYSIS (EDA)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Before training any model, a good data scientist always looks at the data.
# EDA helps us spot anomalies, understand distributions, and see which features
# separate the risk classes most clearly.

print("\n[STEP 3]  Exploratory Data Analysis...")
print("\n          Basic statistics per feature:")
print(df[FEATURE_NAMES].describe().round(2).to_string())

# ── PLOT 1: Feature Distributions by Risk Class ───────────────────────────────
# A violin plot combines a box plot (showing median and quartiles) with a kernel
# density estimate (showing the shape of the distribution). Wider sections mean
# more data points at that value.
print("\n          Saving Plot 1: Feature distributions by risk class...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Feature Distributions by Maternal Risk Class\n"
    "Mbarara University of Science and Technology — MSc Health Informatics",
    fontsize=14, fontweight="bold", y=1.01
)

palette = {0: "#2E7D32", 1: "#F57C00", 2: "#C62828"}   # Green / Amber / Red
label_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
df["RiskLabel"] = df[TARGET_NAME].map(label_map)
palette_labels = {"Low Risk": "#2E7D32", "Moderate Risk": "#F57C00", "High Risk": "#C62828"}

for ax, feature in zip(axes.flat, FEATURE_NAMES):
    sns.violinplot(
        data=df,
        x="RiskLabel",
        y=feature,
        palette=palette_labels,
        order=["Low Risk", "Moderate Risk", "High Risk"],
        ax=ax,
        inner="box",      # Show a mini box-plot inside the violin
        linewidth=1.2,
    )
    ax.set_title(feature, fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(feature)
    ax.tick_params(axis="x", labelsize=9)
    # Draw a light horizontal grid to make values easier to read
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"          → Saved: {PLOTS_DIR}/01_feature_distributions.png")

# ── PLOT 2: Correlation Heatmap ───────────────────────────────────────────────
# A correlation matrix shows how strongly each pair of features moves together.
# Values close to +1.0 = strong positive correlation (both go up together).
# Values close to -1.0 = strong negative correlation (one goes up, other goes down).
# Values near 0 = no linear relationship.
# This helps us check: are any features redundant (correlated with each other)?
print("          Saving Plot 2: Feature correlation heatmap...")

fig, ax = plt.subplots(figsize=(9, 7))
corr_matrix = df[FEATURE_NAMES + [TARGET_NAME]].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Show only lower triangle

sns.heatmap(
    corr_matrix,
    annot=True,          # Print the correlation number inside each cell
    fmt=".2f",           # Format to 2 decimal places
    cmap="RdYlGn",       # Red = negative, Green = positive
    center=0,
    square=True,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"shrink": 0.8},
    mask=mask,
)
ax.set_title(
    "Feature Correlation Matrix\n(Values near ±1.0 indicate strong linear relationship)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"          → Saved: {PLOTS_DIR}/02_correlation_heatmap.png")

# ── PLOT 3: Class Balance Bar Chart ───────────────────────────────────────────
print("          Saving Plot 3: Class balance chart...")

fig, ax = plt.subplots(figsize=(7, 5))
counts = df[TARGET_NAME].value_counts().sort_index()
bars = ax.bar(
    [CLASS_NAMES[i] for i in counts.index],
    counts.values,
    color=["#2E7D32", "#F57C00", "#C62828"],
    edgecolor="white", linewidth=1.5, width=0.55
)
# Label each bar with its count
for bar, count in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 8,
        str(count),
        ha="center", va="bottom", fontsize=12, fontweight="bold"
    )
ax.set_title("Class Distribution — Maternal Risk Labels\n(Balanced dataset: equal records per class)", fontweight="bold")
ax.set_ylabel("Number of Patient Records")
ax.set_ylim(0, max(counts.values) * 1.2)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_class_balance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"          → Saved: {PLOTS_DIR}/03_class_balance.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4 — PREPARE DATA FOR TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# We split data into:
#   X  (features / inputs)  — the clinical measurements the model reads
#   y  (target / label)     — the risk level the model tries to predict

print("\n[STEP 4]  Preparing data for training...")

X = df[FEATURE_NAMES].values.astype(np.float32)   # Shape: (1200, 6)
y = df[TARGET_NAME].values                         # Shape: (1200,) — integers 0/1/2

# ── Train / Validation / Test Split ───────────────────────────────────────────
#
# We divide the data into three independent sets:
#
#   Training set (70%) — the model LEARNS from this data.
#   Validation set (10%) — used DURING training to monitor overfitting.
#   Test set (20%) — used ONLY ONCE at the very end to measure final performance.
#                    The model never sees this data during training.
#
# The key rule: NEVER use the test set for any decision during training.
# Using it would give you an over-optimistic accuracy estimate.
#
# stratify=y ensures that each split has the same class proportions.

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,        # 20% for final testing
    random_state=SEED,
    stratify=y             # Keep class proportions equal in each split
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.125,       # 0.125 × 80% = 10% of total dataset
    random_state=SEED,
    stratify=y_temp
)

print(f"\n          Data splits:")
print(f"            Training   : {X_train.shape[0]:>4} records ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"            Validation : {X_val.shape[0]:>4} records ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"            Test       : {X_test.shape[0]:>4} records ({X_test.shape[0]/len(X)*100:.0f}%)")

# ── Feature Normalisation (StandardScaler) ─────────────────────────────────────
#
# WHY NORMALISE?
# Our six features have very different numeric ranges:
#   Age ≈ 18–65, SystolicBP ≈ 90–180, BloodGlucose ≈ 6–15, BodyTemp ≈ 36–40
#
# Without normalisation, the neural network's weight updates would be dominated
# by features with large absolute values (like BP), and features with small
# ranges (like temperature) would barely influence the model.
#
# StandardScaler transforms each feature to have:
#   mean (μ) = 0      (centre the values around zero)
#   std dev (σ) = 1   (all features have the same spread)
#
# Formula applied to every value x: z = (x − μ) / σ
#
# CRITICAL RULE: Fit the scaler ONLY on the training data.
# Then apply the SAME scaler to validation and test data.
# Fitting on all data would "leak" test information into training — data leakage.

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)    # Compute μ and σ, then scale
X_val_s   = scaler.transform(X_val)          # Use training μ and σ — no refit
X_test_s  = scaler.transform(X_test)         # Same

print(f"\n          StandardScaler fitted on training data:")
for i, fname in enumerate(FEATURE_NAMES):
    print(f"            {fname:<15} mean={scaler.mean_[i]:>8.3f}   std={scaler.scale_[i]:>7.3f}")

# ── Save scaler parameters ─────────────────────────────────────────────────────
# The Android app must apply exactly the same normalisation to new inputs.
# We save μ and σ for each feature so they can be hard-coded in Java.
scaler_params = {
    "mean":          scaler.mean_.tolist(),
    "scale":         scaler.scale_.tolist(),
    "feature_names": FEATURE_NAMES,
    "note":          "Apply: z = (x - mean[i]) / scale[i] for each feature i"
}
with open("scaler_params.json", "w") as f:
    json.dump(scaler_params, f, indent=2)
print(f"\n          Scaler parameters saved → scaler_params.json")

# ── Class Weights ──────────────────────────────────────────────────────────────
# Even with a balanced dataset, it is good practice to compute class weights.
# In a real clinical dataset, high-risk cases are rare, so we penalise the model
# more heavily for missing them. The compute_class_weight function does this:
# weight[c] = total_samples / (n_classes × samples_in_class_c)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights_array)}
print(f"\n          Class weights for training: {class_weight_dict}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5 — BUILD THE NEURAL NETWORK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ARCHITECTURE OVERVIEW:
#
#   Input Layer   → 6 features (one neuron per feature)
#       ↓
#   Hidden Layer 1 → 32 neurons + ReLU activation + Batch Normalisation + Dropout
#       ↓
#   Hidden Layer 2 → 16 neurons + ReLU activation
#       ↓
#   Output Layer  → 3 neurons + Softmax activation (one per risk class)
#
# KEY CONCEPTS:
#
# Dense layer:
#   Every input is connected to every neuron. Each connection has a "weight"
#   (a number). During training the weights are adjusted to reduce errors.
#
# ReLU activation (Rectified Linear Unit):
#   f(x) = max(0, x)  — turns negative values into 0, keeps positive values.
#   Without activation functions, a neural network is just linear regression.
#   Activation functions let the network learn non-linear, curved boundaries.
#
# Batch Normalisation:
#   Normalises the outputs of a layer so the next layer receives stable input.
#   This makes training faster and more stable, especially in early epochs.
#
# Dropout (rate=0.3):
#   During each training step, randomly switches off 30% of neurons.
#   This sounds destructive — but it prevents overfitting.
#   Overfitting = the model memorises the training data instead of learning
#   general patterns. With dropout the network must learn redundant
#   representations, making it more robust on new, unseen data.
#
# Softmax output:
#   Converts the raw output of 3 neurons into probabilities that sum to 1.
#   E.g., [0.05, 0.15, 0.80] means: 5% Low, 15% Moderate, 80% High Risk.
#   The predicted class is whichever has the highest probability.

print("\n[STEP 5]  Building the neural network architecture...")

model = keras.Sequential(
    [
        # ── Input layer ──────────────────────────────────────────────────────
        # shape=(6,) tells Keras to expect 6 input features per patient.
        keras.layers.Input(shape=(6,), name="vitals_input"),

        # ── First hidden layer ────────────────────────────────────────────────
        keras.layers.Dense(
            32,                     # 32 neurons (units)
            activation="relu",      # ReLU activation function
            name="hidden_layer_1",
            kernel_regularizer=keras.regularizers.l2(0.001),  # L2: penalise large weights
        ),
        keras.layers.BatchNormalization(name="batch_norm_1"),
        keras.layers.Dropout(0.3, name="dropout_1"),

        # ── Second hidden layer ───────────────────────────────────────────────
        keras.layers.Dense(
            16,
            activation="relu",
            name="hidden_layer_2",
            kernel_regularizer=keras.regularizers.l2(0.001),
        ),
        keras.layers.Dropout(0.2, name="dropout_2"),

        # ── Output layer ─────────────────────────────────────────────────────
        # 3 neurons = 3 classes. Softmax converts outputs to probabilities.
        keras.layers.Dense(3, activation="softmax", name="risk_output"),
    ],
    name="MaternalRiskMLP",
)

# ── Compile the model ──────────────────────────────────────────────────────
# Compilation links the model to its:
#   optimizer  : the algorithm that updates weights (Adam is the most popular)
#   loss       : the function that measures how wrong predictions are
#   metrics    : what to display during training (accuracy)
#
# Adam optimizer: short for Adaptive Moment Estimation. It adjusts the learning
# rate for each weight individually based on how the gradient has been changing.
# It is the default choice for most classification tasks.
#
# Sparse categorical crossentropy: the standard loss function for multi-class
# classification when labels are integers (0, 1, 2) rather than one-hot vectors.

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Print a summary table: every layer, its output shape, and parameter count
print()
model.summary()

total_params = model.count_params()
print(f"\n          Total trainable parameters: {total_params:,}")
print(f"          (Each parameter is one number the model adjusts during training)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6 — TRAIN THE MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Training Callbacks — these are automatic actions triggered during training:
#
# EarlyStopping:
#   Monitors the validation loss. If it stops improving for 12 consecutive
#   epochs, training stops automatically. This prevents overfitting.
#   restore_best_weights=True means we restore the best version of the model.
#
# ReduceLROnPlateau:
#   If the validation loss stops improving for 6 epochs, the learning rate
#   is halved. A smaller learning rate makes the model take smaller steps,
#   which can help it fine-tune when it is close to the optimal solution.

print("\n[STEP 6]  Training the model...")
print("          Reading the training log:")
print("          loss     = how wrong predictions are (lower = better)")
print("          accuracy = fraction of correct predictions (higher = better)")
print("          val_*    = same metrics on the validation set\n")

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=12,               # Wait 12 epochs before stopping
    restore_best_weights=True, # Keep the best model, not the last one
    verbose=1,
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,      # Multiply learning rate by 0.5
    patience=6,
    min_lr=1e-6,     # Never let learning rate go below this
    verbose=1,
)

history = model.fit(
    X_train_s, y_train,
    epochs=80,              # Maximum number of training cycles
    batch_size=32,          # Process 32 patients at a time before updating weights
    validation_data=(X_val_s, y_val),
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1,              # Print one line per epoch
)

epochs_run = len(history.history["loss"])
best_val_acc = max(history.history["val_accuracy"])
print(f"\n          Training completed in {epochs_run} epochs")
print(f"          Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 7 — EVALUATE ON TEST SET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The test set has never been seen by the model during training or validation.
# This is our true measure of how well the model generalises to new patients.

print("\n[STEP 7]  Evaluating on test set (unseen data)...")

test_loss, test_accuracy = model.evaluate(X_test_s, y_test, verbose=0)
print(f"\n          Test Loss      : {test_loss:.4f}")
print(f"          Test Accuracy  : {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")

if test_accuracy >= 0.85:
    print(f"\n          ✓ Model performance is GOOD (≥ 85% for clinical screening)")
elif test_accuracy >= 0.75:
    print(f"\n          ~ Model performance is FAIR (≥ 75% — acceptable for research)")
else:
    print(f"\n          ✗ Model performance is POOR (< 75% — consider retraining)")

# ── Generate predictions ───────────────────────────────────────────────────────
y_pred_probs = model.predict(X_test_s, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# ── Classification Report ──────────────────────────────────────────────────────
print("\n          Classification Report (per-class performance):")
print("          " + "="*60)
report = classification_report(
    y_test, y_pred,
    target_names=CLASS_NAMES,
    digits=4
)
for line in report.split('\n'):
    print(f"          {line}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 8 — SAVE VISUALIZATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n[STEP 8]  Generating evaluation plots...")

# ── PLOT 4: Training History ──────────────────────────────────────────────────
print("          Saving Plot 4: Training history (loss and accuracy)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axes[0].plot(history.history["loss"], label="Training Loss", linewidth=2)
axes[0].plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
axes[0].set_xlabel("Epoch", fontsize=11)
axes[0].set_ylabel("Loss (lower is better)", fontsize=11)
axes[0].set_title("Model Loss Over Training", fontweight="bold")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy plot
axes[1].plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("Accuracy (higher is better)", fontsize=11)
axes[1].set_title("Model Accuracy Over Training", fontweight="bold")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_training_history.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"          → Saved: {PLOTS_DIR}/04_training_history.png")

# ── PLOT 5: Confusion Matrix ───────────────────────────────────────────────────
print("          Saving Plot 5: Confusion matrix...")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix on Test Set", fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"          → Saved: {PLOTS_DIR}/05_confusion_matrix.png")

# ── PLOT 6: ROC Curves (One-vs-Rest) ───────────────────────────────────────────
print("          Saving Plot 6: ROC curves (One-vs-Rest)...")

fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#2E7D32", "#F57C00", "#C62828"]

for i, (label, color) in enumerate(zip(CLASS_NAMES, colors)):
    # Create binary classification: class i vs. all others
    y_test_binary = (y_test == i).astype(int)
    y_score = y_pred_probs[:, i]
    
    fpr, tpr, _ = roc_curve(y_test_binary, y_score)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})", 
            color=color, linewidth=2.5)

# Diagonal reference line (random classifier)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random Classifier (AUC = 0.5)")

ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves (One-vs-Rest) — Test Set", fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"          → Saved: {PLOTS_DIR}/06_roc_curves.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 9 — EXPORT MODEL TO TFLITE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# TensorFlow Lite (.tflite) is a compressed model format optimised for
# mobile and embedded devices. It supports:
#   - Quantisation (smaller model, faster inference)
#   - Edge device execution (no internet needed)
#   - Java/C++ integration with Android/iOS

print("\n[STEP 9]  Exporting model to TensorFlow Lite format...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
]

tflite_model = converter.convert()

with open("maternal_health.tflite", "wb") as f:
    f.write(tflite_model)

tflite_size = len(tflite_model) / 1024  # Convert to KB
print(f"\n          Model exported → maternal_health.tflite")
print(f"          Model size     : {tflite_size:.1f} KB (very small for mobile)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 10 — FINAL SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "="*72)
print("  TRAINING COMPLETE")
print("="*72)

print("\n✓ All output files saved:")
print(f"  • maternal_health.tflite     ← Model for Android app")
print(f"  • scaler_params.json         ← Feature normalisation for Android")
print(f"  • plots/                     ← All 6 analysis plots:")
print(f"      01_feature_distributions.png")
print(f"      02_correlation_heatmap.png")
print(f"      03_class_balance.png")
print(f"      04_training_history.png")
print(f"      05_confusion_matrix.png")
print(f"      06_roc_curves.png")

print(f"\n✓ Model Performance Summary:")
print(f"  • Test Accuracy     : {test_accuracy*100:.1f}%")
print(f"  • Training Epochs   : {epochs_run}")
print(f"  • Total Parameters  : {total_params:,}")
print(f"  • Model Size (LITE) : {tflite_size:.1f} KB")

print(f"\n✓ Next Steps:")
print(f"  1. Download maternal_health.tflite to your Android Studio project")
print(f"  2. Download scaler_params.json for feature normalisation")
print(f"  3. Integrate the model into your Android app's TensorFlow Lite runtime")
print(f"  4. Normalise input features using the scaler_params before prediction")

print("\n" + "="*72)
print("  Thank you for using this training script!")
print("  Mbarara University of Science and Technology — MSc Health Informatics")
print("="*72 + "\n")
