%%writefile README.md
# Maternal Health Risk Prediction — Machine Learning Training Script

## Project Overview
This project develops and trains a neural network model to predict maternal health risk based on six key clinical vitals: Age, Systolic Blood Pressure, Diastolic Blood Pressure, Blood Glucose, Body Temperature, and Heart Rate. The goal is to classify patients into one of three risk categories: Low Risk (Class 0), Moderate Risk (Class 1), and High Risk (Class 2). The trained model is then exported as a TensorFlow Lite (`.tflite`) file for seamless integration into an Android application, enabling offline risk assessment.

### Author & Version
*   **Author**: Dr. Richard Kimera
*   **Institution**: Mbarara University of Science and Technology (MUST)
*   **Course**: MSc Health Informatics — HIT 7102
*   **Version**: 1.0 (2026)

## How to Run This Script

### Prerequisites
*   Python 3.8 or newer.

### Installation
Install the required Python libraries using pip:
```bash
pip install tensorflow scikit-learn matplotlib seaborn pandas numpy
```

### Execution
Run the script from your terminal (assuming you save it as `maternal_health_train.py`):
```bash
python maternal_health_train.py
```

## Dataset
The model is trained on a synthetic dataset generated based on the statistical properties of the UCI Maternal Health Risk dataset (Ahmed et al., 2020). This approach ensures: 
*   No external data files are needed.
*   A balanced class distribution across all risk levels.
*   Full transparency and auditability of the data generation process.

## Model Architecture
The neural network is a Feedforward Neural Network (Multi-Layer Perceptron) with the following structure:
*   **Input Layer**: 6 features (vitals).
*   **Hidden Layer 1**: 32 neurons, ReLU activation, Batch Normalization, Dropout (0.3).
*   **Hidden Layer 2**: 16 neurons, ReLU activation, Dropout (0.2).
*   **Output Layer**: 3 neurons, Softmax activation (for probability distribution over the three risk classes).

## Output Files
Upon successful execution, the script generates the following files and directories in the same folder:
*   `maternal_health.tflite`: The TensorFlow Lite model, ready for deployment in the Android application.
*   `scaler_params.json`: JSON file containing the mean and standard deviation values used for feature normalization. These parameters are crucial for preprocessing new patient data in the Android app.
*   `plots/`: A directory containing various diagnostic and evaluation plots, including:
    *   `01_feature_distributions.png`: Violin plots showing feature distributions by risk class.
    *   `02_correlation_heatmap.png`: Feature correlation matrix.
    *   `03_class_balance.png`: Class distribution bar chart.
    *   `04_training_history.png`: Loss and accuracy curves over training epochs.
    *   `05_confusion_matrix.png`: Confusion matrix illustrating correct and incorrect predictions.
    *   `06_roc_curves.png`: Receiver Operating Characteristic (ROC) curves and AUC scores for each class.
    *   `07_confidence_distribution.png`: Model prediction confidence analysis.
    *   `08_feature_importance.png`: Permutation-based feature importance scores.
    *   `09_probability_calibration.png`: Per-class probability calibration plots.

## Model Performance Highlights
*   **Test Accuracy**: 100.0% (on synthetic data)
*   **Key Metric**: Recall for 'High Risk' is crucial in clinical settings to avoid missed diagnoses. The model achieved 100% recall for all classes on the synthetic test set.

## Glossary for Beginners
*   **Neural Network**: A mathematical model inspired by the brain, learning patterns from data.
*   **Epoch**: One complete pass through the entire training dataset.
*   **Loss**: A measure of how inaccurate the model's predictions are (lower is better).
*   **Accuracy**: The proportion of correct predictions (higher is better).
*   **Normalisation**: Rescaling numerical features to a standard range to improve model training.
*   **TFLite**: A lightweight format for machine learning models, optimized for mobile and edge devices.
