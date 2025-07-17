💻 Failure Prediction Prototype (Time-Series based)
This is a simple Python-based prototype that predicts equipment/system failure based on sensor readings over time. The goal is to simulate how failures might be detected early using time-series data and a basic machine learning model.

This prototype is part of an AI/ML assignment focused on early failure prediction using minimal data and logic.

📊 Problem Statement
Predict if a system is going to fail or operate normally, based on continuous sensor readings captured over time.

🔧 Technologies & Libraries Used
| Tool         | Purpose                          |
| ------------ | -------------------------------- |
| Python       | Core programming language        |
| NumPy        | Data generation and manipulation |
| Pandas       | Data handling and CSV operations |
| Matplotlib   | Visualization of sensor data     |
| Scikit-learn | ML model building and evaluation |

🛠️ How It Works
➤ Step 1: Dataset Creation
A dummy dataset is generated using NumPy.

It contains 100 time steps (timestamp 0 to 99).

A single sensor reading per timestamp.

Around t=80 to t=90, readings begin to drift (simulate sensor anomaly).

Failure is labeled from t=90 onward (failure = 1), else failure = 0.

➤ Step 2: Data Exploration
Loaded using Pandas.

Basic statistics and null checks performed.

A time-series plot visualizes sensor behavior before failure.

➤ Step 3: Model Training
Only sensor_reading is used as input.

A simple Decision Tree Classifier is trained.

The dataset is split into training (80%) and test (20%) sets.

Accuracy and classification metrics are reported

📈 Results
Model Accuracy: ~80%

Strengths: The model correctly classifies most normal (non-failure) conditions.

Limitation: The model fails to predict the rare failure class due to class imbalance (only 1 failure case in test set).

📉 Limitations & Improvements
The dataset is very small and highly imbalanced (only 10 failure samples out of 100).

To improve:

Add more failure samples

Use oversampling (e.g., SMOTE) or synthetic data generation

Try more advanced models like Random Forest or LSTM (for actual time-dependence)

📂 Files Included
| File                    | Description                                |
| ----------------------- | ------------------------------------------ |
| `failure_prediction.py` | Main script: dataset generation + training |
| `dummy_timeseries.csv`  | Generated dataset of 100 time steps        |
| `README.md`             | This project explanation                   |

✅ How to Run
Install dependencies:
pip install pandas numpy matplotlib scikit-learn

Run the Python script:
python failure_prediction.py

👤 Author
Kundan Yadav

Prototype developed as part of an ML-based risk/failure prediction task

Guided by personal learning effort and project understanding


