import pandas as pd
import numpy as np

# for reproducibility
np.random.seed(42)

# generate timestamps
timestamp = np.arange(100)

# generate sensor readings
sensor_reading = np.random.normal(loc=50, scale=5, size=100)

# add sensor drift before failure
sensor_reading[80:90] += 15  # drift

# failure labels (no failure until 90)
failure = np.zeros(100)
failure[90:] = 1  # failure after time 90

# create DataFrame
df = pd.DataFrame({
    'timestamp': timestamp,
    'sensor_reading': sensor_reading,
    'failure': failure
})

# save to CSV
df.to_csv("dummy_timeseries.csv", index=False)

print("✅ dummy_timeseries.csv created successfully!")
print(df.head())
print(df.tail())
print("DataFrame shape:", df.shape)


# Step 2: Load and Explore the Data


# load CSV
data = pd.read_csv("dummy_timeseries.csv")

# quick summary
print("\nData Overview:")
print(data.head())

# check for missing values
print("\nMissing values:\n", data.isnull().sum())

# basic statistics
print("\nData Statistics:\n", data.describe())

# optional visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(data['timestamp'], data['sensor_reading'], label="Sensor Reading")
plt.plot(data['timestamp'], data['failure']*60, label="Failure (scaled)")
plt.xlabel("Time")
plt.ylabel("Sensor Reading")
plt.legend()
plt.title("Sensor Readings and Failures Over Time")
plt.show()


# Step 3: Train a simple classifier


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# features and target
X = data[['sensor_reading']]  # just sensor reading
y = data['failure']           # target

# split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create and train the decision tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

