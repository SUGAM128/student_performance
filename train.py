import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("student_performance.csv")

# Show basic information
print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())


# Remove duplicate rows
df = df.drop_duplicates()


# Separate features and target
X = df.drop("pass", axis=1)
y = df["pass"]


# Convert categorical columns into numbers
X = pd.get_dummies(X)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Test model
y_pred = model.predict(X_test)


# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save model AND feature columns
model_data = {
    "model": model,
    "columns": X.columns.tolist()
}

joblib.dump(model_data, "student_model.pkl")

print("\nModel saved as student_model.pkl")
print (df.head())