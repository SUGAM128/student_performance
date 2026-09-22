import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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


print("\nFirst five rows:")
print(df.head())


# =========================
# Data Visualization
# =========================

sns.set_theme(style="whitegrid")


# 1. Pass and Fail Count
plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="pass")

plt.title("Number of Students Who Passed and Failed")
plt.xlabel("Result")
plt.ylabel("Number of Students")
plt.xticks([0, 1], ["Fail", "Pass"])

plt.show()


# 2. Study Time Distribution
plt.figure(figsize=(7, 4))

sns.histplot(data=df, x="study_time", bins=15, kde=True)

plt.title("Distribution of Study Time")
plt.xlabel("Study Time (hours per week)")
plt.ylabel("Number of Students")

plt.show()


# 3. Attendance Distribution
plt.figure(figsize=(7, 4))

sns.histplot(data=df, x="attendance", bins=15, kde=True)

plt.title("Distribution of Student Attendance")
plt.xlabel("Attendance (%)")
plt.ylabel("Number of Students")

plt.show()


# 4. Previous Score vs Result
plt.figure(figsize=(7, 4))

sns.boxplot(data=df, x="pass", y="previous_score")

plt.title("Previous Score by Student Result")
plt.xlabel("Result")
plt.ylabel("Previous Score")
plt.xticks([0, 1], ["Fail", "Pass"])

plt.show()


# 5. Attendance vs Previous Score
plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="attendance",
    y="previous_score",
    hue="pass"
)

plt.title("Attendance vs Previous Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Previous Score")
plt.legend(title="Result", labels=["Fail", "Pass"])

plt.show()


# 6. Study Time vs Result
plt.figure(figsize=(7, 4))

sns.boxplot(data=df, x="pass", y="study_time")

plt.title("Study Time by Student Result")
plt.xlabel("Result")
plt.ylabel("Study Time (hours per week)")
plt.xticks([0, 1], ["Fail", "Pass"])

plt.show()


# 7. Correlation Heatmap
plt.figure(figsize=(8, 6))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.show()


# =========================
# Machine Learning
# =========================

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


# =========================
# Feature Importance
# =========================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)


plt.figure(figsize=(8, 5))

sns.barplot(
    x=importance.values,
    y=importance.index
)

plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.show()


# Save model and feature columns
model_data = {
    "model": model,
    "columns": X.columns.tolist()
}

joblib.dump(model_data, "student_model.pkl")

print("\nModel saved as student_model.pkl")