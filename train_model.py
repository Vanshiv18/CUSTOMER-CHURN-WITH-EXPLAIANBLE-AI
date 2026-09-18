import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib


df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Loaded Successfully!")
print(df.head())




df = df.drop("customerID", axis=1)


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())



encoders = {}

for col in df.select_dtypes(include=["object", "string", "str"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le



X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

print("\nData Types:\n")
print(X.dtypes)

print("\nObject Columns:")
print(X.select_dtypes(include=["object"]).columns)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("=" * 50)
print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1 Score  :", round(f1 * 100, 2), "%")
print("=" * 50)

print("\nDetailed Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))



joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Model Saved Successfully!")



import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.png")
print("SHAP Summary Plot Saved Successfully!")