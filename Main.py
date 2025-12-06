import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Load dataset from scikit-learn
# -------------------------
# Using load_breast_cancer as a binary classification example.
data_bunch = load_breast_cancer()
X_raw = pd.DataFrame(data_bunch.data, columns=data_bunch.feature_names)
y = pd.Series(data_bunch.target, name='risk')  # 0/1 already

# Quick overview (uncomment to inspect)
# print(X_raw.shape)
# print(X_raw.columns.tolist())
# print(y.value_counts())

# -------------------------
# Optional: Create engineered features to keep pipeline shape
# -------------------------
# We'll create two aggregated features (adherence_score, provenance_score)
# by normalizing and combining some existing columns. These are synthetic
# proxies so the downstream pipeline stays the same.
minmax = MinMaxScaler()

# Choose some columns as proxies for "behavioral consistency" and "reliability".
# This is arbitrary and only for pipeline/illustration purposes.
behavior_columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
provenance_columns = ['worst radius', 'worst texture', 'worst perimeter', 'worst area']

# Ensure chosen columns exist (they do in this dataset). Normalize them 0-1.
behavior_norm = pd.DataFrame(minmax.fit_transform(X_raw[behavior_columns]), columns=behavior_columns)
provenance_norm = pd.DataFrame(minmax.fit_transform(X_raw[provenance_columns]), columns=provenance_columns)

# Adherence-like score: higher means more 'consistent' (we'll take 1 - variance proxy)
adherence_score = 1 - behavior_norm.var(axis=1)  # stable (low variance) => higher adherence
# Provenance-like score: higher means more 'reliable' (we'll take mean of normalized provenance cols)
provenance_score = provenance_norm.mean(axis=1)

# Attach engineered features to the dataframe
X_raw = X_raw.copy()  # avoid modifying original
X_raw['adherence_score'] = adherence_score
X_raw['provenance_score'] = provenance_score

# -------------------------
# Prepare features and target
# -------------------------
features = X_raw.columns.tolist()  # use all original features + engineered ones
X = X_raw[features]

# -------------------------
# Scale features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# Split data (stratify to preserve class balance)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Train Random Forest with hyperparameter tuning
# -------------------------
param_grid = {
    'n_estimators': [100, 200],           # smaller grid for speed; extend if you like
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# -------------------------
# Evaluate
# -------------------------
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("Best hyperparameters:", grid_search.best_params_)
print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# -------------------------
# Feature importances (interpretable output)
# -------------------------
importances = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("\nTop 20 Feature Importances (for preventive-health guidance):")
print(feature_importance_df.head(20).to_string(index=False))

# -------------------------
# Visualize confusion matrix
# -------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()
