import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate synthetic data (since real data is not provided)
# Features: BMI, age, sleep_hours, sleep_irregularity, daily_steps, step_variance, meal_timing_variability, HRV, diet_score, phone_usage_variance, weight_change_trends
# Target: risk (1 for high risk of diabetes/hypertension, 0 for low risk)
np.random.seed(42)
n_samples = 1000

data = {
    'BMI': np.random.normal(25, 5, n_samples),
    'age': np.random.randint(18, 80, n_samples),
    'sleep_hours': np.random.normal(7, 1.5, n_samples),
    'sleep_irregularity': np.random.uniform(0, 1, n_samples),  # Higher means more irregular
    'daily_steps': np.random.normal(8000, 2000, n_samples),
    'step_variance': np.random.uniform(0, 1, n_samples),  # Higher means more variance
    'meal_timing_variability': np.random.uniform(0, 1, n_samples),  # Higher means more variable
    'HRV': np.random.normal(50, 10, n_samples),  # Heart Rate Variability
    'diet_score': np.random.uniform(0, 100, n_samples),  # Self-reported diet quality
    'phone_usage_variance': np.random.uniform(0, 1, n_samples),  # Higher means more variance
    'weight_change_trends': np.random.normal(0, 2, n_samples)  # Positive for weight gain, negative for loss
}

df = pd.DataFrame(data)

# Generate target based on some logic (simplified)
df['risk'] = ((df['BMI'] > 25) & (df['age'] > 40) & (df['sleep_irregularity'] > 0.5) & (df['step_variance'] > 0.5)).astype(int)

# Step 2: Compute Adherence-State Estimation (Behavioral Consistency Score)
# Score based on sleep regularity (1 - irregularity), step consistency (1 - variance), meal stability (1 - variability), phone usage consistency (1 - variance)
df['adherence_score'] = (
    (1 - df['sleep_irregularity']) + 
    (1 - df['step_variance']) + 
    (1 - df['meal_timing_variability']) + 
    (1 - df['phone_usage_variance'])
) / 4  # Average score, higher means more consistent behavior

# Step 3: Compute Data-Provenance Reliability Score
# Detect unreliable diet data: If diet_score is high but weight_change_trends is positive (inconsistent), or HRV is low (stress indicator)
# Reliability score: Higher means more reliable
diet_reliability = 1 - (df['weight_change_trends'] > 0).astype(int) * (df['diet_score'] / 100)  # Penalize if diet claims healthy but weight increasing
hrv_penalty = 1 - (df['HRV'] < 40).astype(int) * 0.5  # Penalize low HRV
df['provenance_score'] = (diet_reliability + hrv_penalty) / 2  # Average reliability score

# Step 4: Prepare features
features = ['BMI', 'age', 'sleep_hours', 'sleep_irregularity', 'daily_steps', 'step_variance', 
            'meal_timing_variability', 'HRV', 'diet_score', 'phone_usage_variance', 'weight_change_trends', 
            'adherence_score', 'provenance_score']
X = df[features]
y = df['risk']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train Random Forest with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Step 7: Evaluate
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# Step 8: Interpretable output (Feature Importances for preventive guidance)
importances = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances (for preventive-health guidance):")
print(feature_importance_df)

# Visualize confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
