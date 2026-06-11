
import pandas as pd
import sqlite3
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report

# 1. Load Data from Database
conn = sqlite3.connect('F:/New folder/Inttrvu/capstone/Capstone_Projects/Database.db')
df = pd.read_sql_query("SELECT * FROM Fraud_detection", conn)
conn.close()

# 2. Preprocessing
df = df.drop(['nameOrig', 'nameDest'], axis=1)
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Random Forest Hyperparameter Tuning
param_dist = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
rf_random = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, scoring='f1', verbose=2, random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)

# 5. Best Model
best_rf_model = rf_random.best_estimator_

# 6. Evaluation
y_pred = best_rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Save Model
joblib.dump(best_rf_model, 'best_random_forest_model.pkl')
print("Model saved as best_random_forest_model.pkl")
