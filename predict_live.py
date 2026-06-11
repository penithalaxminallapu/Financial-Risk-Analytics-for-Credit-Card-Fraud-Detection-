
import pandas as pd
import sqlite3
import joblib

# 1. Load Model
model = joblib.load('best_random_forest_model.pkl')

# 2. Load New Data from Database
conn = sqlite3.connect('F:/New folder/Inttrvu/capstone/Capstone_Projects/Database.db')
live_data = pd.read_sql_query("SELECT * FROM Fraud_detection_live", conn)
conn.close()

# 3. Preprocessing
live_data_processed = live_data.drop(['nameOrig', 'nameDest'], axis=1)

# 4. Predict
y_pred_live = model.predict(live_data_processed)
y_prob_live = model.predict_proba(live_data_processed)[:, 1]

# 5. Attach Predictions
live_data['isFraud_predicted'] = y_pred_live
live_data['fraud_probability'] = y_prob_live

# 6. Save Predictions
live_data.to_csv('live_predictions.csv', index=False)
print("Predictions saved to live_predictions.csv")
