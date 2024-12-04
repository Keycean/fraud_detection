import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib
from datetime import datetime
import os


class FraudDetectionModel:
    def __init__(self, eps=0.3, min_samples=10):
        self.scaler = StandardScaler()
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)

    def preprocess_transaction(self, transaction_data):
        """Preprocess a single transaction or batch of transactions."""
        if isinstance(transaction_data, dict):
            timestamp = datetime.fromisoformat(transaction_data['time'].replace('Z', '+00:00'))
            features = pd.DataFrame([{
                'amount': float(transaction_data['amount']),
                'hour': timestamp.hour,
                'day_of_week': timestamp.weekday()
            }])
        else:
            features = pd.DataFrame({
                'amount': transaction_data['amount'].astype(float),
                'hour': pd.to_datetime(transaction_data['time']).dt.hour,
                'day_of_week': pd.to_datetime(transaction_data['time']).dt.weekday
            })
        return features

    def fit(self, transactions_df):
        """Train the model on historical transaction data."""
        features = self.preprocess_transaction(transactions_df)
        scaled_features = self.scaler.fit_transform(features)
        self.dbscan.fit(scaled_features)
        return self

    def predict(self, transaction_data):
        """Predict if a transaction or batch of transactions is fraudulent."""
        features = self.preprocess_transaction(transaction_data)
        
        if features.shape[1] != 3:
            raise ValueError(f"Unexpected number of features: {features.shape[1]}. Expected 3.")
        
        scaled_features = self.scaler.transform(features)
        labels = self.dbscan.fit_predict(scaled_features)
        return (labels == -1).astype(int)

    def save_model(self, model_filepath, scaler_filepath):
        """Save the DBSCAN model and scaler as separate files."""
        os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_filepath), exist_ok=True)
        joblib.dump(self.dbscan, model_filepath)
        joblib.dump(self.scaler, scaler_filepath)

    def load_model(self, model_filepath, scaler_filepath):
        """Load the DBSCAN model and scaler from separate files."""
        self.dbscan = joblib.load(model_filepath)
        self.scaler = joblib.load(scaler_filepath)
        return self


def generate_training_data(n_samples=10000):
    """Generate synthetic transaction data."""
    np.random.seed(42)
    normal_transactions = pd.DataFrame({
        'amount': np.random.normal(100, 50, n_samples),
        'time': pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    })
    n_fraud = int(n_samples * 0.05)
    fraud_transactions = pd.DataFrame({
        'amount': np.random.normal(1000, 200, n_fraud),
        'time': pd.date_range(start='2024-01-01', periods=n_fraud, freq='H')
    })
    all_transactions = pd.concat([normal_transactions, fraud_transactions])
    return all_transactions.sample(frac=1).reset_index(drop=True)


def train_and_save_model():
    """Train the model and save it to disk."""
    training_data = generate_training_data()
    model = FraudDetectionModel(eps=0.3, min_samples=10)
    model.fit(training_data)
    model.save_model('models/fraud_detection_model.joblib', 'models/fraud_detection_scaler.joblib')
    print("Model and scaler saved successfully.")
    return model
