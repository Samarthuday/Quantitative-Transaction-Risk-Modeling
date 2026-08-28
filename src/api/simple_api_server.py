import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime, timedelta
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

# Define FeatureSelector class for pickle loading
class FeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selector class used in the trained model"""
    def __init__(self, selected_features):
        self.selected_features = selected_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.selected_features]

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
model = None
metadata = None

# Simple in-memory storage for monitoring data
monitoring_stats = {
    'total_transactions': 0,
    'high_risk_count': 0,
    'medium_risk_count': 0,
    'low_risk_count': 0,
    'minimal_risk_count': 0,
    'pending_reviews': 0,
    'alerts_generated': 0,
    'processing_rate': 0.0,
    'uptime_seconds': 0.0
}

recent_alerts = []
high_risk_transactions = []
start_time = datetime.now()

def load_model():
    """Load the trained model and metadata"""
    global model, metadata
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, '..', 'models')
        
        # Load model
        model_path = os.path.join(models_dir, "best_model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(models_dir, "model_metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        logger.info(f"Model loaded successfully: {metadata.get('model_name', 'Unknown')}")
        logger.info(f"Model threshold: {metadata.get('threshold', 0.5):.4f}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def predict_risk(features):
    """Make prediction using the loaded model"""
    if model is None:
        raise ValueError("Model not loaded")
    
    try:
        # Get prediction probabilities
        proba = model.predict_proba(features)
        prediction_proba = proba[:, 1] if proba.shape[1] > 1 else proba.flatten()
        
        # Apply threshold to get binary prediction
        threshold = metadata.get('threshold', 0.5)
        prediction_class = (prediction_proba >= threshold).astype(int)
        
        return prediction_proba, prediction_class
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise

def get_risk_level(risk_score):
    """Convert risk score to risk level"""
    if risk_score < 0.2:
        return "MINIMAL"
    elif risk_score < 0.5:
        return "LOW"
    elif risk_score < 0.8:
        return "MEDIUM"
    else:
        return "HIGH"

def prepare_features(transaction_data):
    """Prepare features for model prediction"""
    try:
        # Create a DataFrame with all required features
        import pandas as pd
        
        # Get timestamp
        timestamp = datetime.fromisoformat(transaction_data.get('timestamp', datetime.now().isoformat()))
        
        # Create feature dictionary with all required features
        features_dict = {
            'Amount': float(transaction_data['amount']),
            'Log_amount': np.log1p(float(transaction_data['amount'])),
            'Receiver_account': hash(transaction_data['receiver_id']) % 1000000,  # Simplified encoding
            'Sender_account': hash(transaction_data['sender_id']) % 1000000,  # Simplified encoding
            'Payment_type': hash(transaction_data['transaction_type']) % 100,  # Simplified encoding
            'Received_currency': hash(transaction_data.get('received_currency', 'USD')) % 100,
            'Hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
            'Hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
            'Month_cos': np.cos(2 * np.pi * timestamp.month / 12),
            'Month_sin': np.sin(2 * np.pi * timestamp.month / 12),
            'Day_of_week_sin': np.sin(2 * np.pi * timestamp.weekday() / 7),
            'Receiver_bank_location': hash(transaction_data.get('receiver_bank_location', 'Unknown')) % 100,
            'Day_of_week_cos': np.cos(2 * np.pi * timestamp.weekday() / 7),
            'Payment_currency': hash(transaction_data.get('payment_currency', 'USD')) % 100,
            'Is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'Sender_bank_location': hash(transaction_data.get('sender_bank_location', 'Unknown')) % 100,
            'Is_night': 1 if timestamp.hour >= 22 or timestamp.hour <= 6 else 0,
            'Amount_rounded': 1 if float(transaction_data['amount']) % 1 == 0 else 0
        }
        
        # Create DataFrame with the exact feature order expected by the model
        expected_features = [
            'Amount', 'Log_amount', 'Receiver_account', 'Sender_account', 'Payment_type',
            'Received_currency', 'Hour_sin', 'Hour_cos', 'Month_cos', 'Month_sin',
            'Day_of_week_sin', 'Receiver_bank_location', 'Day_of_week_cos', 'Payment_currency',
            'Is_weekend', 'Sender_bank_location', 'Is_night', 'Amount_rounded'
        ]
        
        df = pd.DataFrame([features_dict])
        df = df[expected_features]  # Ensure correct order
        
        return df
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        raise

# Load model on startup
if not load_model():
    logger.error("Failed to load model. Server may not function properly.")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'model_loaded': model is not None,
        'model_name': metadata.get('model_name', 'Unknown') if metadata else 'Unknown'
    })

@app.route('/api/process_transaction', methods=['POST'])
def process_transaction():
    """Process a transaction and return risk assessment"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['transaction_id', 'amount', 'sender_id', 'receiver_id', 'transaction_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare features
        features = prepare_features(data)
        
        # Get model prediction
        risk_score, prediction = predict_risk(features)
        risk_score = float(risk_score[0]) if hasattr(risk_score, '__len__') else float(risk_score)
        
        # Determine risk level
        risk_level = get_risk_level(risk_score)
        
        # Determine if review is required
        requires_review = risk_score >= 0.5  # Medium risk threshold
        
        # Get flagged features
        flagged_features = []
        if data['amount'] > 100000:
            flagged_features.append('large_amount')
        if risk_score > 0.8:
            flagged_features.append('high_risk_score')
        if data['amount'] > 50000 and risk_score > 0.6:
            flagged_features.append('suspicious_pattern')
        
        # Update monitoring statistics
        monitoring_stats['total_transactions'] += 1
        
        if risk_level == 'HIGH':
            monitoring_stats['high_risk_count'] += 1
        elif risk_level == 'MEDIUM':
            monitoring_stats['medium_risk_count'] += 1
        elif risk_level == 'LOW':
            monitoring_stats['low_risk_count'] += 1
        else:
            monitoring_stats['minimal_risk_count'] += 1
        
        if requires_review:
            monitoring_stats['pending_reviews'] += 1
        
        # Store high-risk transactions
        if risk_level == 'HIGH':
            high_risk_transactions.append({
                'transaction_id': data['transaction_id'],
                'risk_probability': risk_score,
                'risk_level': risk_level,
                'amount': data['amount'],
                'sender_id': data['sender_id'],
                'receiver_id': data['receiver_id'],
                'timestamp': datetime.now().isoformat()
            })
        
        # Generate alerts for high-risk transactions
        if risk_score > 0.8:
            alert = {
                'type': 'HIGH_RISK_TRANSACTION',
                'severity': 'HIGH',
                'message': f'High risk transaction detected: {data["transaction_id"]}',
                'risk_probability': risk_score,
                'amount': data['amount'],
                'sender': data['sender_id'],
                'receiver': data['receiver_id'],
                'timestamp': datetime.now().isoformat()
            }
            recent_alerts.append(alert)
            monitoring_stats['alerts_generated'] += 1
        
        return jsonify({
            'transaction_id': data['transaction_id'],
            'risk_probability': risk_score,
            'risk_level': risk_level,
            'compliance_status': 'PENDING' if requires_review else 'APPROVED',
            'requires_review': requires_review,
            'flagged_features': flagged_features,
            'processed_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing transaction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    try:
        if metadata is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            else:
                return obj
            
        return jsonify({
            'model_name': metadata.get('model_name', 'Unknown'),
            'threshold': convert_numpy_types(metadata.get('threshold', 0.5)),
            'metrics': convert_numpy_types(metadata.get('metrics', {})),
            'features_used': metadata.get('features_used', []),
            'version': metadata.get('version', '1.0')
        })
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bulk_process', methods=['POST'])
def bulk_process_transactions():
    """Process multiple transactions in bulk"""
    try:
        data = request.json
        transactions_data = data.get('transactions', [])
        
        if not transactions_data:
            return jsonify({'error': 'No transactions provided'}), 400
        
        results = []
        
        for tx_data in transactions_data:
            try:
                # Validate required fields
                required_fields = ['transaction_id', 'amount', 'sender_id', 'receiver_id', 'transaction_type']
                for field in required_fields:
                    if field not in tx_data:
                        results.append({
                            'transaction_id': tx_data.get('transaction_id', 'unknown'),
                            'error': f'Missing required field: {field}'
                        })
                        continue
                
                # Prepare features
                features = prepare_features(tx_data)
                
                # Get model prediction
                risk_score, prediction = predict_risk(features)
                risk_score = float(risk_score[0]) if hasattr(risk_score, '__len__') else float(risk_score)
                
                # Determine risk level
                risk_level = get_risk_level(risk_score)
                
                # Determine if review is required
                requires_review = risk_score >= 0.5
                
                # Get flagged features
                flagged_features = []
                if tx_data['amount'] > 100000:
                    flagged_features.append('large_amount')
                if risk_score > 0.8:
                    flagged_features.append('high_risk_score')
                if tx_data['amount'] > 50000 and risk_score > 0.6:
                    flagged_features.append('suspicious_pattern')
                
                results.append({
                    'transaction_id': tx_data['transaction_id'],
                    'risk_probability': risk_score,
                    'risk_level': risk_level,
                    'compliance_status': 'PENDING' if requires_review else 'APPROVED',
                    'requires_review': requires_review,
                    'flagged_features': flagged_features,
                    'processed_at': datetime.now().isoformat()
                })
                
            except Exception as e:
                results.append({
                    'transaction_id': tx_data.get('transaction_id', 'unknown'),
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'successful': len([r for r in results if 'error' not in r])
        })
        
    except Exception as e:
        logger.error(f"Error in bulk processing: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/stats', methods=['GET'])
def get_monitoring_stats():
    """Get real-time monitoring statistics"""
    try:
        # Update uptime
        monitoring_stats['uptime_seconds'] = (datetime.now() - start_time).total_seconds()
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            else:
                return obj
        
        threshold = metadata.get('threshold', 0.5) if metadata else 0.5
        threshold = convert_numpy_types(threshold)
        
        return jsonify({
            'total_transactions': monitoring_stats['total_transactions'],
            'risk_distribution': {
                'high': monitoring_stats['high_risk_count'],
                'medium': monitoring_stats['medium_risk_count'],
                'low': monitoring_stats['low_risk_count'],
                'minimal': monitoring_stats['minimal_risk_count']
            },
            'pending_reviews': monitoring_stats['pending_reviews'],
            'alerts_generated': monitoring_stats['alerts_generated'],
            'processing_rate': monitoring_stats['processing_rate'],
            'uptime_seconds': monitoring_stats['uptime_seconds'],
            'queue_size': 0,
            'last_alert_time': None,
            'model_info': {
                'model_name': metadata.get('model_name', 'Unknown') if metadata else 'Unknown',
                'threshold': threshold
            }
        })
    except Exception as e:
        logger.error(f"Error getting monitoring stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/alerts', methods=['GET'])
def get_recent_alerts():
    """Get recent alerts"""
    try:
        hours = request.args.get('hours', 24, type=int)
        # Return alerts from the last specified hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_alerts = [alert for alert in recent_alerts if alert.get('timestamp') and datetime.fromisoformat(alert['timestamp']) > cutoff_time]
        
        return jsonify({
            'alerts': filtered_alerts,
            'count': len(filtered_alerts),
            'hours': hours
        })
    except Exception as e:
        logger.error(f"Error getting recent alerts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/high-risk', methods=['GET'])
def get_high_risk_transactions():
    """Get recent high-risk transactions"""
    try:
        limit = request.args.get('limit', 100, type=int)
        return jsonify({
            'transactions': high_risk_transactions[:limit],
            'count': len(high_risk_transactions[:limit]),
            'limit': limit
        })
    except Exception as e:
        logger.error(f"Error getting high-risk transactions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/start', methods=['POST'])
def start_monitoring():
    """Start real-time monitoring"""
    try:
        return jsonify({'message': 'Real-time monitoring started'})
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/stop', methods=['POST'])
def stop_monitoring():
    """Stop real-time monitoring"""
    try:
        return jsonify({'message': 'Real-time monitoring stopped'})
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting simple API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug) 