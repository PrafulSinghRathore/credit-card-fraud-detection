"""
Credit Card Fraud Detection - Flask Web Application
====================================================
REST API + Serves the frontend for real-time fraud prediction.
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ============================================================
# App Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

# ============================================================
# Load Model & Scalers
# ============================================================
model = None
scaler_amount = None
scaler_time = None
model_info = None
feature_names = None


def load_model():
    """Load the trained model and scalers."""
    global model, scaler_amount, scaler_time, model_info, feature_names

    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    if not os.path.exists(model_path):
        print("⚠️  No trained model found! Please run 'python train_model.py' first.")
        return False

    model = joblib.load(model_path)
    scaler_amount = joblib.load(os.path.join(MODELS_DIR, 'scaler_amount.pkl'))
    scaler_time = joblib.load(os.path.join(MODELS_DIR, 'scaler_time.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))

    info_path = os.path.join(MODELS_DIR, 'model_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            model_info = json.load(f)

    print(f"✅ Model loaded: {model_info['model_name'] if model_info else 'Unknown'}")
    return True


# ============================================================
# Routes
# ============================================================

@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(STATIC_DIR, filename)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict fraud probability for a transaction."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 503

    try:
        data = request.get_json()

        # Extract features
        time_val = float(data.get('time', 0))
        amount_val = float(data.get('amount', 0))

        # V1-V28 features
        v_features = []
        for i in range(1, 29):
            v_features.append(float(data.get(f'v{i}', 0)))

        # Scale Amount and Time
        scaled_amount = scaler_amount.transform(np.array([[amount_val]]))[0][0]
        scaled_time = scaler_time.transform(np.array([[time_val]]))[0][0]

        # Build feature vector in correct order
        feature_vector = v_features + [scaled_amount, scaled_time]
        feature_df = pd.DataFrame([feature_vector], columns=feature_names)

        # Predict
        probability = float(model.predict_proba(feature_df)[0][1])
        prediction = int(probability >= 0.5)

        # Risk level
        if probability < 0.1:
            risk_level = 'low'
            risk_label = 'Low Risk'
        elif probability < 0.5:
            risk_level = 'medium'
            risk_label = 'Medium Risk'
        elif probability < 0.8:
            risk_level = 'high'
            risk_label = 'High Risk'
        else:
            risk_level = 'critical'
            risk_label = 'Critical Risk'

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': probability,
            'risk_level': risk_level,
            'risk_label': risk_label,
            'is_fraud': bool(prediction),
            'verdict': 'Fraudulent transaction detected!' if prediction else 'Normal transaction - Proceed safely',
            'model_name': model_info['model_name'] if model_info else 'Unknown',
            'confidence': float(max(probability, 1 - probability))
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 400


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Process an uploaded CSV dataset for batch fraud detection."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a CSV (.csv)'}), 400

    try:
        # Read CSV
        df = pd.read_csv(file)
        if len(df) == 0:
            return jsonify({'error': 'Uploaded CSV is empty'}), 400

        # Limit to 50,000 rows for web processing speed
        original_count = len(df)
        df_proc = df.head(50000).copy()

        # Check required columns
        for col in ['Amount', 'Time']:
            if col not in df_proc.columns:
                # case insensitive fallback
                matching = [c for c in df_proc.columns if c.lower() == col.lower()]
                if matching:
                    df_proc[col] = df_proc[matching[0]]
                else:
                    df_proc[col] = 0.0

        # Scale Amount and Time
        scaled_amount = scaler_amount.transform(df_proc['Amount'].values.reshape(-1, 1)).flatten()
        scaled_time = scaler_time.transform(df_proc['Time'].values.reshape(-1, 1)).flatten()

        # Build feature DataFrame matching trained model
        feature_data = {}
        for i in range(1, 29):
            col_name = f'V{i}'
            if col_name in df_proc.columns:
                feature_data[col_name] = df_proc[col_name].fillna(0).values
            elif col_name.lower() in df_proc.columns:
                feature_data[col_name] = df_proc[col_name.lower()].fillna(0).values
            else:
                feature_data[col_name] = np.zeros(len(df_proc))

        feature_data['Scaled_Amount'] = scaled_amount
        feature_data['Scaled_Time'] = scaled_time

        feature_df = pd.DataFrame(feature_data)[feature_names]

        # Batch predict
        probabilities = model.predict_proba(feature_df)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        # Statistics
        total = len(df_proc)
        fraud_count = int(np.sum(predictions))
        fraud_percentage = float((fraud_count / total) * 100) if total > 0 else 0.0

        # Flagged transactions (sorted by risk probability descending)
        flagged_indices = np.where(predictions == 1)[0]
        flagged_list = []
        for idx in flagged_indices:
            row_data = {
                'row_index': int(idx) + 1,
                'amount': float(df_proc.iloc[idx]['Amount']),
                'time': float(df_proc.iloc[idx]['Time']),
                'probability': float(probabilities[idx]),
                'risk_level': 'Critical' if probabilities[idx] >= 0.8 else 'High'
            }
            flagged_list.append(row_data)

        # Sort flagged by highest fraud probability
        flagged_list = sorted(flagged_list, key=lambda x: x['probability'], reverse=True)
        total_fraud_amount = sum(item['amount'] for item in flagged_list)

        return jsonify({
            'success': True,
            'total_analyzed': total,
            'original_count': original_count,
            'fraud_count': fraud_count,
            'normal_count': total - fraud_count,
            'fraud_percentage': fraud_percentage,
            'total_fraud_amount': total_fraud_amount,
            'flagged_samples': flagged_list[:50]  # top 50 flagged
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Return model information and metrics."""
    if model_info is None:
        return jsonify({'error': 'Model info not available'}), 503

    return jsonify({
        'success': True,
        'model_name': model_info['model_name'],
        'metrics': model_info['metrics'],
        'all_model_metrics': model_info.get('all_model_metrics', {}),
        'trained_at': model_info.get('trained_at', 'Unknown'),
        'feature_names': model_info.get('feature_names', [])
    })


@app.route('/api/charts', methods=['GET'])
def get_charts():
    """Return list of available charts."""
    charts_dir = os.path.join(STATIC_DIR, 'charts')
    if not os.path.exists(charts_dir):
        return jsonify({'charts': []})

    charts = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
    return jsonify({
        'success': True,
        'charts': charts
    })


@app.route('/api/sample-transaction', methods=['GET'])
def sample_transaction():
    """Return a sample transaction for testing."""
    # Typical normal transaction values
    samples = {
        'normal': {
            'time': 80000,
            'amount': 149.62,
            'v1': -1.3598, 'v2': -0.0728, 'v3': 2.5363, 'v4': 1.3782,
            'v5': -0.3383, 'v6': 0.4624, 'v7': 0.2396, 'v8': 0.0987,
            'v9': 0.3638, 'v10': 0.0908, 'v11': -0.5516, 'v12': -0.6178,
            'v13': -0.9914, 'v14': -0.3111, 'v15': 1.4682, 'v16': -0.4704,
            'v17': 0.2080, 'v18': 0.0258, 'v19': 0.4040, 'v20': 0.2514,
            'v21': -0.0183, 'v22': 0.2778, 'v23': -0.1105, 'v24': 0.0669,
            'v25': 0.1285, 'v26': -0.1891, 'v27': 0.1336, 'v28': -0.0211,
            'label': 'Normal Transaction'
        },
        'fraud': {
            'time': 4462.0,
            'amount': 239.93,
            'v1': -2.3033, 'v2': 1.7592, 'v3': -0.3597, 'v4': 2.3302,
            'v5': -0.8216, 'v6': -0.0758, 'v7': 0.5623, 'v8': -0.3991,
            'v9': -0.2383, 'v10': -1.5254, 'v11': 2.0329, 'v12': -6.5601,
            'v13': 0.0229, 'v14': -1.4701, 'v15': -0.6988, 'v16': -2.2822,
            'v17': -4.7818, 'v18': -2.6157, 'v19': -1.3344, 'v20': -0.4300,
            'v21': -0.2942, 'v22': -0.9324, 'v23': 0.1727, 'v24': -0.0873,
            'v25': -0.1561, 'v26': -0.5426, 'v27': 0.0396, 'v28': -0.1530,
            'label': 'Fraudulent Transaction'
        }
    }

    sample_type = request.args.get('type', 'normal')
    sample = samples.get(sample_type, samples['normal'])

    return jsonify({'success': True, 'sample': sample})


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("\n" + "🚀" * 20)
    print("  CREDIT CARD FRAUD DETECTION - WEB APP")
    print("🚀" * 20 + "\n")

    load_model()

    print(f"\n🌐 Starting server at http://localhost:5000")
    print(f"   Static files: {STATIC_DIR}\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
