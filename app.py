from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class EnhancedSP500Predictor:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=300, min_samples_split=30, max_depth=15, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.model_scores = {}
        self.feature_names = []
        self.sp500_data = None
        self.last_trained = None
        self.prediction_history = []

    def fetch_latest_data(self, period="max"):
        try:
            ticker = yf.Ticker("^GSPC")
            data = ticker.history(period=period)
            data = data.reset_index().set_index("Date")
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def prepare_features(self, data):
        if data is None or data.empty:
            return None
            
        data = data.copy()
        data["Tomorrow"] = data["Close"].shift(-1)
        data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
        data = data.loc["1990-01-01":].copy()
        
        horizons = [2, 5, 60, 250, 1000]
        new_predictors = []
        for horizon in horizons:
            rolling_avg = data['Close'].rolling(horizon).mean()
            ratio_col = f"Close_Ratio_{horizon}"
            trend_col = f"Trend_{horizon}"
            data[ratio_col] = data['Close'] / rolling_avg
            data[trend_col] = data['Target'].shift(1).rolling(horizon).sum()
            new_predictors += [ratio_col, trend_col]
        
        data = data.dropna(subset=data.columns[data.columns != "Tomorrow"])
        self.feature_names = ["Close", "Volume", "Open", "High", "Low"] + new_predictors
        self.sp500_data = data
        return data

    def train_ensemble(self):
        data = self.sp500_data
        if data is None or data.empty:
            raise ValueError("No data to train on")
            
        X = data[self.feature_names].iloc[:-1]
        y = data["Target"].iloc[:-1]
        
        self.trained_models = {}
        self.model_scores = {}
        
        for name, model in self.models.items():
            try:
                if name == 'lr':
                    X_scaled = self.scaler.fit_transform(X)
                    model.fit(X_scaled, y)
                    score = cross_val_score(model, X_scaled, y, cv=5, scoring='precision').mean()
                else:
                    model.fit(X, y)
                    score = cross_val_score(model, X, y, cv=5, scoring='precision').mean()
                
                self.trained_models[name] = model
                self.model_scores[name] = score
                print(f"Model {name} trained with score: {score:.3f}")
            except Exception as e:
                print(f"Error training model {name}: {e}")
        
        self.last_trained = datetime.now()

    def predict_next_day(self):
        if self.sp500_data is None or self.sp500_data.empty:
            data = self.fetch_latest_data()
            if data is None:
                return None, None, None
            self.prepare_features(data)
        
        if (self.last_trained is None or 
            datetime.now() - self.last_trained > timedelta(hours=1) or 
            not self.trained_models):
            try:
                self.train_ensemble()
            except Exception as e:
                print(f"Error training models: {e}")
                return None, None, None

        X_latest = self.sp500_data[self.feature_names].iloc[-1:].copy()
        preds = {}
        probs = {}
        
        for name, model in self.trained_models.items():
            try:
                if name == 'lr':
                    X_scaled = self.scaler.transform(X_latest)
                    pred = model.predict(X_scaled)[0]
                    prob = model.predict_proba(X_scaled)[0, 1]
                else:
                    pred = model.predict(X_latest)[0]
                    prob = model.predict_proba(X_latest)[0, 1]
                
                preds[name] = pred
                probs[name] = prob
            except Exception as e:
                print(f"Error predicting with model {name}: {e}")
                continue
        
        if not probs:
            return None, None, None
        
        weights = np.array(list(self.model_scores.values()))
        if weights.sum() > 0:
            weights /= weights.sum()
            ensemble_prob = np.dot(list(probs.values()), weights)
        else:
            ensemble_prob = np.mean(list(probs.values()))
            
        ensemble_pred = 1 if ensemble_prob >= 0.6 else 0
        current_price = float(self.sp500_data['Close'].iloc[-1])
        
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'prediction': int(ensemble_pred),
            'confidence': float(ensemble_prob),
            'current_price': current_price,
        }
        self.prediction_history.append(prediction_data)
        
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
        
        return ensemble_pred, float(ensemble_prob), current_price

    def get_model_accuracy(self):
        if self.model_scores:
            return np.mean(list(self.model_scores.values()))
        return 0.65

predictor = EnhancedSP500Predictor()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/predict', methods=['GET'])
def get_prediction():
    try:
        prediction, confidence, current_price = predictor.predict_next_day()
        
        if prediction is None:
            return jsonify({'error': 'Unable to generate prediction'}), 500
        
        response = {
            'prediction': int(prediction),
            'confidence': float(confidence),
            'current_price': float(current_price),
            'timestamp': datetime.now().isoformat(),
            'market_direction': 'bullish' if prediction == 1 else 'bearish',
            'model_accuracy': float(predictor.get_model_accuracy())
        }
        
        return jsonify(response)
    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_prediction_history():
    try:
        recent_history = predictor.prediction_history[-30:] if predictor.prediction_history else []
        return jsonify({
            'history': recent_history,
            'total_predictions': len(predictor.prediction_history)
        })
    except Exception as e:
        print(f"Error in history endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-data', methods=['GET'])
def get_market_data():
    try:
        if predictor.sp500_data is not None and not predictor.sp500_data.empty:
            latest_data = predictor.sp500_data.iloc[-1]
            
            response = {
                'current_price': float(latest_data['Close']),
                'open_price': float(latest_data['Open']),
                'high_price': float(latest_data['High']),
                'low_price': float(latest_data['Low']),
                'volume': int(latest_data['Volume']),
                'last_updated': datetime.now().isoformat(),
                'model_accuracy': float(predictor.get_model_accuracy()),
                'total_predictions': len(predictor.prediction_history)
            }
            
            return jsonify(response)
        else:
            return jsonify({'error': 'No market data available'}), 500
            
    except Exception as e:
        print(f"Error in market-data endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    try:
        status = {
            'server_status': 'online',
            'last_trained': predictor.last_trained.isoformat() if predictor.last_trained else None,
            'models_trained': len(predictor.trained_models),
            'data_available': predictor.sp500_data is not None and not predictor.sp500_data.empty,
            'prediction_history_count': len(predictor.prediction_history),
            'timestamp': datetime.now().isoformat()
        }
        
        if predictor.sp500_data is not None:
            status['data_last_date'] = str(predictor.sp500_data.index[-1].date())
            status['data_records'] = len(predictor.sp500_data)
        
        return jsonify(status)
        
    except Exception as e:
        print(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting S&P 500 Prediction Server...")
    print("Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)