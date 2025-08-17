import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class EnhancedSP500Predictor:
    def __init__(self):
        # Initialize ensemble models
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

    def fetch_latest_data(self, period="max"):
        ticker = yf.Ticker("^GSPC")
        data = ticker.history(period=period)
        data = data.reset_index().set_index("Date")
        return data

    def prepare_features(self, data):
        """Basic features to match the old dashboard interface"""
        data = data.copy()
        data["Tomorrow"] = data["Close"].shift(-1)
        data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
        data = data.loc["1990-01-01":].copy()
        
        # Add simple moving averages and trend ratios similar to old SP500Predictor
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
        if data is None:
            raise ValueError("No data to train on")
        X = data[self.feature_names].iloc[:-1]
        y = data["Target"].iloc[:-1]
        self.trained_models = {}
        self.model_scores = {}
        for name, model in self.models.items():
            if name == 'lr':
                X_scaled = self.scaler.fit_transform(X)
                model.fit(X_scaled, y)
                score = cross_val_score(model, X_scaled, y, cv=5, scoring='precision').mean()
            else:
                model.fit(X, y)
                score = cross_val_score(model, X, y, cv=5, scoring='precision').mean()
            self.trained_models[name] = model
            self.model_scores[name] = score

    def predict_next_day(self):
        if self.sp500_data is None:
            self.fetch_latest_data()
            self.prepare_features(self.sp500_data)
        X_latest = self.sp500_data[self.feature_names].iloc[-1:].copy()
        preds = {}
        probs = {}
        for name, model in self.trained_models.items():
            if name == 'lr':
                X_scaled = self.scaler.transform(X_latest)
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0,1]
            else:
                pred = model.predict(X_latest)[0]
                prob = model.predict_proba(X_latest)[0,1]
            preds[name] = pred
            probs[name] = prob
        # Simple weighted average ensemble
        weights = np.array(list(self.model_scores.values()))
        weights /= weights.sum()
        ensemble_prob = np.dot(list(probs.values()), weights)
        ensemble_pred = 1 if ensemble_prob >= 0.6 else 0
        current_price = self.sp500_data['Close'].iloc[-1]
        return ensemble_pred, float(ensemble_prob), float(current_price)

    # Example usage for web dashboard
    def make_prediction_for_web():
        predictor = EnhancedSP500Predictor()
        predictor.fetch_latest_data()
        predictor.prepare_features(predictor.sp500_data)
        predictor.train_ensemble()
        prediction, confidence, current_price = predictor.predict_next_day()
        return {
            "prediction": prediction,
            "confidence": confidence,
            "current_price": current_price
        }
