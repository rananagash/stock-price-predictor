import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class EnhancedSP500Predictor:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=300, min_samples_split=30, 
                                       max_depth=15, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                           max_depth=6, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.sp500_data = None
        self.performance_metrics = {}
        
    def fetch_comprehensive_data(self, period="max"):
        """Fetch comprehensive market data"""
        print("Fetching comprehensive market data...")
        
        # S&P 500 data
        sp500 = yf.Ticker("^GSPC").history(period=period)
        
        # VIX (Volatility Index)
        try:
            vix = yf.Ticker("^VIX").history(period=period)['Close']
            vix.name = 'VIX'
        except:
            vix = pd.Series(index=sp500.index, name='VIX')
            vix.fillna(20, inplace=True)  # Default VIX value
        
        # Treasury yield (10-year)
        try:
            treasury = yf.Ticker("^TNX").history(period=period)['Close']
            treasury.name = 'Treasury_10Y'
        except:
            treasury = pd.Series(index=sp500.index, name='Treasury_10Y')
            treasury.fillna(2.5, inplace=True)  # Default yield
        
        # Dollar Index
        try:
            dxy = yf.Ticker("DX-Y.NYB").history(period=period)['Close']
            dxy.name = 'DXY'
        except:
            dxy = pd.Series(index=sp500.index, name='DXY')
            dxy.fillna(100, inplace=True)  # Default DXY value
        
        # Combine all data
        combined_data = pd.concat([sp500, vix, treasury, dxy], axis=1)
        combined_data.fillna(method='ffill', inplace=True)
        
        return combined_data
    
    def create_advanced_features(self, data):
        """Create advanced technical and market features"""
        df = data.copy()
        
        # Basic features
        df['Tomorrow'] = df['Close'].shift(-1)
        df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Technical indicators using TA-Lib
        try:
            # RSI
            df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(df['Close'].values)
            df['MACD'] = macd
            df['MACD_Signal'] = macdsignal
            df['MACD_Hist'] = macdhist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'].values)
            df['BB_Upper'] = bb_upper
            df['BB_Lower'] = bb_lower
            df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic
            df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'].values, 
                                                      df['Low'].values, 
                                                      df['Close'].values)
            
            # Average True Range
            df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values)
            
            # Williams %R
            df['Williams_R'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values)
            
        except:
            # Manual calculations if TA-Lib is not available
            df['RSI'] = self.calculate_rsi(df['Close'])
            df['BB_Position'] = 0.5  # Neutral position
            df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        
        # Volume features
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Price_Volume'] = df['Close'] * df['Volume']
        
        # Multiple timeframe features
        horizons = [5, 10, 20, 50, 200]
        for horizon in horizons:
            # Moving averages
            df[f'SMA_{horizon}'] = df['Close'].rolling(horizon).mean()
            df[f'Price_SMA_Ratio_{horizon}'] = df['Close'] / df[f'SMA_{horizon}']
            
            # Volatility
            df[f'Volatility_{horizon}'] = df['Returns'].rolling(horizon).std()
            
            # Momentum
            df[f'Momentum_{horizon}'] = df['Close'] / df['Close'].shift(horizon) - 1
            
            # Trend strength
            df[f'Trend_{horizon}'] = df['Target'].shift(1).rolling(horizon).mean()
        
        # Market regime features
        if 'VIX' in df.columns:
            df['VIX_MA'] = df['VIX'].rolling(20).mean()
            df['VIX_Ratio'] = df['VIX'] / df['VIX_MA']
            df['Fear_Greed'] = np.where(df['VIX'] > 20, 1, 0)  # Fear indicator
        
        # Seasonal features
        df['Month'] = df.index.month
        df['DayOfWeek'] = df.index.dayofweek
        df['Quarter'] = df.index.quarter
        df['IsMonthEnd'] = df.index.is_month_end.astype(int)
        df['IsMonthStart'] = df.index.is_month_start.astype(int)
        
        # Economic cycle features
        if 'Treasury_10Y' in df.columns:
            df['Yield_MA'] = df['Treasury_10Y'].rolling(50).mean()
            df['Yield_Trend'] = df['Treasury_10Y'] - df['Yield_MA']
        
        # Filter from 1990 onwards
        df = df.loc["1990-01-01":].copy()
        
        # Remove rows with too many NaN values
        df = df.dropna(thresh=len(df.columns)*0.7)  # Keep rows with at least 70% non-null values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Manual RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def select_features(self, df):
        """Select the most important features"""
        # Exclude non-predictive columns
        exclude_cols = ['Tomorrow', 'Target', 'Dividends', 'Stock Splits']
        
        # Get all potential feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove features with too many NaN or infinite values
        valid_features = []
        for col in feature_cols:
            if df[col].isna().sum() / len(df) < 0.1:  # Less than 10% missing
                if not np.isinf(df[col]).any():  # No infinite values
                    valid_features.append(col)
        
        self.feature_names = valid_features
        return valid_features
    
    def create_ensemble_model(self, X_train, y_train):
        """Create and train ensemble model"""
        print("Training ensemble model...")
        
        # Train individual models
        trained_models = {}
        scores = {}
        
        for name, model in self.models.items():
            if name == 'lr':
                # Scale features for logistic regression
                X_scaled = self.scaler.fit_transform(X_train)
                model.fit(X_scaled, y_train)
                cv_scores = cross_val_score(model, X_scaled, y_train, cv=5, scoring='precision')
            else:
                model.fit(X_train, y_train)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='precision')
            
            trained_models[name] = model
            scores[name] = cv_scores.mean()
            print(f"{name.upper()} CV Precision: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.trained_models = trained_models
        self.model_scores = scores
        
        return trained_models
    
    def predict_ensemble(self, X_test):
        """Make ensemble predictions"""
        predictions = {}
        probabilities = {}
        
        for name, model in self.trained_models.items():
            if name == 'lr':
                X_scaled = self.scaler.transform(X_test)
                pred_proba = model.predict_proba(X_scaled)[:, 1]
                pred = model.predict(X_scaled)
            else:
                pred_proba = model.predict_proba(X_test)[:, 1]
                pred = model.predict(X_test)
            
            predictions[name] = pred
            probabilities[name] = pred_proba
        
        # Weighted ensemble based on CV scores
        weights = np.array(list(self.model_scores.values()))
        weights = weights / weights.sum()  # Normalize weights
        
        # Weighted average of probabilities
        ensemble_proba = np.average(np.column_stack(list(probabilities.values())), 
                                   weights=weights, axis=1)
        
        # Apply dynamic threshold based on market conditions
        threshold = self.calculate_dynamic_threshold(X_test)
        ensemble_pred = (ensemble_proba >= threshold).astype(int)
        
        return ensemble_pred, ensemble_proba, threshold
    
    def calculate_dynamic_threshold(self, X_test):
        """Calculate dynamic threshold based on market conditions"""
        base_threshold = 0.55
        
        # Adjust based on VIX if available
        if 'VIX' in self.feature_names:
            vix_idx = self.feature_names.index('VIX')
            current_vix = X_test.iloc[-1, vix_idx] if hasattr(X_test, 'iloc') else X_test[-1, vix_idx]
            
            if current_vix > 25:  # High volatility
                base_threshold += 0.05
            elif current_vix < 15:  # Low volatility
                base_threshold -= 0.03
        
        return np.clip(base_threshold, 0.5, 0.7)
    
    def backtest_model(self, data, start_idx=2000, step=250):
        """Comprehensive backtesting"""
        print("Running comprehensive backtest...")
        
        all_predictions = []
        all_targets = []
        
        for i in range(start_idx, len(data), step):
            if i + step >= len(data):
                break
                
            # Split data
            train_data = data.iloc[:i].copy()
            test_data = data.iloc[i:i+step].copy()
            
            if len(train_data) < 500:  # Need minimum training data
                continue
            
            # Prepare features
            feature_cols = self.select_features(train_data)
            X_train = train_data[feature_cols]
            y_train = train_data['Target']
            X_test = test_data[feature_cols]
            y_test = test_data['Target']
            
            # Train and predict
            self.create_ensemble_model(X_train, y_train)
            predictions, probabilities, threshold = self.predict_ensemble(X_test)
            
            all_predictions.extend(predictions)
            all_targets.extend(y_test.values)
        
        # Calculate performance metrics
        if all_predictions:
            precision = precision_score(all_targets, all_predictions)
            accuracy = accuracy_score(all_targets, all_predictions)
            
            self.performance_metrics = {
                'precision': precision,
                'accuracy': accuracy,
                'total_predictions': len(all_predictions),
                'positive_predictions': sum(all_predictions)
            }
            
            print(f"Backtest Results:")
            print(f"Precision: {precision:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Total Predictions: {len(all_predictions)}")
        
        return self.performance_metrics
    
    def get_prediction_with_analysis(self):
        """Get prediction with comprehensive analysis"""
        try:
            print("Fetching data and making prediction...")
            
            # Get comprehensive data
            raw_data = self.fetch_comprehensive_data()
            
            # Create features
            processed_data = self.create_advanced_features(raw_data)
            
            # Run backtest for performance validation
            self.backtest_model(processed_data)
            
            # Prepare for prediction
            feature_cols = self.select_features(processed_data)
            X = processed_data[feature_cols]
            y = processed_data['Target']
            
            # Train final model on all available data except last row
            X_train = X.iloc[:-1]
            y_train = y.iloc[:-1]
            X_latest = X.iloc[-1:].copy()
            
            # Train ensemble
            self.create_ensemble_model(X_train, y_train)
            
            # Make prediction
            prediction, probability, threshold = self.predict_ensemble(X_latest)
            
            # Get current market data
            current_price = processed_data['Close'].iloc[-1]
            current_vix = processed_data.get('VIX', pd.Series([20])).iloc[-1]
            
            # Feature importance analysis
            rf_model = self.trained_models['rf']
            feature_importance = pd.Series(rf_model.feature_importances_, 
                                         index=feature_cols).sort_values(ascending=False)
            
            return {
                'prediction': int(prediction[0]),
                'probability': float(probability[0]),
                'threshold': float(threshold),
                'current_price': float(current_price),
                'current_vix': float(current_vix),
                'performance_metrics': self.performance_metrics,
                'top_features': feature_importance.head(10).to_dict(),
                'model_scores': self.model_scores,
                'processed_data': processed_data
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None

# Global predictor instance
predictor = EnhancedSP500Predictor()