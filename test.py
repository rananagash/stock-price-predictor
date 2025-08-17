# test_metrics.py
from predictor import EnhancedSP500Predictor

def main():
    # Initialize
    predictor = EnhancedSP500Predictor()
    
    # Load smaller dataset for testing (2 years)
    print("Loading data...")
    predictor.fetch_latest_data(period="2y")  
    predictor.prepare_features(predictor.sp500_data)
    
    # Train models
    print("Training models...")
    predictor.train_ensemble()
    
    # Print metrics
    print("\n=== Model Validation ===")
    for name, score in predictor.model_scores.items():
        print(f"{name.upper()}: {score:.1%} precision")
    
    # Next-day prediction
    pred, prob, price = predictor.predict_next_day()
    print(f"\nNext Day Prediction: {'UP' if pred else 'DOWN'} (Confidence: {prob:.1%} at ${price:.2f})")

if __name__ == "__main__":
    main()
