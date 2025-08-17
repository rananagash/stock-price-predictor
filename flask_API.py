from flask import Flask, jsonify
from predictor import EnhancedSP500Predictor
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow cross-origin requests

# Global predictor instance (optional, to reuse across requests)
predictor = EnhancedSP500Predictor()

@app.route("/api/predict", methods=["GET"])
def predict():
    try:
        result = predictor.get_prediction_with_analysis()
        
        if result is None:
            raise ValueError("Prediction failed. Check server logs.")
        
        # Store top_features in predictor for Flask response
        predictor.top_features = result.get('top_features', {})

        response = {
            "prediction": int(result['prediction']),
            "probability": float(result['probability']),
            "threshold": float(result['threshold']),
            "current_price": float(result['current_price']),
            "current_vix": float(result['current_vix']),
            "performance_metrics": result.get('performance_metrics', {}),
            "top_features": predictor.top_features,
            "model_scores": result.get('model_scores', {})
        }

    except Exception as e:
        response = {"error": str(e)}
        print(f"Error in /api/predict: {str(e)}")

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
