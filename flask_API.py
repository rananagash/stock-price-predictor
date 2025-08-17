from flask import Flask, jsonify
from predictor import EnhancedSP500Predictor  # your class

app = Flask(__name__)

@app.route("/api/predict", methods=["GET"])
def predict():
    predictor = EnhancedSP500Predictor()
    result = predictor.get_prediction_with_analysis()

    # Build JSON response
    response = {
        "prediction": int(result['prediction']),
        "probability": float(result['probability']),
        "threshold": float(result['threshold']),
        "current_price": float(result['current_price']),
        "current_vix": float(result['current_vix']),
        "performance_metrics": predictor.performance_metrics,
        "top_features": predictor.top_features,   # however you store them
        "model_scores": predictor.model_scores
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
