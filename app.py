from flask import Flask, request, jsonify
import pandas as pd
from Retail_Ops_Pipeline.pipeline.prediction_pipeline import PredictionPipeline
from Retail_Ops_Pipeline.utils.logger import get_logger

logger = get_logger("Production_API_Service")
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    """
    Root endpoint to verify the health and status of the Sales Forecasting API.
    
    Returns:
        JSON: Service health information.
    """
    return jsonify({
        "service": "Retail Ops Sales Forecasting API",
        "status": "Healthy",
        "version": "1.0.0"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Inference endpoint that processes incoming feature sets and returns forecasts.
    Expects a JSON body with raw feature data.
    
    Returns:
        JSON: Prediction results or error message.
    """
    try:
        logger.info("Received inference request.")
        data = request.json
        if not data:
            return jsonify({"status": "Error", "message": "No input data provided"}), 400
            
        df = pd.DataFrame(data)
        
        # Initializing the prediction pipeline
        pipeline = PredictionPipeline()
        predictions = pipeline.predict(df)
        
        logger.info("Inference completed successfully.")
        return jsonify({
            "status": "Success",
            "predictions": predictions.tolist()
        })

    except Exception as e:
        logger.error(f"Inference service error: {str(e)}")
        return jsonify({
            "status": "Error",
            "message": str(e)
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
