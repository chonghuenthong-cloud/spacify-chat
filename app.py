import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from preprocess import LeadPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("belive-alps")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
preprocessor = None
ARTIFACT_DIR = os.path.dirname(__file__)
MODEL_FILENAME = "rf_model.pkl"

def load_preprocessor():
    """Load the ML model and preprocessor"""
    global preprocessor
    model_path = os.path.join(ARTIFACT_DIR, MODEL_FILENAME)
    try:
        preprocessor = LeadPreprocessor.load_rf_model(model_path)
        logger.info(f"✅ Loaded preprocessor with model: {type(preprocessor.best_model).__name__} from {model_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load preprocessor from {model_path}: {e}")
        return False

def map_form_to_model_data(form_data):
    """Map form data to the format expected by the ML model"""
    current_date = datetime.now()
    
    # Create the model data with ALL expected columns
    model_data = {
        'budget': float(form_data.get('budget', 0)),
        'no_of_pax': int(form_data.get('pax', 0)),
        'gender': form_data.get('gender', 'Unknown'),
        'nationality': form_data.get('nationality', 'Unknown'),
        'contact_hour': current_date.hour,
        'contact_month': current_date.month,
        'contact_dayofweek': current_date.strftime('%A'),
        
        # Map chat fields to model expected fields
        'location_search': form_data.get('area', 'Unknown'),
        'selected_property': form_data.get('property', 'Unknown'),
        'lead_source': form_data.get('lead_source', 'Unknown'),
        'combined_lead_source': form_data.get('lead_source', 'Unknown'),
        'source_from': form_data.get('lead_source', 'Unknown'),
        'transportation': form_data.get('has_car', 'Unknown'),
        'parking': form_data.get('need_parking', 'Unknown'),
        
        # Handle tenancy period
        'tenancy_period': f"{form_data.get('tenancy_months', 'Unknown')} months",
        
        # Date fields
        'initial_contact_date': current_date,
        'last_action_date': current_date,
        
        # Required fields that the model expects but chat doesn't collect
        'customer_journey': 'Information_Collection',  # Default since this is a chat
        'room_type': form_data.get('room_type', 'Unknown'),
        'rental_proposed': 0,
        'frequency': 1,
        'recencydays': 0,

        'move_in_date': f"{form_data.get('move_in_date', 'Unknown')}",
    }
    return model_data

@app.route('/api/score', methods=['POST'])
def score():
    """Handle score requests"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    if preprocessor is None or preprocessor.best_model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Empty or invalid JSON payload"}), 400
    
    logger.info(f"Incoming keys: {list(payload.keys())}")
    
    try:
        # Map form data to model format
        model_data = map_form_to_model_data(payload)
        
        # Create DataFrame
        df = pd.DataFrame([model_data])
        
        logger.info(f"Created DataFrame with shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        
        # Log sample of the data being sent to model
        logger.info(f"Sample data: {df.iloc[0].to_dict()}")
        
        # Predict
        predictions, probabilities = preprocessor.predict(df)

        # Add predictions, score, probability columns
        probability = float(probabilities[0])
        score_value = max(0.0, min(100.0, probability * 100.0))

        logger.info(f"Model prediction - probability={probability:.4f}, score={score_value:.2f}")
        
        return jsonify({
            "score": round(score_value, 2),
            "success_probability": round(probability, 4),
            "timestamp": datetime.now().isoformat(),
            "model_type": type(preprocessor.best_model).__name__
        })
        
    except Exception as e:
        logger.exception(f"Scoring failed: {e}")
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

@app.route('/')
def root():
    """Root endpoint showing API status"""
    if preprocessor is None or preprocessor.best_model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    return jsonify({
        "status": "BeLive ALPS API running",
        "model_type": type(preprocessor.best_model).__name__,
        "timestamp": datetime.now().isoformat()
    })

# Load the model when starting up
if not load_preprocessor():
    raise RuntimeError("Failed to load the model. Cannot start the application.")

if __name__ == "__main__":
    app.run(port=5000, debug=True)