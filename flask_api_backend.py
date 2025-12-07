"""
Flask API Backend for Exoplanet Atmospheric Composition Prediction
Minimal API with one endpoint for predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import os
import keras
keras.config.enable_unsafe_deserialization()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and preprocessing
model = None
scaler = None
feature_names = None
target_molecules = None

def custom_loss(y_true, y_pred):
    """Custom loss function for model"""
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    sum_constraint = tf.reduce_mean(tf.square(tf.reduce_sum(y_pred, axis=1) - 100))
    return mae + 0.1 * sum_constraint

def load_model_components():
    """Load model and preprocessing components at startup"""
    global model, scaler, feature_names, target_molecules
    
    try:
        # Custom objects for model loading
        from exoplanet_dl_model import TransformerBlock
        
        model = keras.models.load_model(
            'exoplanet_atmosphere_model.keras',
            custom_objects={
                'TransformerBlock': TransformerBlock,
                'custom_loss': custom_loss
            }
        )
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        target_molecules = joblib.load('target_molecules.pkl')
        
        print("✓ Model and components loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'Exoplanet Atmosphere Predictor',
        'version': '1.0',
        'endpoints': {
            'predict': '/predict [POST]'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Expected JSON input:
    {
        "pl_name": "Kepler-1239 b",
        "pl_orbper": 5.19104016,
        "pl_rade": 2.9,
        "pl_bmasse": null,
        "pl_orbeccen": null,
        "st_teff": 6564,
        "st_rad": 1.57,
        "st_mass": 1.4,
        "sy_dist": 2140.6,
        "pl_eqt": null
    }
    
    Returns JSON with atmospheric composition percentages
    """
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please send JSON data in request body'
            }), 400
        
        # Extract planet name
        planet_name = data.get('pl_name', 'Unknown')
        
        # Prepare features
        features = []
        missing_features = []
        
        for name in feature_names:
            if name in data and data[name] is not None:
                features.append(float(data[name]))
            elif name == 'insolation_flux':
                # Calculate derived feature
                st_rad = data.get('st_rad')
                pl_orbper = data.get('pl_orbper')
                if st_rad and pl_orbper:
                    features.append((float(st_rad) ** 2) / (float(pl_orbper) ** (2/3)))
                else:
                    features.append(0.0)
                    missing_features.append(name)
            elif name == 'surface_gravity':
                pl_bmasse = data.get('pl_bmasse')
                pl_rade = data.get('pl_rade')
                if pl_bmasse and pl_rade:
                    features.append(float(pl_bmasse) / (float(pl_rade) ** 2))
                else:
                    features.append(1.0)  # Default
                    missing_features.append(name)
            elif name == 'stellar_luminosity':
                st_rad = data.get('st_rad')
                st_teff = data.get('st_teff')
                if st_rad and st_teff:
                    features.append((float(st_rad) ** 2) * ((float(st_teff) / 5778) ** 4))
                else:
                    features.append(1.0)  # Default
                    missing_features.append(name)
            else:
                # Use default values for missing features
                if name == 'pl_bmasse':
                    features.append(5.0)  # Default mass
                elif name == 'pl_orbeccen':
                    features.append(0.0)  # Circular orbit
                elif name == 'pl_eqt':
                    # Estimate from stellar temp if missing
                    st_teff = data.get('st_teff', 5778)
                    features.append(float(st_teff) * 0.5)
                else:
                    features.append(0.0)
                missing_features.append(name)
        
        # Scale features
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled, verbose=0)[0]
        
        # Create composition dictionary
        composition = {}
        for i, molecule in enumerate(target_molecules):
            composition[molecule] = round(float(prediction[i]), 2)
        
        # Calculate total (should be ~100)
        total = sum(composition.values())
        
        # Prepare response
        response = {
            'success': True,
            'planet_name': planet_name,
            'atmospheric_composition': composition,
            'total_percentage': round(total, 2),
            'confidence': 'high' if len(missing_features) < 3 else 'medium',
            'missing_features': missing_features if missing_features else None,
            'metadata': {
                'model': 'Transformer-based Deep Learning',
                'molecules_predicted': len(target_molecules),
                'input_features': len(feature_names)
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error processing prediction'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    
    return jsonify({
        'model_loaded': model is not None,
        'input_features': feature_names if feature_names else [],
        'output_molecules': target_molecules if target_molecules else [],
        'feature_count': len(feature_names) if feature_names else 0,
        'molecule_count': len(target_molecules) if target_molecules else 0
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/', '/predict', '/model-info']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500

if __name__ == '__main__':
    print("="*60)
    print("Exoplanet Atmosphere Prediction API")
    print("="*60)
    
    # Load model components
    print("\nLoading model components...")
    if load_model_components():
        print(f"\nInput features: {len(feature_names)}")
        print(f"Output molecules: {target_molecules}")
        print("\n" + "="*60)
        print("Starting Flask server...")
        print("API will be available at: http://localhost:5000")
        print("="*60 + "\n")
        
        # Run Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False  # Set to False for production
        )
    else:
        print("\n✗ Failed to load model. Please train the model first.")
        print("Run: python exoplanet_dl_model.py")