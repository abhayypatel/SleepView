from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configure CORS properly for production
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins temporarily
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

# Global variables to store model and encoders
model = None
label_encoders = {}
target_encoder = None
scaler = None
feature_columns = []

def load_model_and_encoders():
    """Load the trained model and create encoders based on training data"""
    global model, label_encoders, target_encoder, scaler, feature_columns
    
    try:
        # Load the trained model
        model_path = 'model.pkl'  # Try current directory first
        if not os.path.exists(model_path):
            model_path = os.path.join('..', 'model.pkl')  # Try parent directory
            
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        # Extract model and metadata
        if isinstance(model_data, dict):
            model = model_data['model']
            label_encoders = model_data.get('label_encoders', {})
            target_encoder = model_data.get('target_encoder', None)
            scaler = model_data.get('scaler', None)
            feature_columns = model_data.get('feature_columns', [])
        else:
            # If it's just the model, we'll need to recreate encoders
            model = model_data
            setup_encoders()
            
        print("Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Setup default encoders if model loading fails
        setup_encoders()
        return False

def setup_encoders():
    """Setup encoders with expected values"""
    global label_encoders, target_encoder, scaler, feature_columns
    
    # Define expected categorical values based on training data
    categorical_mappings = {
        'gender': ['Male', 'Female'],
        'occupation': ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse',
                      'Engineer', 'Accountant', 'Scientist', 'Lawyer', 'Salesperson', 'Manager'],
        'bmi_category': ['Underweight', 'Normal', 'Overweight', 'Obese'],
        'bp_category': ['Normal', 'Elevated', 'Stage 1', 'Stage 2'],
        'age_group': ['Young', 'Middle', 'Senior']
    }
    
    # Create label encoders
    label_encoders = {}
    for col, values in categorical_mappings.items():
        le = LabelEncoder()
        le.fit(values)
        label_encoders[col] = le
    
    # Create target encoder
    target_encoder = LabelEncoder()
    target_encoder.fit(['None', 'Sleep Apnea', 'Insomnia'])
    
    # Create scaler
    scaler = StandardScaler()
    
    # Define feature columns
    feature_columns = [
        'gender', 'age', 'occupation', 'sleep_duration', 'quality_of_sleep',
        'physical_activity_level', 'stress_level', 'bmi_category',
        'blood_pressure_systolic', 'blood_pressure_diastolic', 'heart_rate', 'daily_steps',
        'bmi_numeric', 'sleep_efficiency', 'activity_steps_ratio', 'bp_category', 'age_group'
    ]

def preprocess_input(data):
    """Preprocess input data to match training format"""
    try:
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Handle blood pressure N/A values with age-based averages
        if data.get('blood_pressure_systolic') == 'N/A' or data.get('blood_pressure_diastolic') == 'N/A':
            age = float(data['age'])
            # Age-based average blood pressure values
            if age < 30:
                avg_systolic, avg_diastolic = 115, 75
            elif age < 50:
                avg_systolic, avg_diastolic = 125, 80
            elif age < 65:
                avg_systolic, avg_diastolic = 135, 85
            else:
                avg_systolic, avg_diastolic = 145, 90
            
            df['blood_pressure_systolic'] = avg_systolic
            df['blood_pressure_diastolic'] = avg_diastolic
        
        # Feature engineering (same as training)
        # 1. BMI Category to numeric
        bmi_mapping = {'Underweight': 1, 'Normal': 2, 'Overweight': 3, 'Obese': 4}
        df['bmi_numeric'] = df['bmi_category'].map(bmi_mapping)
        
        # 2. Sleep efficiency
        df['sleep_efficiency'] = pd.to_numeric(df['quality_of_sleep']) / pd.to_numeric(df['sleep_duration'])
        
        # 3. Activity to steps ratio (handle division by zero)
        daily_steps_safe = pd.to_numeric(df['daily_steps']).replace(0, 1)
        df['activity_steps_ratio'] = pd.to_numeric(df['physical_activity_level']) / (daily_steps_safe / 1000)
        
        # Handle infinite values
        df['activity_steps_ratio'] = df['activity_steps_ratio'].replace([np.inf, -np.inf], np.nan)
        df['activity_steps_ratio'] = df['activity_steps_ratio'].fillna(df['activity_steps_ratio'].median())
        
        # 4. Blood pressure category
        def categorize_bp(systolic, diastolic):
            systolic = float(systolic)
            diastolic = float(diastolic)
            if systolic < 120 and diastolic < 80:
                return 'Normal'
            elif systolic < 130 and diastolic < 80:
                return 'Elevated'
            elif systolic < 140 or diastolic < 90:
                return 'Stage 1'
            else:
                return 'Stage 2'
        
        df['bp_category'] = df.apply(lambda row: categorize_bp(
            row['blood_pressure_systolic'], row['blood_pressure_diastolic']), axis=1)
        
        # 5. Age groups
        def categorize_age(age):
            age = float(age)
            if age < 30:
                return 'Young'
            elif age < 50:
                return 'Middle'
            else:
                return 'Senior'
        
        df['age_group'] = df['age'].apply(categorize_age)
        
        # Convert numeric columns
        numeric_columns = ['age', 'sleep_duration', 'quality_of_sleep', 'physical_activity_level',
                          'stress_level', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                          'heart_rate', 'daily_steps']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Select and order features
        df_features = df[feature_columns].copy()
        
        # Encode categorical variables
        categorical_columns = ['gender', 'occupation', 'bmi_category', 'bp_category', 'age_group']
        
        for col in categorical_columns:
            if col in label_encoders:
                # Handle unknown categories
                try:
                    df_features[col] = label_encoders[col].transform(df_features[col].astype(str))
                except ValueError:
                    # If unknown category, use the most common one (index 0)
                    df_features[col] = 0
        
        # Fill any remaining NaN values
        df_features = df_features.fillna(df_features.median())
        
        return df_features.values
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise e

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields (blood pressure can be N/A)
        required_fields = [
            'gender', 'age', 'occupation', 'sleep_duration', 'quality_of_sleep',
            'physical_activity_level', 'stress_level', 'bmi_category',
            'blood_pressure_systolic', 'blood_pressure_diastolic', 'heart_rate', 'daily_steps'
        ]
        
        missing_fields = [field for field in required_fields if field not in data or data[field] == '']
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        # Preprocess the input data
        processed_data = preprocess_input(data)
        
        # Make prediction
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Get prediction and probabilities
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]
        
        # Convert prediction back to original label
        if target_encoder:
            prediction_label = target_encoder.inverse_transform([prediction])[0]
            class_labels = target_encoder.classes_
        else:
            # Fallback mapping
            label_mapping = {0: 'None', 1: 'Sleep Apnea', 2: 'Insomnia'}
            prediction_label = label_mapping.get(prediction, 'Unknown')
            class_labels = ['None', 'Sleep Apnea', 'Insomnia']
        
        # Create probabilities dictionary
        prob_dict = {}
        for i, label in enumerate(class_labels):
            prob_dict[label] = float(probabilities[i])
        
        # Get confidence (highest probability)
        confidence = float(max(probabilities))
        
        # Prepare response
        response = {
            'prediction': prediction_label,
            'confidence': confidence,
            'probabilities': prob_dict,
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'Sleep disorder prediction API is running'
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Sleep Disorder Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Make sleep disorder prediction',
            '/health': 'GET - Health check',
            '/': 'GET - API information'
        },
        'model_status': 'loaded' if model is not None else 'not loaded'
    })

if __name__ == '__main__':
    print("Starting Sleep Disorder Prediction API...")
    
    # Load model and encoders
    model_loaded = load_model_and_encoders()
    
    if model_loaded:
        print("✅ Model loaded successfully!")
    else:
        print("⚠️  Model not found, using default encoders")
    
    print("🚀 Starting Flask server...")
    
    # For production, don't specify host/port - let the platform handle it
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 