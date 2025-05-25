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
    
    # Debug info
    print("CWD:", os.getcwd(), "– listing:", os.listdir(os.getcwd()))
    
    try:
        # Try multiple possible paths for the model file
        possible_paths = [
            'model.pkl',  # Current directory
            os.path.join('backend', 'model.pkl'),  # In backend subdirectory
            os.path.join('..', 'model.pkl'),  # Parent directory
            os.path.join(os.path.dirname(__file__), 'model.pkl'),  # Same directory as this script
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.pkl'),  # Parent of script directory
        ]
        
        model_path = None
        for path in possible_paths:
            print(f"Checking path: {path}")
            if os.path.exists(path):
                model_path = path
                print(f"Found model at: {model_path}")
                break
        
        if not model_path:
            print("ERROR: Model file not found in any of the expected locations!")
            print("Searched paths:", possible_paths)
            setup_encoders()
            return False
            
        print(f"Attempting to load model from: {model_path}")
        
        with open(model_path, 'rb') as f:
            try:
                model_data = pickle.load(f)
                print("Pickle loaded successfully!")
            except Exception as pickle_error:
                print(f"Pickle loading failed: {pickle_error}")
                print("This is likely a numpy/scikit-learn version mismatch")
                raise pickle_error
            
        print(f"Model data type: {type(model_data)}")
        
        # Extract model and metadata
        if isinstance(model_data, dict):
            print("Model data keys:", list(model_data.keys()))
            model = model_data['model']
            label_encoders = model_data.get('label_encoders', {})
            target_encoder = model_data.get('target_encoder', None)
            scaler = model_data.get('scaler', None)
            
            # Get feature columns from model data, with proper fallback
            loaded_feature_columns = model_data.get('feature_columns', [])
            
            # Define the expected feature columns (same as training)
            expected_feature_columns = [
                'gender', 'age', 'occupation', 'sleep_duration', 'quality_of_sleep',
                'physical_activity_level', 'stress_level', 'bmi_category',
                'blood_pressure_systolic', 'blood_pressure_diastolic', 'heart_rate', 'daily_steps',
                'bmi_numeric', 'sleep_efficiency', 'activity_steps_ratio', 'bp_category', 'age_group'
            ]
            
            # Use loaded feature columns if available and valid, otherwise use expected
            if loaded_feature_columns and len(loaded_feature_columns) > 0:
                feature_columns = loaded_feature_columns
                print(f"Using feature columns from model data: {len(feature_columns)} features")
            else:
                feature_columns = expected_feature_columns
                print(f"Using fallback feature columns: {len(feature_columns)} features")
            
            print(f"Loaded model: {model_data.get('model_name', 'Unknown')}")
            
            # Validate model features
            if hasattr(model, 'n_features_in_'):
                print(f"Model expects {model.n_features_in_} features")
                if len(feature_columns) != model.n_features_in_:
                    print(f"WARNING: Feature columns ({len(feature_columns)}) don't match model's expected features ({model.n_features_in_})")
                    print("Feature columns:", feature_columns)
                    # If mismatch, use the expected feature columns
                    if len(expected_feature_columns) == model.n_features_in_:
                        feature_columns = expected_feature_columns
                        print("Using expected feature columns to match model requirements")
            
            if hasattr(model, 'feature_names_in_'):
                print("Model's feature names:", model.feature_names_in_)
                # If model has feature names, use those
                if len(model.feature_names_in_) > 0:
                    feature_columns = list(model.feature_names_in_)
                    print("Using feature names from trained model")
        else:
            # If it's just the model, we'll need to recreate encoders
            model = model_data
            setup_encoders()
            
            # Validate model features
            if hasattr(model, 'n_features_in_'):
                print(f"Model expects {model.n_features_in_} features")
                if len(feature_columns) != model.n_features_in_:
                    print(f"WARNING: Feature columns ({len(feature_columns)}) don't match model's expected features ({model.n_features_in_})")
                    print("Feature columns:", feature_columns)
            
            if hasattr(model, 'feature_names_in_'):
                print("Model's feature names:", model.feature_names_in_)
                # If model has feature names, use those
                if len(model.feature_names_in_) > 0:
                    feature_columns = list(model.feature_names_in_)
                    print("Using feature names from trained model")
            
        print("Model loaded successfully!")
        print(f"Final feature columns ({len(feature_columns)}): {feature_columns}")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Define feature columns (ensure this matches training exactly)
    feature_columns = [
        'gender', 'age', 'occupation', 'sleep_duration', 'quality_of_sleep',
        'physical_activity_level', 'stress_level', 'bmi_category',
        'blood_pressure_systolic', 'blood_pressure_diastolic', 'heart_rate', 'daily_steps',
        'bmi_numeric', 'sleep_efficiency', 'activity_steps_ratio', 'bp_category', 'age_group'
    ]
    
    print(f"Setup encoders completed with {len(feature_columns)} feature columns")

# Initialize with default values to prevent empty state
def initialize_defaults():
    """Initialize default encoders and feature columns as fallback"""
    global label_encoders, target_encoder, scaler, feature_columns
    
    if not label_encoders:
        setup_encoders()
    
    if not feature_columns:
        feature_columns = [
            'gender', 'age', 'occupation', 'sleep_duration', 'quality_of_sleep',
            'physical_activity_level', 'stress_level', 'bmi_category',
            'blood_pressure_systolic', 'blood_pressure_diastolic', 'heart_rate', 'daily_steps',
            'bmi_numeric', 'sleep_efficiency', 'activity_steps_ratio', 'bp_category', 'age_group'
        ]

# Initialize defaults immediately
initialize_defaults()

def preprocess_input(data):
    """Preprocess input data to match training format"""
    global feature_columns  # Move global declaration to the top
    
    try:
        print("Starting preprocessing with input data:", data)
        print(f"Global feature_columns length: {len(feature_columns)}")
        print(f"Global feature_columns: {feature_columns}")
        
        # Ensure feature_columns is not empty
        if not feature_columns or len(feature_columns) == 0:
            print("ERROR: feature_columns is empty! Setting up default columns...")
            feature_columns = [
                'gender', 'age', 'occupation', 'sleep_duration', 'quality_of_sleep',
                'physical_activity_level', 'stress_level', 'bmi_category',
                'blood_pressure_systolic', 'blood_pressure_diastolic', 'heart_rate', 'daily_steps',
                'bmi_numeric', 'sleep_efficiency', 'activity_steps_ratio', 'bp_category', 'age_group'
            ]
            print(f"Set feature_columns to default: {len(feature_columns)} features")
        
        # Create DataFrame from input
        df = pd.DataFrame([data])
        print("Initial DataFrame shape:", df.shape)
        print("Initial columns:", list(df.columns))
        
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
        
        print("After blood pressure handling:", list(df.columns))
        
        # Convert numeric columns first
        numeric_columns = ['age', 'sleep_duration', 'quality_of_sleep', 'physical_activity_level',
                          'stress_level', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                          'heart_rate', 'daily_steps']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        print("After numeric conversion:", list(df.columns))
        
        # Feature engineering (same as training)
        # 1. BMI Category to numeric
        bmi_mapping = {'Underweight': 1, 'Normal': 2, 'Overweight': 3, 'Obese': 4}
        df['bmi_numeric'] = df['bmi_category'].map(bmi_mapping)
        
        # 2. Sleep efficiency
        df['sleep_efficiency'] = df['quality_of_sleep'] / df['sleep_duration']
        
        # 3. Activity to steps ratio (handle division by zero)
        daily_steps_safe = df['daily_steps'].replace(0, 1)
        df['activity_steps_ratio'] = df['physical_activity_level'] / (daily_steps_safe / 1000)
        
        # Handle infinite values
        df['activity_steps_ratio'] = df['activity_steps_ratio'].replace([np.inf, -np.inf], np.nan)
        df['activity_steps_ratio'] = df['activity_steps_ratio'].fillna(df['activity_steps_ratio'].median())
        
        print("After feature engineering:", list(df.columns))
        
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
        
        print("After category creation:", list(df.columns))
        
        # Encode categorical variables BEFORE feature selection
        categorical_columns = ['gender', 'occupation', 'bmi_category', 'bp_category', 'age_group']
        
        # Ensure we have label encoders - if not, set them up
        if not label_encoders:
            print("WARNING: No label encoders found, setting up default encoders...")
            setup_encoders()
        
        for col in categorical_columns:
            if col in df.columns:
                try:
                    print(f"Encoding {col} with values:", df[col].values)
                    if col in label_encoders:
                        # Handle unseen categories by using the first class
                        df[col] = df[col].astype(str).apply(
                            lambda x: x if x in label_encoders[col].classes_ else label_encoders[col].classes_[0]
                        )
                        df[col] = label_encoders[col].transform(df[col])
                        print(f"Successfully encoded {col}")
                    else:
                        print(f"No encoder found for {col}, using default encoding")
                        # Create a simple mapping for unknown encoders
                        unique_vals = df[col].unique()
                        mapping = {val: i for i, val in enumerate(unique_vals)}
                        df[col] = df[col].map(mapping)
                except ValueError as e:
                    print(f"Error encoding {col}: {e}")
                    if col in label_encoders:
                        print(f"Available categories for {col}:", label_encoders[col].classes_)
                    # If unknown category, use the most common one (index 0)
                    df[col] = 0
        
        print("After categorical encoding:", list(df.columns))
        print("Sample of encoded data:")
        print(df.head())
        
        # Now select and order features
        print("Expected feature columns:", feature_columns)
        print(f"Expected feature columns length: {len(feature_columns)}")
        
        # Ensure all feature columns exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")
        
        df_features = df[feature_columns].copy()
        print("Selected features shape:", df_features.shape)
        print("Selected features columns:", list(df_features.columns))
        print("Data types after selection:")
        print(df_features.dtypes)
        
        # Apply scaling if scaler exists (but only for non-categorical features)
        if scaler is not None:
            try:
                print("Applying feature scaling")
                df_features = pd.DataFrame(
                    scaler.transform(df_features),
                    columns=feature_columns,
                    index=df_features.index
                )
                print("Features shape after scaling:", df_features.shape)
            except Exception as scaling_error:
                print(f"Scaling failed: {scaling_error}")
                print("Continuing without scaling...")
        else:
            print("No scaler available, skipping scaling")
        
        # Fill any remaining NaN values - but only for numeric columns
        print("Filling NaN values...")
        for col in df_features.columns:
            if df_features[col].dtype in ['object', 'string']:
                print(f"Skipping median fill for non-numeric column: {col}")
                df_features[col] = df_features[col].fillna(0)  # Fill categorical with 0
            else:
                if df_features[col].isna().any():
                    median_val = df_features[col].median()
                    print(f"Filling {col} NaN values with median: {median_val}")
                    df_features[col] = df_features[col].fillna(median_val)
        
        print("Final features shape:", df_features.shape)
        print("Final data types:")
        print(df_features.dtypes)
        print("Final features sample:")
        print(df_features.head())
        
        return df_features.values
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        print("Current DataFrame state:")
        if 'df' in locals():
            print(df.head())
            print("DataFrame columns:", list(df.columns))
        print("\nFeature columns:", feature_columns)
        import traceback
        traceback.print_exc()
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
        print("\n=== Starting new prediction request ===")
        print("Received data:", data)
        
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
            print("Missing required fields:", missing_fields)
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        print("All required fields present")
        
        # Ensure we have all necessary components
        if not feature_columns or not label_encoders:
            print("Missing essential components, reinitializing...")
            initialize_defaults()
        
        # Preprocess the input data
        try:
            processed_data = preprocess_input(data)
            print("Data preprocessing successful")
            print("Processed data shape:", processed_data.shape)
        except Exception as preprocess_error:
            print("Preprocessing error:", str(preprocess_error))
            return jsonify({
                'error': 'Data preprocessing failed',
                'details': str(preprocess_error),
                'data_received': data
            }), 400
        
        # Make prediction
        if model is None:
            print("Model is None, attempting to reload...")
            model_loaded = load_model_and_encoders()
            if not model_loaded or model is None:
                error_msg = 'Model not loaded. Please check server logs for details.'
                print(f"CRITICAL ERROR: {error_msg}")
                return jsonify({
                    'error': error_msg,
                    'debug_info': {
                        'cwd': os.getcwd(),
                        'files_in_cwd': os.listdir(os.getcwd()),
                        'model_status': 'failed_to_load'
                    }
                }), 500
        
        try:
            # Get prediction and probabilities
            print("Making prediction with data shape:", processed_data.shape)
            prediction = model.predict(processed_data)[0]
            probabilities = model.predict_proba(processed_data)[0]
            print("Prediction successful")
        except Exception as predict_error:
            print("Prediction error:", str(predict_error))
            return jsonify({
                'error': 'Prediction failed',
                'details': str(predict_error),
                'data_shape': processed_data.shape if processed_data is not None else 'None',
                'model_type': str(type(model))
            }), 500
        
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
        
        print("Prediction completed successfully")
        return jsonify(response)
        
    except Exception as e:
        print(f"Unexpected error in prediction endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'Sleep disorder prediction API is running',
        'debug_info': {
            'cwd': os.getcwd(),
            'files_in_cwd': os.listdir(os.getcwd()),
            'model_type': str(type(model)) if model else 'None',
            'encoders_loaded': len(label_encoders) if label_encoders else 0,
            'feature_columns_count': len(feature_columns) if feature_columns else 0
        }
    })

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """Debug endpoint to manually reload the model"""
    try:
        print("Manual model reload requested...")
        model_loaded = load_model_and_encoders()
        
        return jsonify({
            'success': model_loaded,
            'model_loaded': model is not None,
            'message': 'Model reload completed' if model_loaded else 'Model reload failed',
            'debug_info': {
                'cwd': os.getcwd(),
                'files_in_cwd': os.listdir(os.getcwd()),
                'model_type': str(type(model)) if model else 'None',
                'encoders_loaded': len(label_encoders) if label_encoders else 0,
                'feature_columns_count': len(feature_columns) if feature_columns else 0
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Model reload failed with exception'
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Sleep Disorder Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Make sleep disorder prediction',
            '/health': 'GET - Health check',
            '/reload-model': 'POST - Reload model',
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
        print(f"Model type: {type(model)}")
        print(f"Feature columns: {len(feature_columns)}")
        print(f"Label encoders: {len(label_encoders)}")
    else:
        print("⚠️  Model not found, using default encoders")
        print("⚠️  Predictions may not work correctly without the trained model")
        print("⚠️  Please ensure model.pkl is available in the deployment")
    
    print("🚀 Starting Flask server...")
    
    # For production, don't specify host/port - let the platform handle it
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 
