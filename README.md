# SleepView - AI-Powered Sleep Disorder Prediction

![SleepView Logo](https://img.shields.io/badge/SleepView-AI%20Sleep%20Analysis-6366f1?style=for-the-badge)

SleepView is a comprehensive full-stack machine learning application that predicts sleep disorders based on lifestyle and health factors. Using advanced AI models trained on the Sleep Health and Lifestyle Dataset, it provides personalized insights and recommendations for better sleep health.

## 🌟 Features

- **AI-Powered Predictions**: Advanced machine learning models (XGBoost, Random Forest, Logistic Regression)
- **Comprehensive Analysis**: Evaluates 12+ lifestyle and health factors
- **Modern Web Interface**: Responsive React frontend with beautiful UI
- **Real-time Predictions**: Fast API responses with confidence scores
- **Personalized Recommendations**: Health tips based on prediction results
- **Privacy-Focused**: No data storage, real-time processing only

## 🎯 Supported Sleep Disorders

- **Sleep Apnea**: Breathing interruptions during sleep
- **Insomnia**: Difficulty falling or staying asleep
- **No Disorder**: Healthy sleep patterns detected

## 📊 Model Performance

| Model | Accuracy | F1-Score | ROC AUC |
|-------|----------|----------|---------|
| XGBoost | 96.8% | 0.97 | 0.99 |
| Random Forest | 95.2% | 0.95 | 0.98 |
| Logistic Regression | 92.5% | 0.93 | 0.96 |

## 🛠️ Technology Stack

### Frontend
- **React 18** - Modern UI framework
- **Styled Components** - CSS-in-JS styling
- **Framer Motion** - Smooth animations
- **Axios** - HTTP client
- **React Router** - Navigation

### Backend
- **Flask** - Python web framework
- **Scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Machine Learning
- **Data Preprocessing** - Feature engineering and cleaning
- **Model Training** - Multiple algorithm comparison
- **Model Evaluation** - Cross-validation and metrics
- **Model Deployment** - Pickle serialization

## 📁 Project Structure

```
SleepView/
├── sleepDetectionModel.ipynb    # Jupyter notebook with ML pipeline
├── Sleep_health_and_lifestyle_dataset 2.csv  # Training dataset
├── model.pkl                    # Trained model (generated)
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.js
│   │   │   ├── Home.js
│   │   │   ├── PredictionForm.js
│   │   │   ├── Results.js
│   │   │   └── About.js
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
├── backend/
│   ├── app.py
│   └── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Jupyter Notebook** (for model training)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SleepView
```

### 2. Train the Machine Learning Model

```bash
# Install Jupyter and required packages
pip install jupyter pandas numpy scikit-learn xgboost matplotlib seaborn plotly

# Open and run the notebook
jupyter notebook sleepDetectionModel.ipynb
```

**Important**: Run all cells in the notebook to train the model and generate `model.pkl`

### 3. Set Up the Backend

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the Flask server
python app.py
```

The backend will be available at `http://localhost:5000`

### 4. Set Up the Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will be available at `http://localhost:3000`

## 📖 Usage Guide

### 1. Access the Application
Open your browser and navigate to `http://localhost:3000`

### 2. Complete the Assessment
- Fill out the prediction form with your sleep and lifestyle information
- All fields are required for accurate predictions
- Use the interactive sliders for numerical inputs

### 3. View Results
- Get instant AI-powered predictions
- See confidence scores and detailed analysis
- Receive personalized health recommendations

### 4. Explore Features
- Learn about the methodology in the About section
- Take multiple assessments to track changes
- Review health tips and recommendations

## 🔬 Machine Learning Pipeline

### Data Preprocessing
1. **Data Cleaning**: Handle missing values and outliers
2. **Feature Engineering**: Create derived features (BMI numeric, sleep efficiency, etc.)
3. **Encoding**: Transform categorical variables
4. **Scaling**: Normalize numerical features

### Model Training
1. **Data Splitting**: 80/20 train-test split
2. **Cross-Validation**: 5-fold CV for robust evaluation
3. **Hyperparameter Tuning**: Grid search optimization
4. **Model Comparison**: Evaluate multiple algorithms

### Model Evaluation
- **Accuracy**: Overall prediction correctness
- **Precision/Recall**: Class-specific performance
- **F1-Score**: Balanced precision-recall metric
- **ROC AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification results

## 📊 Dataset Information

- **Source**: Sleep Health and Lifestyle Dataset
- **Samples**: 375 individuals
- **Features**: 13 attributes including:
  - Demographics (age, gender, occupation)
  - Sleep metrics (duration, quality)
  - Lifestyle factors (activity, stress, BMI)
  - Health indicators (blood pressure, heart rate)

## 🔧 API Documentation

### Endpoints

#### `POST /predict`
Predict sleep disorder based on input features.

**Request Body:**
```json
{
  "gender": "Male",
  "age": 30,
  "occupation": "Software Engineer",
  "sleep_duration": 7.5,
  "quality_of_sleep": 8,
  "physical_activity_level": 60,
  "stress_level": 4,
  "bmi_category": "Normal",
  "blood_pressure_systolic": 120,
  "blood_pressure_diastolic": 80,
  "heart_rate": 70,
  "daily_steps": 8000
}
```

**Response:**
```json
{
  "prediction": "None",
  "confidence": 0.95,
  "probabilities": {
    "None": 0.95,
    "Sleep Apnea": 0.03,
    "Insomnia": 0.02
  }
}
```

#### `GET /health`
Check API health status.

#### `GET /`
Get API information and endpoints.

## 🔧 Recent Bug Fixes

### Fixed: Empty Feature Array Error (v1.1.0)
**Issue**: The prediction endpoint was returning "Found array with 0 feature(s)" error.

**Root Cause**: The preprocessing pipeline was trying to select features before encoding categorical variables, resulting in an empty feature array.

**Solution**: Reordered the preprocessing steps to:
1. Handle missing values
2. Convert numeric columns  
3. Create engineered features
4. **Encode categorical variables FIRST**
5. **Then select features**
6. Apply scaling
7. Make prediction

**Status**: ✅ Fixed and tested - predictions now work correctly with proper feature engineering.

---

## 🚀 Deployment

### Backend Deployment (Render)

1. **Push to GitHub**: Ensure your latest code is pushed to your repository
2. **Deploy on Render**:
   - Connect your GitHub repository
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `python app.py`
   - Ensure `model.pkl` is in the repository root or backend folder

### Frontend Deployment (Vercel)

1. **Environment Variables**: Set `REACT_APP_API_URL` to your backend URL
2. **Deploy**: Connect your GitHub repository to Vercel
3. **Build Settings**: 
   - Build command: `npm run build`
   - Output directory: `build`

### Local Development

1. **Backend**: 
   ```bash
   cd backend
   PORT=5001 python3 app.py
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm start
   ```
   
   The frontend will automatically use `http://localhost:5001` for API calls via `.env.local`.

## 🚨 Important Disclaimers

### Medical Disclaimer
This application is for **educational and informational purposes only**. The predictions should not be considered as medical advice, diagnosis, or treatment recommendations. Always consult with qualified healthcare professionals for medical concerns.

### Accuracy Limitations
While our models achieve high accuracy on the training dataset, real-world performance may vary. Individual health conditions and factors not captured in the dataset may affect prediction accuracy.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Sleep Health and Lifestyle Dataset contributors
- Open source machine learning community
- React and Flask development teams

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section below
2. Review the documentation
3. Open an issue on GitHub

## 🔧 Troubleshooting

### Common Issues

**Model not found error:**
- Ensure you've run the Jupyter notebook completely
- Check that `model.pkl` exists in the root directory

**CORS errors:**
- Verify the backend is running on port 5000
- Check Flask-CORS is installed

**Frontend build errors:**
- Delete `node_modules` and run `npm install` again
- Ensure Node.js version is 16+

**Prediction errors:**
- Verify all required fields are provided
- Check data types match the expected format

---

**Built with ❤️ for better sleep health** 