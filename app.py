import pandas as pd
from flask import Flask, request, jsonify, render_template, abort
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__, template_folder='templates')

# Ensure the dataset exists
DATASET_PATH = "Healthcare-Diabetes.csv"
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please ensure the file is available.")

# Load and prepare the dataset
df = pd.read_csv(DATASET_PATH)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[features]
y = df['Outcome']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# Save the model
MODEL_PATH = 'diabetes_model.pkl'
joblib.dump(model, MODEL_PATH)

@app.route('/')
def home():
    """Render the main HTML page."""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading template: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get user input from the form
        features = [
            float(request.json.get('Pregnancies', 0)),
            float(request.json.get('Glucose', 0)),
            float(request.json.get('BloodPressure', 0)),
            float(request.json.get('SkinThickness', 0)),
            float(request.json.get('Insulin', 0)),
            float(request.json.get('BMI', 0)),
            float(request.json.get('DiabetesPedigreeFunction', 0)),
            float(request.json.get('Age', 0))
        ]
        # Make prediction
        prediction = model.predict([features])
        result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
