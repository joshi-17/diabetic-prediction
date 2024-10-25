import pandas as pd
from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__)

# Load and prepare the dataset
df = pd.read_csv("Healthcare-Diabetes.csv")  # Ensure the path is correct for your environment
a = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[a]
y = df['Outcome']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'diabetes_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    features = [float(request.json['Pregnancies']),
                float(request.json['Glucose']),
                float(request.json['BloodPressure']),
                float(request.json['SkinThickness']),
                float(request.json['Insulin']),
                float(request.json['BMI']),
                float(request.json['DiabetesPedigreeFunction']),
                float(request.json['Age'])]
    
    prediction = model.predict([features])
    result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

    return jsonify({'prediction': result})  # Return JSON response
