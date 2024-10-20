import pandas as pd
from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'diabetes_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the saved model
        model = joblib.load(model_path)

        # Get user input from the form
        features = [float(request.form['Pregnancies']),
                    float(request.form['Glucose']),
                    float(request.form['BloodPressure']),
                    float(request.form['SkinThickness']),
                    float(request.form['Insulin']),
                    float(request.form['BMI']),
                    float(request.form['DiabetesPedigreeFunction']),
                    float(request.form['Age'])]

        # Make a prediction
        prediction = model.predict([features])

        # Determine the result
        result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
