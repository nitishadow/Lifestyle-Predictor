from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load and preprocess the data
df = pd.read_csv('Lifestyle Data.csv')
df = pd.get_dummies(df, columns=['Gender', 'Stress_Level'], drop_first=True)
X = df.drop('Healthy_Lifestyle_Score', axis=1).values
y = df['Healthy_Lifestyle_Score'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Function to train the model
def RandomForest(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    print(f'Cross-validated R2 Score: {np.mean(r2_scores) * 100:.2f}%')

    model.fit(X, y)
    return model

# Train the model
rfmodel = RandomForest(X, y)

# Endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request (JSON format expected)
        data = request.json

        # Extract the values from the input
        age = float(data['age'])
        gender = data['gender'].strip().capitalize()
        daily_steps = float(data['daily_steps'])
        calories_consumed = float(data['calories_consumed'])
        sleep_hours = float(data['sleep_hours'])
        water_intake_liters = float(data['water_intake_liters'])
        stress_level = data['stress_level'].strip().capitalize()
        exercise_hours = float(data['exercise_hours'])
        bmi = float(data['bmi'])

        # Validate gender and stress level input
        if gender not in ['Male', 'Female']:
            return jsonify({"error": "Invalid gender input. Please enter 'Male' or 'Female'."}), 400
        if stress_level not in ['Low', 'Medium', 'High']:
            return jsonify({"error": "Invalid stress level input. Please enter 'Low', 'Medium', or 'High'."}), 400

        # One-hot encode categorical values
        gender_male = 1 if gender == 'Male' else 0
        stress_level_medium = 1 if stress_level == 'Medium' else 0
        stress_level_high = 1 if stress_level == 'High' else 0

        # Create a DataFrame for the input
        user_df = pd.DataFrame({
            'Age': [age],
            'Daily_Steps': [daily_steps],
            'Calories_Consumed': [calories_consumed],
            'Sleep_Hours': [sleep_hours],
            'Water_Intake_Liters': [water_intake_liters],
            'Exercise_Hours': [exercise_hours],
            'BMI': [bmi],
            'Gender_Male': [gender_male],
            'Stress_Level_Medium': [stress_level_medium],
            'Stress_Level_High': [stress_level_high]
        })

        # Scale the input data
        user_input = scaler.transform(user_df.values)

        # Make a prediction
        prediction = rfmodel.predict(user_input)

        # Return the prediction as a JSON response
        return jsonify({"Predicted_Healthy_Lifestyle_Score": prediction[0]})

    except ValueError as e:
        return jsonify({"error": f"Input error: {str(e)}"}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=4000)