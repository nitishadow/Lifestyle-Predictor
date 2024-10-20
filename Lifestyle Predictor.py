from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)


# Load the dataset and preprocess
df = pd.read_csv('Lifestyle Data.csv')
df = pd.get_dummies(df, columns=['Gender', 'Stress_Level'], drop_first=True)
X = df.drop('Healthy_Lifestyle_Score', axis=1).values
y = df['Healthy_Lifestyle_Score'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the RandomForest model function
def RandomForest(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    print(f'Cross-validated R2 Score: {np.mean(r2_scores) * 100:.2f}%')

    model.fit(X, y)
    return model

# Initialize the model
rfmodel = RandomForest(X, y)

# Create the Flask application
app = Flask(__name__)

# Define suggestion function (same as in your script)
def give_suggestions(age, gender, daily_steps, calories_consumed, sleep_hours, water_intake_liters, stress_level, exercise_hours, bmi):
    suggestions = []

    if gender == 'Male':

        if age < 30:
            if daily_steps < 8800:
                suggestions.append("Aim for a few extra steps—every little bit counts!")
            if daily_steps > 8800:
                suggestions.append("You’re active, but a few more steps never hurt!")
            if not (2400 <= calories_consumed <= 3000):
                suggestions.append("Focus on balanced meals with more whole foods.")
            if (2400 <= calories_consumed <= 3000):
                suggestions.append("You’re fueling your body well—keep up the balanced diet.")
            if not (7 <= sleep_hours <= 9):
                suggestions.append("Work on a bedtime routine to improve your sleep quality.")
            if (7 <= sleep_hours <= 9):
                suggestions.append("Amazing! You are sticking to a regular sleep schedule.")
            if water_intake_liters < 2.5:
                suggestions.append("Drink more water—start with 6-8 glasses a day.")
            if water_intake_liters >= 2.5:
                suggestions.append("Fantastic! It is good to keep up with your hydration routine.")
            if exercise_hours < 0.5:
                suggestions.append("Try to start with 10-15 minutes of light activity each day.")
            if exercise_hours >= 0.5:
                suggestions.append("Great consistency! You can add variety to keep things exciting!")
            if not (18.5 < bmi < 23):
                suggestions.append("It’s time to work on balance—small adjustments in your diet and activity will help.")
            if (18.5 < bmi < 23):
                suggestions.append("You’re in a healthy range—keep maintaining it with your current routine.")


        elif 30 <= age <= 54:
            if daily_steps < 5000:
                suggestions.append("Aim for a few extra steps—every little bit counts!")
            if daily_steps >= 5000:
                suggestions.append("You’re active, but a few more steps never hurt!")
            if not (2200 <= calories_consumed <= 3000):
                suggestions.append("Focus on balanced meals with more whole foods.")
            if (2200 <= calories_consumed <= 3000):
                suggestions.append("You’re fueling your body well—keep up the balanced diet.")
            if not (7 <= sleep_hours <= 9):
                suggestions.append("Work on a bedtime routine to improve your sleep quality.")
            if (7 <= sleep_hours <= 9):
                suggestions.append("Amazing! You are sticking to a regular sleep schedule")
            if water_intake_liters < 2.5:
                suggestions.append("Drink more water—start with 6-8 glasses a day.")
            if water_intake_liters >= 2.5:
                suggestions.append("Fantastic! It is good to keep up with your hydration routine.")
            if exercise_hours < 0.5:
                suggestions.append("Try to start with 10-15 minutes of light activity each day.")
            if exercise_hours >= 0.5:
                suggestions.append("Great consistency! You can add variety to keep things exciting!")
            if not (18.5 < bmi < 23):
                suggestions.append(
                    "It’s time to work on balance—small adjustments in your diet and activity will help.")
            if (18.5 < bmi < 23):
                suggestions.append("You’re in a healthy range—keep maintaining it with your current routine.")


        elif age > 54:
            if daily_steps < 6500:
                suggestions.append("Aim for a few extra steps—every little bit counts!")
            if daily_steps >= 6500:
                suggestions.append("You’re active, but a few more steps never hurt!")
            if not (2000 <= calories_consumed <= 2600):
                suggestions.append("Focus on balanced meals with more whole foods.")
            if (2000 <= calories_consumed <= 2600):
                suggestions.append("You’re fueling your body well—keep up the balanced diet.")
            if not (7 <= sleep_hours <= 8):
                suggestions.append("Work on a bedtime routine to improve your sleep quality.")
            if (7 <= sleep_hours <= 8):
                suggestions.append("Amazing! You are sticking to a regular sleep schedule.")
            if water_intake_liters < 2.5:
                suggestions.append("Drink more water—start with 6-8 glasses a day.")
            if water_intake_liters >= 2.5:
                suggestions.append("Fantastic! It is good to keep up with your hydration routine.")
            if exercise_hours < 0.5:
                suggestions.append("Try to start with 10-15 minutes of light activity each day.")
            if exercise_hours >= 0.5:
                suggestions.append("Great consistency! You can add variety to keep things exciting!")
            if not (18.5 < bmi < 23):
                suggestions.append(
                    "It’s time to work on balance—small adjustments in your diet and activity will help.")
            if (18.5 < bmi < 23):
                suggestions.append("You’re in a healthy range—keep maintaining it with your current routine.")

    elif gender == 'Female':

        if age < 30:
            if daily_steps < 8000:
                suggestions.append("Aim for a few extra steps—every little bit counts!")
            if daily_steps >= 8000:
                suggestions.append("You’re active, but a few more steps never hurt!")
            if not (2000 <= calories_consumed <= 2300):
                suggestions.append("Focus on balanced meals with more whole foods.")
            if (2000 <= calories_consumed <= 2300):
                suggestions.append("You’re fueling your body well—keep up the balanced diet.")
            if not (7 <= sleep_hours <= 9):
                suggestions.append("Work on a bedtime routine to improve your sleep quality.")
            if (7 <= sleep_hours <= 9):
                suggestions.append("Amazing! You are sticking to a regular sleep schedule")
            if water_intake_liters < 2.5:
                suggestions.append("Drink more water—start with 6-8 glasses a day.")
            if water_intake_liters >= 2.5:
                suggestions.append("Fantastic! It is good to keep up with your hydration routine.")
            if exercise_hours < 0.5:
                suggestions.append("Try to start with 10-15 minutes of light activity each day.")
            if exercise_hours >= 0.5:
                suggestions.append("Great consistency! You can add variety to keep things exciting!")
            if not (18.5 < bmi < 23):
                suggestions.append(
                    "It’s time to work on balance—small adjustments in your diet and activity will help.")
            if (18.5 < bmi < 23):
                suggestions.append("You’re in a healthy range—keep maintaining it with your current routine.")


        elif 30 <= age <= 54:
            if daily_steps < 5500:
                suggestions.append("Aim for a few extra steps—every little bit counts!")
            if daily_steps >= 5500:
                suggestions.append("You’re active, but a few more steps never hurt!")
            if not (1800 <= calories_consumed <= 2100):
                suggestions.append("Focus on balanced meals with more whole foods.")
            if (1800 <= calories_consumed <= 2100):
                suggestions.append("You’re fueling your body well—keep up the balanced diet.")
            if not (7 <= sleep_hours <= 9):
                suggestions.append("Work on a bedtime routine to improve your sleep quality.")
            if (7 <= sleep_hours <= 9):
                suggestions.append("Amazing! You are sticking to a regular sleep schedule.")
            if water_intake_liters < 2.5:
                suggestions.append("Drink more water—start with 6-8 glasses a day.")
            if water_intake_liters >= 2.5:
                suggestions.append("Fantastic! It is good to keep up with your hydration routine.")
            if exercise_hours < 0.5:
                suggestions.append("Try to start with 10-15 minutes of light activity each day.")
            if exercise_hours >= 0.5:
                suggestions.append("Great consistency! You can add variety to keep things exciting!")
            if not (18.5 < bmi < 23):
                suggestions.append("It’s time to work on balance—small adjustments in your diet and activity will help.")
            if (18.5 < bmi < 23):
                suggestions.append("You’re in a healthy range—keep maintaining it with your current routine.")


        elif age > 54:
            if daily_steps < 7000:
                suggestions.append("Aim for a few extra steps—every little bit counts!")
            if daily_steps >= 7000:
                suggestions.append("You’re active, but a few more steps never hurt!")
            if not (1800 <= calories_consumed <= 2100):
                suggestions.append("Focus on balanced meals with more whole foods.")
            if (1800 <= calories_consumed <= 2100):
                suggestions.append("You’re fueling your body well—keep up the balanced diet.")
            if not (7 <= sleep_hours <= 9):
                suggestions.append("Work on a bedtime routine to improve your sleep quality.")
            if (7 <= sleep_hours <= 9):
                suggestions.append("Amazing! You are sticking to a regular sleep schedule.")
            if water_intake_liters < 2.5:
                suggestions.append("Drink more water—start with 6-8 glasses a day.")
            if water_intake_liters >= 2.5:
                suggestions.append("Fantastic! It is good to keep up with your hydration routine.")
            if exercise_hours < 0.5:
                suggestions.append("Try to start with 10-15 minutes of light activity each day.")
            if exercise_hours >= 0.5:
                suggestions.append("Great consistency! You can add variety to keep things exciting!")
            if not (18.5 < bmi < 23):
                suggestions.append("It’s time to work on balance—small adjustments in your diet and activity will help.")
            if (18.5 < bmi < 23):
                suggestions.append("You’re in a healthy range—keep maintaining it with your current routine.")

    if stress_level == 'High':
        suggestions.append("Why so Tensed? Consider stress-relief techniques like meditation or exercise.")

    return suggestions

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        # Parse input data
        age = float(data['age'])
        gender = data['gender']
        height = float(data['height'])
        weight = float(data['weight'])
        daily_steps = float(data['daily_steps'])
        calories_consumed = float(data['calories_consumed'])
        exercise_hours = float(data['exercise_hours'])
        water_intake_liters = float(data['water_intake_liters'])
        sleep_hours = float(data['sleep_hours'])
        stress_level = data['stress_level']

        bmi = weight / ((height / 100) * (height / 100))

        # One-hot encoding for gender and stress level
        gender_male = 1 if gender == 'Male' else 0
        stress_level_medium = 1 if stress_level == 'Medium' else 0
        stress_level_high = 1 if stress_level == 'High' else 0

        # Create DataFrame for the user's input
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

        # Scale the input
        user_input = scaler.transform(user_df.values)

        # Generate suggestions
        suggestions = give_suggestions(age, gender, daily_steps, calories_consumed, sleep_hours, water_intake_liters, stress_level, exercise_hours, bmi)

        # Predict healthy lifestyle score
        prediction = rfmodel.predict(user_input)

        # Return the response
        return jsonify({
            'predicted_score': float(prediction[0]),
            'suggestions': suggestions
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {e}'})

# Run the app on port 4000
if __name__ == '__main__':
    app.run(port=4000, debug=True)
