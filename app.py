import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time

import warnings
warnings.filterwarnings('ignore')

st.write("## Personal Fitness Tracker")
#st.image("", use_column_width=True)
st.write("In this WebApp you will be able to observe your predicted calories burned in your body. Pass your parameters such as `Age`, `Gender`, `BMI`, etc., into this WebApp and then you will see the predicted value of kilocalories burned.")

st.sidebar.header("User Input Parameters: ")

def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    # Use column names to match the training data
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart Rate": heart_rate,
        "Gender_male": gender  # Gender is encoded as 1 for male, 0 for female
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

st.write("---")
st.header("Your Parameters: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Load and preprocess data
exercise = pd.read_csv("exercise_dataset.csv")
exercise.drop(columns="ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise, test_size=0.2, random_state=1)
# add exercise intensity column to both training and test sets
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart Rate", "Calories Burn"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart Rate", "Calories Burn"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate features and labels
X_train = exercise_train_data.drop("Calories Burn", axis=1)
y_train = exercise_train_data["Calories Burn"]

X_test = exercise_test_data.drop("Calories Burn", axis=1)
y_test = exercise_test_data["Calories Burn"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df)
# Exercise Intensity Calculation
df['Predicted Intensity'] = (df['Heart Rate'] * df['Duration']) / df['BMI']
# Define thresholds for intensity levels (adjustable)
low_intensity_threshold = 50  # Below this value is low intensity
moderate_intensity_threshold = 100  # Between this and high is moderate

# Categorize intensity level
def classify_intensity(value):
    if value < low_intensity_threshold:
        return "Low Intensity"
    elif value < moderate_intensity_threshold:
        return "Moderate Intensity"
    else:
        return "High Intensity"

df['Intensity Level'] = df['Predicted Intensity'].apply(classify_intensity)
# Ideal Exercise Suggestion
def suggest_exercise(bmi):
    if bmi < 18.5:
        return "Strength Training"
    elif 18.5 <= bmi < 25:
        return "Cardio + Strength"
    elif 25 <= bmi < 30:
        return "Cardio + Yoga"
    else:
        return "Walking + Low Impact"

df['Ideal Exercise'] = df['BMI'].apply(suggest_exercise)

st.write("---")
st.header("Prediction: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.write(f"Calories Burned: {round(prediction[0], 2)} **kilocalories**")
st.write(f"**{df['Intensity Level'][0]}** (Predicted Intensity: {round(df['Predicted Intensity'][0], 2)})")
st.write(f"Recommended Exercise: {df['Ideal Exercise'][0]}")

st.write("---")
st.header("Similar Results: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

# Find similar results based on predicted calories
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise[(exercise["Calories Burn"] >= calorie_range[0]) & (exercise["Calories Burn"] <= calorie_range[1])]
st.write(similar_data.sample(5))

st.write("---")
st.header("General Information: ")

# Boolean logic for age, duration, etc., compared to the user's input
boolean_age = (exercise["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise["Duration"] < df["Duration"].values[0]).tolist()
boolean_heart_rate = (exercise["Heart Rate"] < df["Heart Rate"].values[0]).tolist()

st.write("You are older than", round(sum(boolean_age) / len(boolean_age), 2) * 100, "% of other people.")
st.write("Your exercise duration is higher than", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other people.")
st.write("You have a higher heart rate than", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other people during exercise.")
