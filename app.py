
import gradio as gr
import joblib
import numpy as np
import pandas as pd
import os


# Load trained model
model = joblib.load('my_model.pkl')


# Features expected by the trained model
expected_features = [
    'age',
    'hypertension',
    'heart_disease',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level',
    'gender_Male',
    'gender_Other',
    'smoking_history_current',
    'smoking_history_ever',
    'smoking_history_former',
    'smoking_history_never',
    'smoking_history_not current'
]


def predict(gender, age, hypertension, heart_disease,
            smoking_history, bmi, HbA1c_level,
            blood_glucose_level):

    # Create input DataFrame
    raw_input = pd.DataFrame([[
        gender,
        age,
        hypertension,
        heart_disease,
        smoking_history,
        bmi,
        HbA1c_level,
        blood_glucose_level
    ]], columns=[
        'gender',
        'age',
        'hypertension',
        'heart_disease',
        'smoking_history',
        'bmi',
        'HbA1c_level',
        'blood_glucose_level'
    ])


    # Define all possible categories
    # This ensures the same encoding as model training

    raw_input['gender'] = pd.Categorical(
        raw_input['gender'],
        categories=['Female', 'Male', 'Other']
    )

    raw_input['smoking_history'] = pd.Categorical(
        raw_input['smoking_history'],
        categories=[
            'No Info',
            'current',
            'ever',
            'former',
            'never',
            'not current'
        ]
    )


    # Apply one-hot encoding exactly as during training
    encoded_input = pd.get_dummies(
        raw_input,
        columns=['gender', 'smoking_history'],
        drop_first=True
    )


    # Align features with the trained model
    for col in expected_features:
        if col not in encoded_input.columns:
            encoded_input[col] = 0


    # Keep exactly the expected features in the correct order
    encoded_input = encoded_input[expected_features]

    # Convert features to numeric values
    encoded_input = encoded_input.astype(float)


    # Make prediction
    prediction = model.predict(encoded_input)[0]

    probability = model.predict_proba(encoded_input)[0]


    # Display prediction result
    if prediction == 1:

        return (
            f"☑  Diabetes Detected\n"
            f"Confidence: {probability[1]*100:.2f}%"
        )

    else:

        return (
            f"✅ No Diabetes Detected\n"
            f"Confidence: {probability[0]*100:.2f}%"
        )


# Build the dashboard
demo = gr.Interface(

    fn=predict,

    inputs=[

        gr.Radio(
            choices=['Female', 'Male', 'Other'],
            label="Gender",
            value='Male'
        ),

        gr.Number(
            label="Age (years)",
            value=50
        ),

        gr.Slider(
            0, 1,
            step=1,
            label="Hypertension",
            value=1
        ),

        gr.Radio(
            choices=[0, 1],
            label="Heart Disease",
            value=1
        ),

        gr.Radio(
            choices=[
                'current',
                'ever',
                'former',
                'never',
                'No Info',
                'not current'
            ],
            label="Smoking History",
            value='never'
        ),

        gr.Slider(
            10, 70,
            step=1,
            label="BMI",
            value=27
        ),

        gr.Slider(
            3, 9,
            label="HbA1c_level",
            value=5.5
        ),

        gr.Slider(
            80, 300,
            label="blood_glucose_level",
            value=140
        )
    ],

    outputs=gr.Textbox(
        label="Prediction Result",
        lines=3
    ),

    title="Diabetes Prediction Dashboard",

    description="Enter patient details and click Submit to predict diabetes."
)


# Launch application
demo.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)


# Create requirements.txt
with open('requirements.txt', 'w') as f:

    f.write("""gradio
scikit-learn
joblib
numpy
pandas
""")


print("✅ requirements.txt created!")


# Download all required files
from google.colab import files

files.download('app.py')
files.download('requirements.txt')
files.download('my_model.pkl')

print("✅ All 3 files downloaded!")


# Run application
!python app.py
