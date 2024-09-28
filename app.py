import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('best_model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Feature names
feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Map categorical features
def map_categorical_features(input_data, feature_names):
    mapping = {
        "sex": {"Male": 1, "Female": 0},
        "cp": {"Typical Angina": 1, "Atypical Angina": 2, "Non-Anginal": 3, "Asymptomatic": 4},
        "fbs": {"Yes": 1, "No": 0},
        "restecg": {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2},
        "exang": {"Yes": 1, "No": 0},
        "slope": {"Upsloping": 1, "Flat": 2, "Downsloping": 3},
        "thal": {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7},
    }

    mapped_data = {}
    for i, feature in enumerate(feature_names):
        if feature in mapping:
            mapped_data[feature] = mapping[feature][input_data[i][0]]
        else:
            mapped_data[feature] = input_data[i]

    return mapped_data

# Predict function
def predict_heart_disease(input_data):
    mapped_input = map_categorical_features(input_data, feature_names)
    input_df = pd.DataFrame([mapped_input])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction[0]

# Streamlit App UI
st.title("Heart Risk Prediction")

# Sidebar for Usage Instructions
with st.sidebar:
    st.header("Usage")
    st.write("""
    ### How to Use
    1. **Input Your Data**: 
       - Fill in the required fields with your personal health information. This includes:
         - **Age**: Your age in years.
         - **Gender**: Select your gender (Male or Female).
         - **Resting Blood Pressure (mmHg)**: Enter your resting blood pressure reading.
         - **Serum Cholesterol (mg/dl)**: Input your cholesterol level.
         - **Fasting Blood Sugar**: Indicate whether your fasting blood sugar is greater than 120 mg/dl.
         - **Maximum Heart Rate Achieved**: Enter the highest heart rate you have reached.
         - **Exercise Induced Angina**: Select whether you experience chest pain during physical activity.
         - **ST Depression Induced by Exercise**: Input the ST (segment) depression value after exercise.
         - **Chest Pain Type**: Choose the type of chest pain you experience.
         - **Resting ECG Results**: Select your resting ECG (electrocardiogram) results.
         - **Slope of Peak Exercise ST Segment**: Indicate the slope of your peak exercise ST segment.
         - **Number of Major Vessels**: Enter the number of major vessels colored by fluoroscopy (0-3).
         - **Thalassemia**: Select the type of thalassemia you have.

    2. **Submit Your Data**: 
       - Once all fields are filled out, click the **"Predict"** button. 

    3. **View Your Results**: 
       - The app will analyze your input and provide feedback on your risk of heart disease:
         - A message indicating **"Low risk of heart disease."** or **"High risk of heart disease."** will be displayed based on your inputs.

    4. **Make Informed Decisions**: 
       - Use the information provided by the app to discuss your health with a healthcare professional and consider necessary lifestyle changes or screenings.

    ### Important Notes
    - The app is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    - Ensure that all input fields are filled out correctly to receive an accurate prediction.
    """)

# Main content
st.header("Introduction")
st.write("""
Welcome to the **Heart Risk Prediction**! 
This application leverages machine learning to assess the likelihood of heart disease based on various health indicators. By inputting key health metrics, the app can provide insights into your heart health, helping you make informed decisions regarding your well-being.
""")

st.write("""
Heart disease remains one of the leading causes of mortality worldwide. Early detection and awareness are crucial for effective management and prevention. This app is designed to offer a user-friendly interface for individuals to input their health data and receive a risk assessment based on their profile.
""")
st.write("""
##### **Please read the usage on the left side bar before using!**
""")
st.text("")
# Input layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=25, help="Enter your age in years.")
    sex = st.selectbox("Gender", options=["Select", ("Male", 1), ("Female", 0)], format_func=lambda x: x[0] if isinstance(x, tuple) else x, help="Select your gender.")
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=300, value=120, help="Enter your resting blood pressure in mmHg.")
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, help="Enter your cholesterol level in mg/dl.")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["Select", ("Yes", 1), ("No", 0)], format_func=lambda x: x[0] if isinstance(x, tuple) else x, help="Select 'Yes' if your fasting blood sugar is greater than 120 mg/dl.")

with col2:
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150, help="Enter the maximum heart rate you have achieved.")
    exang = st.selectbox("Exercise Induced Angina?", options=["Select", ("Yes", 1), ("No", 0)], format_func=lambda x: x[0] if isinstance(x, tuple) else x, help="Select 'Yes' if you experience chest pain during physical activity.")
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, help="Enter ST depression value (difference in ST segment after exercise).")
    cp = st.selectbox("Chest Pain Type", options=["Select", ("Typical Angina", 1), ("Atypical Angina", 2), ("Non-Anginal", 3), ("Asymptomatic", 4)], format_func=lambda x: x[0] if isinstance(x, tuple) else x, help="Select the type of chest pain you experience.")
    restecg = st.selectbox("Resting ECG Results", options=["Select", ("Normal", 0), ("ST-T Abnormality", 1), ("Left Ventricular Hypertrophy", 2)], format_func=lambda x: x[0] if isinstance(x, tuple) else x, help="Select your resting ECG results.")

# Additional inputs
slope = st.selectbox("Slope of Peak Exercise ST Segment", options=["Select", ("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)], format_func=lambda x: x[0] if isinstance(x, tuple) else x, help="Select the slope of the peak exercise ST segment.")
ca = st.number_input("Number of Major Vessels (0-3) Colored by Fluoroscopy", min_value=0, max_value=3, value=0, help="Number of major blood vessels colored by fluoroscopy (0-3).")
thal = st.selectbox("Thalassemia", options=["Select", ("Normal", 3), ("Fixed Defect", 6), ("Reversible Defect", 7)], format_func=lambda x: x[0] if isinstance(x, tuple) else x, help="Select the type of thalassemia.")

# Input data
input_data = [
    age,
    sex if isinstance(sex, tuple) else None,
    cp if isinstance(cp, tuple) else None,
    trestbps,
    chol,
    fbs if isinstance(fbs, tuple) else None,
    restecg if isinstance(restecg, tuple) else None,
    thalach,
    exang if isinstance(exang, tuple) else None,
    oldpeak,
    slope if isinstance(slope, tuple) else None,
    ca,
    thal if isinstance(thal, tuple) else None,
]

# Predict button
if st.button("Predict"):
    # Check for unselected dropdowns
    if None in input_data:
        st.error("Please make sure all fields are filled out correctly.")
    else:
        try:
            result = predict_heart_disease(input_data)
            if result == 0:
                st.success("Low risk of heart disease.")
            else:
                st.success("High risk of heart disease.")
        except ValueError as e:
            st.error(f"Error during prediction: {e}")

st.text("")
st.write("""
    ### Important Notes
    - The app is **`NOT`** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    - Ensure that all input fields are filled out correctly to receive an accurate prediction.
    """)
st.markdown("""---""")
st.write("""
### Portfolio
For more projects and insights, please visit my [portfolio](https://kimnguyen2002.github.io/Portfolio/).

### Code Repository
You can find the complete code for this application in my GitHub repository [here]().
""")