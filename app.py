import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.title("Salary Prediction App")
st.divider()
st.write("With this app, you can predict the salary based on multiple factors.")

# Input fields for all 4 features
years = st.number_input("Enter the years of experience", value=1, step=1, min_value=0)
age = st.number_input("Enter your age", value=25, step=1, min_value=18, max_value=65)
performance = st.selectbox("Select performance score", [1, 2, 3, 4, 5])
department = st.selectbox("Select department", 
                         ["HR", "IT", "Finance", "Sales", "Marketing", "Operations", "R&D", "Admin"])

# Load the model and label encoder
model = joblib.load("salary_prediction_model.pkl")

# You'll need to save and load the label encoder too
# For now, recreate it with the same departments
departments = ["Admin", "Finance", "HR", "IT", "Marketing", "Operations", "R&D", "Sales"]
le = LabelEncoder()
le.fit(departments)

st.divider()
predict_btn = st.button("Predict Salary")
st.divider()

# Helper function for Indian number format
def format_inr(number):
    num_str = str(int(number))[::-1]  # Reverse string
    parts = []
    parts.append(num_str[:3])  # First 3 digits
    num_str = num_str[3:]
    while num_str:
        parts.append(num_str[:2])  # Next 2 digits
        num_str = num_str[2:]
    return ','.join(parts)[::-1]

if predict_btn:
    st.balloons()
    
    # Encode the department
    dept_encoded = le.transform([department])[0]
    
    # Create feature array: [YearsExperience, Department, Age, PerformanceScore]
    X = np.array([[years, dept_encoded, age, performance]])
    
    predict = model.predict(X)[0]
    st.write(f"The predicted salary is INR {format_inr(predict)}")
else:
    st.write("Please press the predict salary button to see the result.")