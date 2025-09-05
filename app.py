import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="💰 Employee Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💰 Employee Salary Prediction Dashboard")
st.markdown("---")

# Sidebar for model information
st.sidebar.header("📊 Model Information")
st.sidebar.info("""
**Model Performance:**
- Model: Linear Regression
- MAE: ₹2.91L
- R² Score: 0.31
- Features: Experience, Department, Age, Performance
- Improvement over baseline: 16.4%
""")

st.sidebar.markdown("---")
st.sidebar.header("📈 About the Model")
st.sidebar.write("""
This model predicts employee salaries based on:
- **Years of Experience** (strongest predictor)
- **Department** (significant impact)
- **Age** (moderate influence)
- **Performance Score** (1-5 rating)
""")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Salary Prediction")
    st.write("Enter employee details below to predict their expected salary:")

    
    # Input form
    with st.form("prediction_form"):
        col_input1, col_input2 = st.columns(2)
        
        with col_input1:
            years = st.number_input(
                "📅 Years of Experience", 
                value=5, step=1, min_value=0, max_value=40,
                help="Total years of professional experience"
            )
            age = st.number_input(
                "🎂 Age", 
                value=28, step=1, min_value=18, max_value=65,
                help="Current age of the employee"
            )
        
        with col_input2:
            performance = st.selectbox(
                "⭐ Performance Score", 
                [1, 2, 3, 4, 5],
                index=2,
                help="Performance rating: 1=Poor, 3=Average, 5=Excellent"
            )
            department = st.selectbox(
                "🏢 Department", 
                ["HR", "IT", "Finance", "Sales", "Marketing", "Operations", "R&D", "Admin"],
                index=1,
                help="Employee's department"
            )
        
        # Submit button
        predict_btn = st.form_submit_button("🔮 Predict Salary", use_container_width=True)

with col2:
    st.header("💡 Quick Tips")
    st.info("""
    **Salary Drivers:**
    - IT & R&D pay highest
    - Experience matters most
    - Performance impacts growth
    - Age has moderate effect
    """)
    
    # Compact department salary ranges in expander
    with st.expander("📊 View Department Salary Ranges"):
        dept_ranges = {
            "IT": "₹4.5L - ₹20L+", "R&D": "₹5L - ₹20L+", 
            "Finance": "₹4L - ₹12L", "Sales": "₹3.5L - ₹10L",
            "Marketing": "₹3.8L - ₹10L", "HR": "₹3L - ₹7L",
            "Operations": "₹3.2L - ₹6L", "Admin": "₹2.8L - ₹5L"
        }
        
        # Display in two columns for compactness
        col_dept1, col_dept2 = st.columns(2)
        dept_items = list(dept_ranges.items())
        
        with col_dept1:
            for dept, range_val in dept_items[:4]:
                st.write(f"**{dept}**: {range_val}")
        with col_dept2:
            for dept, range_val in dept_items[4:]:
                st.write(f"**{dept}**: {range_val}")

# Load model and setup
@st.cache_resource
def load_model():
    return joblib.load("salary_prediction_model.pkl")

@st.cache_resource  
def setup_label_encoder():
    departments = ["Admin", "Finance", "HR", "IT", "Marketing", "Operations", "R&D", "Sales"]
    le = LabelEncoder()
    le.fit(departments)
    return le

model = load_model()
le = setup_label_encoder()

# Enhanced Indian number formatting
def format_inr(number):
    """Format number in Indian currency style (Lakhs/Crores)"""
    if number >= 1e7:  # 1 crore and above
        return f"₹{number/1e7:.1f} Cr"
    elif number >= 1e5:  # 1 lakh and above  
        return f"₹{number/1e5:.1f} L"
    else:
        return f"₹{number:,.0f}"

# Prediction and results
if predict_btn:
    with st.spinner("🔄 Calculating salary prediction..."):
        # Encode the department
        dept_encoded = le.transform([department])[0]
        
        # Create feature array: [YearsExperience, Department, Age, PerformanceScore]
        X = np.array([[years, dept_encoded, age, performance]])
        
        # Make prediction
        predicted_salary = model.predict(X)[0]
        
        # Display results
        st.success("✅ Prediction Complete!")
        
        # Main prediction display
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.metric(
                "💰 Predicted Salary", 
                format_inr(predicted_salary),
                help="Estimated annual salary based on provided inputs"
            )
        
        with col_result2:
            monthly_salary = predicted_salary / 12
            st.metric(
                "📅 Monthly Salary", 
                format_inr(monthly_salary),
                help="Estimated monthly salary"
            )
            
        with col_result3:
            # Calculate percentile (rough estimate)
            if predicted_salary < 400000:
                percentile = "Bottom 25%"
                color = "🔴"
            elif predicted_salary < 600000:
                percentile = "25-50%"
                color = "🟡"
            elif predicted_salary < 1000000:
                percentile = "50-75%"
                color = "🟢"
            else:
                percentile = "Top 25%"
                color = "🟢"
                
            st.metric(
                "📊 Salary Band", 
                f"{color} {percentile}",
                help="Estimated salary percentile in the market"
            )
        
        st.markdown("---")
        
        # Additional insights
        st.subheader("🔍 Prediction Insights")
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            st.write("**📋 Input Summary:**")
            st.write(f"• **Experience**: {years} years")
            st.write(f"• **Department**: {department}")
            st.write(f"• **Age**: {age} years")
            st.write(f"• **Performance**: {performance}/5")
            
        with col_insight2:
            st.write("**💡 Salary Factors:**")
            if years < 2:
                exp_impact = "Entry level - room for growth"
            elif years < 5:
                exp_impact = "Junior level - building experience"
            elif years < 10:
                exp_impact = "Mid-level - solid experience"
            else:
                exp_impact = "Senior level - extensive experience"
                
            dept_impact = {
                "IT": "High-paying tech sector",
                "R&D": "Innovation premium",
                "Finance": "Stable, good growth",
                "Sales": "Performance-driven",
                "Marketing": "Creative sector",
                "HR": "Support function",
                "Operations": "Operational efficiency",
                "Admin": "Administrative support"
            }
            
            st.write(f"• **Experience**: {exp_impact}")
            st.write(f"• **Department**: {dept_impact.get(department, 'Various factors')}")
            st.write(f"• **Performance**: {'Excellent' if performance >= 4 else 'Good' if performance >= 3 else 'Needs improvement'}")
        
        # Confidence and model info
        st.info(f"""
        **📈 Model Confidence**: This prediction is based on a Linear Regression model trained on 200,000 employee records.
        **🎯 Accuracy**: Mean Absolute Error of ₹2.91L (Model is 16.4% better than baseline)
        **⚠️ Note**: Actual salaries may vary based on company size, location, specific skills, and market conditions.
        """)
        
        # Balloons for celebration
        st.balloons()

else:
    st.info("👆 Please fill in the employee details above and click 'Predict Salary' to see the estimated compensation.")
    
    # Sample predictions showcase
    st.subheader("📋 Example Predictions")
    sample_data = {
        "Profile": ["Fresh IT Graduate", "Mid-level Finance", "Senior R&D", "Sales Manager"],
        "Experience": ["0 years", "5 years", "10 years", "8 years"],
        "Department": ["IT", "Finance", "R&D", "Sales"],
        "Estimated Salary": ["₹4.5L", "₹8.2L", "₹15.6L", "₹11.3L"]
    }
    
    df_samples = pd.DataFrame(sample_data)
    st.table(df_samples)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    💼 Employee Salary Prediction System | Built with Streamlit & Machine Learning<br>
    📊 Model trained on 200,000+ synthetic employee records | 🎯 For demonstration purposes
</div>
""", unsafe_allow_html=True)