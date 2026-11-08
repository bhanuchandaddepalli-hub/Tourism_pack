
import os
import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------
# Load the trained model
# ------------------------------------------------

model_path = os.path.join(
    os.path.dirname(__file__),
    "best_tourism_model_v1.joblib"
)

model = joblib.load(model_path)

# ------------------------------------------------
# Streamlit App Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Tourism Package Purchase Prediction")

st.write("""
Welcome to the Visit with Us Tourism Package Prediction Application!

This application predicts whether a customer is likely to purchase
the Wellness Tourism Package based on their personal details,
travel preferences, and sales interaction information.

Enter the customer details below and click **Predict Purchase**
to generate a prediction.
""")

st.divider()

# ------------------------------------------------
# 1. Customer Personal Information
# ------------------------------------------------

st.subheader("1. Customer Personal Information")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=61,
        value=36
    )

with col2:
    Gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col3:
    MaritalStatus = st.selectbox(
        "Marital Status",
        ["Married", "Single", "Divorced"]
    )

col4, col5 = st.columns(2)

with col4:
    Occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Free Lancer",
         "Small Business", "Large Business"]
    )

with col5:
    Designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager",
         "AVP", "VP"]
    )

# ------------------------------------------------
# 2. Customer Location and Contact Details
# ------------------------------------------------

st.divider()
st.subheader("2. Location and Contact Information")

col1, col2 = st.columns(2)

with col1:
    CityTier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

with col2:
    TypeofContact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

# ------------------------------------------------
# 3. Travel Preferences
# ------------------------------------------------

st.divider()
st.subheader("3. Travel Preferences")

col1, col2, col3 = st.columns(3)

with col1:
    NumberOfPersonVisiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=5,
        value=3
    )

with col2:
    NumberOfChildrenVisiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=3,
        value=1
    )

with col3:
    NumberOfTrips = st.number_input(
        "Number of Trips Annually",
        min_value=1,
        max_value=22,
        value=3
    )

col4, col5 = st.columns(2)

with col4:
    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star Rating",
        [3, 4, 5]
    )

with col5:
    Passport = st.selectbox(
        "Does the Customer Have a Passport?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# ------------------------------------------------
# 4. Sales Interaction Details
# ------------------------------------------------

st.divider()
st.subheader("4. Sales Interaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    DurationOfPitch = st.number_input(
        "Duration of Pitch (Minutes)",
        min_value=5,
        max_value=127,
        value=14
    )

with col2:
    NumberOfFollowups = st.number_input(
        "Number of Follow-ups",
        min_value=1,
        max_value=6,
        value=4
    )

with col3:
    PitchSatisfactionScore = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5]
    )

col4, col5 = st.columns(2)

with col4:
    ProductPitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "Standard",
         "Super Deluxe", "King"]
    )

with col5:
    OwnCar = st.selectbox(
        "Does the Customer Own a Car?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# ------------------------------------------------
# 5. Customer Income
# ------------------------------------------------

st.divider()
st.subheader("5. Customer Financial Information")

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=98678,
    value=22418,
    step=100
)

# ------------------------------------------------
# 6. Prepare Input Data
# ------------------------------------------------

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

# ------------------------------------------------
# 7. Prediction
# ------------------------------------------------

st.divider()

if st.button("Predict Purchase", type="primary",
             use_container_width=True):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            "The customer is likely to purchase "
            "the Wellness Tourism Package! 🎉"
        )

    else:

        st.warning(
            "The customer is unlikely to purchase "
            "the Wellness Tourism Package."
        )

    # Display purchase probability if supported
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)[0]

        classes = list(model.classes_)
        purchase_probability = probabilities[classes.index(1)] * 100

        st.metric(
            "Purchase Probability",
            f"{purchase_probability:.2f}%"
        )

        st.progress(float(purchase_probability / 100))

    st.subheader("Customer Input Summary")
    st.dataframe(input_data, use_container_width=True)

st.divider()

st.caption(
    "Visit with Us | Wellness Tourism Package "
    "Purchase Prediction | Machine Learning & MLOps Project"
)
