import streamlit as st
from utils import preprocess_input, model

st.set_page_config(page_title="Salary Predictor", layout="centered")

st.title(" Developer Salary Prediction App")
st.markdown("Estimate your expected salary based on your professional profile.")

EDUCATION_LEVELS = [
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)",
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)",
    "Other",
    "Professional degree (JD, MD, etc.)",
    "Some college/university study without earning a degree",
    "Associate degree (A.A., A.S., etc.)",
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
    "Primary/elementary school",
    "I never completed any formal education"
]

with st.form("prediction_form"):
    st.subheader("Enter your details")

    years_exp = st.number_input("Years of Professional Coding Experience", min_value=0, max_value=50, value=1)

    country = st.selectbox("Country", [
        'Brazil', 'Canada', 'France', 'Germany', 'India', 'Italy', 'Netherlands', 'Other',
        'Poland', 'Spain', 'Sweden', 'Switzerland', 'Ukraine',
        'United Kingdom of Great Britain and Northern Ireland', 'United States of America'
    ])

    education = st.selectbox("Education Level", EDUCATION_LEVELS)

    employment = st.multiselect("Employment Status (you can select multiple)", [
    'Employed, full-time',
    'Employed, part-time',
    'Independent contractor, freelancer, or self-employed',
    'Student, part-time',
    'Student, full-time',
    'Not employed, but looking for work',
    'Not employed, and not looking for work'
])

    remote = st.selectbox("Work Arrangement", [
        'Remote', 'In-person', 'Other'
    ])

    org_size = st.selectbox("Organization Size", [
        '10 to 19 employees', '100 to 499 employees', '500 to 999 employees',
        'Just me - I am a freelancer, sole proprietor, etc.', 'Other'
    ])

    submitted = st.form_submit_button("Predict Salary ")

if submitted:
    user_input = {
    'YearsCodePro': years_exp,
    'Country': country,
    'EdLevel': education,
    'Employment': ';'.join(employment),
    'RemoteWork': remote,
    'OrgSize': org_size
}

    X = preprocess_input(user_input)
    prediction = model.predict(X)[0]

    st.success(f"🎯 Estimated Annual Salary: **${prediction:,.2f} USD**")
