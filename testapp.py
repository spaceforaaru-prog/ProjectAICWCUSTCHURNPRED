import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression


# =========================================================
# LOAD MODEL AND COLUMNS
# =========================================================

model = joblib.load("customer_churn_model.pkl")
columns = joblib.load("customer_churn_columns.pkl")


# Check if the saved model is Logistic Regression
# Logistic Regression needs the scaler
if isinstance(model, LogisticRegression):

    scaler = joblib.load("customer_churn_scaler.pkl")
    scaler_available = True

else:

    scaler_available = False


# =========================================================
# PAGE SETTINGS
# =========================================================



st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")


# =========================================================
# WELCOME BOX DESIGN
# =========================================================

st.markdown("""
<style>

.welcome-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #f3f4f6;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# WELCOME BUTTON
# =========================================================

if st.button("✨ Welcome to My App", use_container_width=True):

    st.session_state.welcome = True


if "welcome" in st.session_state and st.session_state.welcome:

    st.markdown("""
    <div class="welcome-box">

    <h2>👋 Welcome!</h2>

    <p>
    Welcome to our Customer Churn Prediction App!
    </p>

    <p>
    Enter the customer's information below and our
    Machine Learning model will estimate their
    probability of churning.
    </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.header("👤 Customer Information")


st.write(
    "Enter the customer's details below to predict "
    "whether they are likely to churn."
)


# =========================================================
# CUSTOMER INFORMATION
# =========================================================



gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)


# =========================================================
# PHONE SERVICES
# =========================================================

st.header("Phone Services")

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)


# =========================================================
# INTERNET SERVICES
# =========================================================

st.header("Internet Services")

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)


# =========================================================
# CONTRACT AND BILLING
# =========================================================

st.header("Contract & Billing")

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


# =========================================================
# CHARGES
# =========================================================

st.header("Charges")

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("🔮 Predict Churn", use_container_width=True):

    # -----------------------------------------------------
    # Create customer data
    # -----------------------------------------------------

    customer = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [senior_citizen],

        "Partner": [partner],

        "Dependents": [dependents],

        "tenure": [tenure],

        "PhoneService": [phone_service],

        "MultipleLines": [multiple_lines],

        "InternetService": [internet_service],

        "OnlineSecurity": [online_security],

        "OnlineBackup": [online_backup],

        "DeviceProtection": [device_protection],

        "TechSupport": [tech_support],

        "StreamingTV": [streaming_tv],

        "StreamingMovies": [streaming_movies],

        "Contract": [contract],

        "PaperlessBilling": [paperless_billing],

        "PaymentMethod": [payment_method],

        "MonthlyCharges": [monthly_charges],

        "TotalCharges": [total_charges]
    })


    # -----------------------------------------------------
    # Convert categorical values into numbers
    # -----------------------------------------------------

    customer = pd.get_dummies(
        customer,
        drop_first=True
    )


    # -----------------------------------------------------
    # Make columns exactly the same as training data
    # -----------------------------------------------------

    customer = customer.reindex(
        columns=columns,
        fill_value=0
    )


    # -----------------------------------------------------
    # MAKE PREDICTION
    # -----------------------------------------------------

    if scaler_available:

        # Logistic Regression
        # needs scaled data

        customer_scaled = scaler.transform(customer)

        prediction = model.predict(customer_scaled)

        probability = model.predict_proba(
            customer_scaled
        )[0][1]

    else:

        # Random Forest
        # does not need scaling

        prediction = model.predict(customer)

        probability = model.predict_proba(
            customer
        )[0][1]


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.header("Prediction Result")


    # Convert probability into percentage

    probability_percentage = probability * 100


    if prediction[0] == 1:

        st.error("⚠️ Customer is likely to CHURN")

    else:

        st.success("✅ Customer is likely to STAY")


    # -----------------------------------------------------
    # Display probability
    # -----------------------------------------------------

    st.metric(
        "Estimated Churn Probability",
        f"{probability_percentage:.4f}%"
    )


    # Probability bar

    st.progress(
        int(round(probability_percentage))
    )


    # -----------------------------------------------------
    # Explanation
    # -----------------------------------------------------

    if probability_percentage >= 70:

        st.write(
            "🔴 The model considers this customer to have "
            "a relatively high risk of churn."
        )

    elif probability_percentage >= 40:

        st.write(
            "🟡 The model considers this customer to have "
            "a moderate risk of churn."
        )

    else:

        st.write(
            "🟢 The model considers this customer to have "
            "a relatively low risk of churn."
        )

# ==============================
# EDA - Dataset Insights
# ==============================

st.markdown("---")
st.header("📊 Dataset Insights")

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Basic information
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    churned = (df["Churn"] == "Yes").sum()
    st.metric("Churned Customers", churned)

with col3:
    stayed = (df["Churn"] == "No").sum()
    st.metric("Customers Stayed", stayed)


# Churn Distribution
st.subheader("Customer Churn Distribution")

churn_counts = df["Churn"].value_counts()

fig, ax = plt.subplots()
ax.bar(churn_counts.index, churn_counts.values)
ax.set_xlabel("Churn")
ax.set_ylabel("Number of Customers")
ax.set_title("Churned vs Stayed Customers")

st.pyplot(fig)


# Contract Type vs Churn
st.subheader("Contract Type vs Churn")

contract_churn = pd.crosstab(df["Contract"], df["Churn"])

st.bar_chart(contract_churn)


# Monthly Charges
st.subheader("Monthly Charges")

st.write(
    "This shows the distribution of monthly charges paid by customers."
)

fig, ax = plt.subplots()
ax.hist(df["MonthlyCharges"], bins=20)
ax.set_xlabel("Monthly Charges")
ax.set_ylabel("Number of Customers")
ax.set_title("Distribution of Monthly Charges")

st.pyplot(fig)        


        
        
        
