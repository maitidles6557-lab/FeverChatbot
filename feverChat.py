import streamlit as st
import pandas as pd
import nltk
from nltk.chat.util import Chat, reflections
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

st.set_page_config(
    page_title="Fever Medical Chatbot",
    page_icon="🌡️",
    layout="centered"
)

df = pd.read_csv("enhanced_fever_medicine_recommendation.csv")
target_column = "Recommended_Medication"

feature_columns = [
    column
    for column in df.columns
    if column != target_column
]
categorical_columns = [
    "Gender",
    "Fever_Severity",
    "Headache",
    "Body_Ache",
    "Fatigue",
    "Chronic_Conditions",
    "Allergies",
    "Smoking_History",
    "Alcohol_Consumption",
    "Physical_Activity",
    "Diet_Type",
    "Previous_Medication"
]

numerical_columns = [
    "Age",
    "BMI",
    "Temperature",
    "Humidity",
    "AQI",
    "Heart_Rate"
]
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)
X = df[feature_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
X_train_encoded = preprocessor.fit_transform(X_train)

X_test_encoded = preprocessor.transform(X_test)

model.fit(X_train_encoded, y_train)
y_pred = model.predict(X_test_encoded)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

st.text("Classification Report")
st.text(report)
def predict_medication(patient_data):
    patient_df = pd.DataFrame([patient_data])

    patient_encoded = preprocessor.transform(patient_df)

    prediction = model.predict(patient_encoded)

    return prediction[0]
test_patient = X_test.iloc[[0]]

test_prediction = model.predict(
    preprocessor.transform(test_patient)
)[0]

st.subheader("🔮 Test Prediction")

st.text(f"Predicted Medication: {test_prediction}")
st.subheader("🧑‍⚕️ Patient Information")

age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=25
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=60.0,
    value=22.0
)

temperature = st.number_input(
    "Temperature",
    min_value=30.0,
    max_value=45.0,
    value=37.0
)
fever_severity = st.selectbox(
    "Fever Severity",
    ["Mild", "Moderate", "High"]
)

headache = st.selectbox(
    "Headache",
    ["Yes", "No"]
)

body_ache = st.selectbox(
    "Body Ache",
    ["Yes", "No"]
)

fatigue = st.selectbox(
    "Fatigue",
    ["Yes", "No"]
)

chronic_conditions = st.text_input(
    "Chronic Conditions",
    value="None"
)

allergies = st.text_input(
    "Allergies",
    value="None"
)

smoking_history = st.selectbox(
    "Smoking History",
    ["Yes", "No"]
)

alcohol_consumption = st.selectbox(
    "Alcohol Consumption",
    ["Yes", "No"]
)

physical_activity = st.selectbox(
    "Physical Activity",
    ["Low", "Moderate", "High"]
)

diet_type = st.selectbox(
    "Diet Type",
    ["Vegetarian", "Non-Vegetarian", "Vegan"]
)

previous_medication = st.text_input(
    "Previous Medication",
    value="None"
)
humidity = st.number_input(
    "Humidity",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

aqi = st.number_input(
    "AQI",
    min_value=0.0,
    max_value=500.0,
    value=50.0
)

heart_rate = st.number_input(
    "Heart Rate",
    min_value=30.0,
    max_value=220.0,
    value=70.0
)
blood_pressure = st.text_input(
    "Blood Pressure",
    value="120/80"
)
patient_data = {
    "Age": age,
    "Gender": gender,
    "BMI": bmi,
    "Temperature": temperature,
    "Fever_Severity": fever_severity,
    "Headache": headache,
    "Body_Ache": body_ache,
    "Fatigue": fatigue,
    "Chronic_Conditions": chronic_conditions,
    "Allergies": allergies,
    "Smoking_History": smoking_history,
    "Alcohol_Consumption": alcohol_consumption,
    "Physical_Activity": physical_activity,
    "Diet_Type": diet_type,
    "Humidity": humidity,
    "AQI": aqi,
    "Heart_Rate": heart_rate,
    "Blood_Pressure": blood_pressure,
    "Previous_Medication": previous_medication
}
if st.button("🔮 Predict Medication"):

    patient_df = pd.DataFrame([patient_data])

    patient_encoded = preprocessor.transform(patient_df)

    prediction = model.predict(patient_encoded)[0]

    st.subheader("💊 Prediction")

    st.text(f"Predicted Medication: {prediction}")

    st.session_state["prediction"] = prediction
pairs = [
    [
        r"^(hi|hello|hey)(.*)",
        ["Welcome! How can I help you?"]
    ],
    [
        r"^(what medicine|which medicine|what medication)(.*)",
        ["You can check the predicted medication above."]
    ],
]

chatbot = Chat(pairs, reflections)

st.markdown("""
<style>

    /* ==================================
       Main Background
       ================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(30, 100, 160, 0.35),
                transparent 35%
            ),
            radial-gradient(
                circle at 90% 90%,
                rgba(20, 80, 130, 0.30),
                transparent 40%
            ),
            linear-gradient(
                135deg,
                #07111f,
                #0b1b2d,
                #07111f
            );

        color: white;
    }


    /* ==================================
       Title
       ================================== */

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 25px;
        margin-bottom: 5px;
    }


    /* ==================================
       Subtitle
       ================================== */

    .subtitle {
        text-align: center;
        color: #aebfd1;
        font-size: 17px;
        margin-bottom: 35px;
    }


    /* ==================================
       Chat Container
       ================================== */

    .chat-box {
        padding: 25px;
        border-radius: 20px;

        background: rgba(255, 255, 255, 0.06);

        border: 1px solid rgba(255, 255, 255, 0.12);

        box-shadow:
            0 8px 30px rgba(0, 0, 0, 0.25);

        backdrop-filter: blur(12px);

        margin-bottom: 20px;
    }


    /* ==================================
       Chat Title
       ================================== */

    .chat-title {
        font-size: 28px;
        font-weight: 600;
        color: white;
        margin-bottom: 20px;
    }


    /* ==================================
       Text Input
       ================================== */

    [data-testid="stTextInput"] label {
        color: #d8e5f2 !important;
        font-weight: 500;
    }


    [data-testid="stTextInput"] input {
        background-color: #101d2c !important;

        color: white !important;

        border: 1px solid #31506d !important;

        border-radius: 12px !important;

        padding: 12px 15px !important;

        font-size: 16px !important;
    }


    [data-testid="stTextInput"] input:focus {
        border: 1px solid #4da6ff !important;

        box-shadow:
            0 0 0 2px rgba(77, 166, 255, 0.15) !important;
    }


    [data-testid="stTextInput"] input::placeholder {
        color: #71859a !important;
    }


    /* ==================================
       Bot Response
       ================================== */

    .bot-message {
        padding: 16px 20px;

        margin-top: 20px;

        border-radius: 14px;

        background: #10263b;

        border-left: 5px solid #2196f3;

        color: #ffffff !important;

        font-size: 17px;

        box-shadow:
            0 5px 20px rgba(0, 0, 0, 0.20);
    }


    .bot-message b {
        color: #70bdff;
    }


    /* ==================================
       Footer
       ================================== */

    .footer {
        text-align: center;

        color: #71859a;

        font-size: 13px;

        margin-top: 25px;
    }

</style>
""", unsafe_allow_html=True)
st.markdown(
    '<div class="main-title">🌡️ Fever Medical Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your friendly assistant for fever-related questions'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="chat-box">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="chat-title">💬 Chat with me</div>',
    unsafe_allow_html=True
)

user_input = st.text_input(
    "You:",
    placeholder="Type your message here..."
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

if user_input:

    if user_input.lower().startswith("what medicine"):

        prediction = st.session_state.get(
            "prediction",
            "No prediction available yet."
        )

        response = f"Based on the model prediction, the medication is {prediction}."

        st.markdown(
            f"""
            <div class="bot-message">
                🤖 <b>Bot:</b> {response}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.snow()

    else:
        response = chatbot.respond(user_input)

        if response:
            st.markdown(
                f"""
                <div class="bot-message">
                    🤖 <b>Bot:</b> {response}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.snow()

        else:
            st.markdown(
                """
                <div class="bot-message">
                    🤖 <b>Bot:</b>
                    Sorry, I don't understand that yet.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.snow()

st.divider()

st.markdown(
    '<div class="footer">⚕️ Fever Medical Chatbot</div>',
    unsafe_allow_html=True
)
