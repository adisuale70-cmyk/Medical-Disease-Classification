
import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Medical Disease Classification System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("final_model.pkl")


# =========================================================
# LOAD DATASETS
# =========================================================

description = pd.read_csv("Description.csv")
precaution = pd.read_csv("Precaution.csv")
dosage = pd.read_csv("Dosage.csv")
diets = pd.read_csv("Diets.csv")
medication = pd.read_csv("Medication.csv")


# =========================================================
# CLEAN DATA
# =========================================================

description["Disease"] = (
    description["Disease"]
    .astype(str)
    .str.strip()
)

precaution["Disease"] = (
    precaution["Disease"]
    .astype(str)
    .str.strip()
)

diets["Disease"] = (
    diets["Disease"]
    .astype(str)
    .str.strip()
)

medication["Disease"] = (
    medication["Disease"]
    .astype(str)
    .str.strip()
)

dosage["Treatment"] = (
    dosage["Treatment"]
    .astype(str)
    .str.strip()
)


# =========================================================
# SYMPTOMS
# =========================================================

symptoms = [
    "abdominal cramp",
    "abdominal discomfort",
    "abdominal distension",
    "abdominal pain",
    "abdominal sounds",
    "aphonia",
    "back pain",
    "bloody diarrhea",
    "burning sensation during urination",
    "chest pain",
    "chills",
    "cough",
    "diarrhea",
    "difficulty swallowing",
    "dry cough",
    "epigastric burning",
    "epigastric pain",
    "fatigability",
    "fever",
    "flank pain",
    "gastrointestinal distress",
    "headache",
    "indigestion",
    "joint pain",
    "loss of appetite",
    "lower abdominal pain",
    "malaise",
    "nasal discharge",
    "nausea",
    "productive cough",
    "rectal pain",
    "skin itching",
    "sore throat",
    "sweating",
    "tenesmus",
    "vomiting",
    "weakness"
]


# =========================================================
# TREATMENT LOOKUP
# =========================================================

def get_treatment_details(disease, age):

    result = medication[
        medication["Disease"] == disease
    ]

    if result.empty:
        return []

    treatment_text = result["Treatment"].iloc[0]

    treatments = [
        item.strip()
        for item in treatment_text.split(",")
    ]

    treatment_details = []

    for treatment in treatments:

        dosage_result = dosage[
            (dosage["Age"] == age) &
            (dosage["Treatment"] == treatment)
        ]

        if dosage_result.empty:
            dose = "Not available in dataset"
        else:
            dose = dosage_result["Dosage"].iloc[0]

        treatment_details.append({
            "Treatment": treatment,
            "Dosage": dose
        })

    return treatment_details


# =========================================================
# DISEASE INFORMATION LOOKUP
# =========================================================

def get_all_disease_information(disease, age):

    information = {}

    result = description[
        description["Disease"] == disease
    ]

    if result.empty:
        information["Description"] = (
            "No description available."
        )
    else:
        information["Description"] = (
            result["Description"].iloc[0]
        )

    result = precaution[
        precaution["Disease"] == disease
    ]

    if result.empty:
        information["Precaution"] = (
            "No precaution information available."
        )
    else:
        information["Precaution"] = (
            result["Precaution"].iloc[0]
        )

    result = diets[
        diets["Disease"] == disease
    ]

    if result.empty:
        information["Diet"] = (
            "No diet information available."
        )
    else:
        information["Diet"] = (
            result["Diets"].iloc[0]
        )

    information["Treatments"] = get_treatment_details(
        disease,
        age
    )

    return information


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🩺 Project Dashboard")

    st.markdown(
        """
        ### Medical Disease Classification

        This application uses a machine-learning
        classification model to classify disease
        categories from patient age and reported symptoms.
        """
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Disease Classes:** 4")
    st.write("**Input Features:** 38")
    st.write("**Training Samples:** 138")
    st.write("**CV Accuracy:** 72.35%")

    st.divider()

    st.subheader("Disease Categories")

    st.write("• Acute Febrile Illness (AFI)")
    st.write("• Acute Gastroenteritis (AGE)")
    st.write("• Intestinal Parasites (I/P)")
    st.write("• Upper Respiratory Tract Infection (URTI)")


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🩺 Medical Disease Classification System")

st.markdown(
    """

   # A machine-learning application that classifies
    disease categories from **patient age and reported symptoms**
    and retrieves supporting information from the provided dataset.
    """
)

st.caption(
    "Machine Learning • Logistic Regression • Streamlit"
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.header("👤 Patient Information")

age = st.number_input(
    "Patient Age",
    min_value=1,
    max_value=100,
    value=22,
    step=1
)


# =========================================================
# SYMPTOM SELECTION
# =========================================================

st.header("🔍 Patient Symptoms")

st.write(
    "Select the symptoms reported by the patient."
)

selected_symptoms = {}

columns = st.columns(3)

for index, symptom in enumerate(symptoms):

    column = columns[index % 3]

    with column:

        selected_symptoms[symptom] = st.checkbox(
            symptom,
            key=f"symptom_{index}"
        )


# =========================================================
# COUNT SELECTED SYMPTOMS
# =========================================================

selected_count = sum(
    selected_symptoms.values()
)


st.info(
    f"Selected symptoms: **{selected_count} / {len(symptoms)}**"
)


# =========================================================
# PREPARE MODEL INPUT
# =========================================================

symptom_values = {
    symptom: int(selected_symptoms[symptom])
    for symptom in symptoms
}

input_data = pd.DataFrame(
    [
        [age] +
        [
            symptom_values[symptom]
            for symptom in symptoms
        ]
    ],
    columns=["Age"] + symptoms
)


st.divider()


# =========================================================
# BUTTONS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    predict_button = st.button(
        "🔍 Predict Disease",
        type="primary",
        use_container_width=True
    )

def reset_symptoms():
    for index in range(len(symptoms)):
        st.session_state[f"symptom_{index}"] = False


with col2:

    st.button(
        "🔄 Reset Symptoms",
        use_container_width=True,
        on_click=reset_symptoms
    )




# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Model prediction
    prediction = model.predict(input_data)

    predicted_disease = prediction[0]


    # Model probabilities
    probabilities = model.predict_proba(
        input_data
    )[0]


    probability_data = pd.DataFrame(
        {
            "Disease": model.classes_,
            "Probability": probabilities
        }
    )


    probability_data = (
        probability_data
        .sort_values(
            "Probability",
            ascending=False
        )
        .reset_index(drop=True)
    )


    predicted_probability = (
        probability_data
        .loc[
            probability_data["Disease"]
            == predicted_disease,
            "Probability"
        ]
        .iloc[0]
    )


    # Disease information
    information = get_all_disease_information(
        predicted_disease,
        age
    )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.header("🎯 Prediction Result")


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Predicted Category",
            predicted_disease
        )


    with result_col2:

        st.metric(
            "Model Probability",
            f"{predicted_probability * 100:.2f}%"
        )


    with result_col3:

        st.metric(
            "Selected Symptoms",
            selected_count
        )


    st.success(
        f"Predicted Disease Category: "
        f"**{predicted_disease}**"
    )


    # =====================================================
    # PROBABILITY DISTRIBUTION
    # =====================================================

    st.subheader("📊 Model Probability Distribution")

    st.write(
        "The chart shows the model's estimated probability "
        "for each disease category."
    )


    chart_data = probability_data.copy()

    chart_data["Probability"] = (
        chart_data["Probability"] * 100
    )

    chart_data = chart_data.set_index(
        "Disease"
    )


    st.bar_chart(
        chart_data["Probability"],
        use_container_width=True
    )


    # =====================================================
    # DISEASE DESCRIPTION
    # =====================================================

    st.subheader("📖 Disease Description")

    st.write(
        information["Description"]
    )


    # =====================================================
    # TREATMENT INFORMATION
    # =====================================================

    st.subheader("💊 Treatment Information")

    if information["Treatments"]:

        treatment_table = pd.DataFrame(
            information["Treatments"]
        )

        st.dataframe(
            treatment_table,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No treatment information available "
            "in the dataset."
        )


    # =====================================================
    # PRECAUTION
    # =====================================================

    st.subheader("🛡️ Precaution")

    st.write(
        information["Precaution"]
    )


    # =====================================================
    # DIET
    # =====================================================

    st.subheader("🥗 Diet Information")

    st.write(
        information["Diet"]
    )


    # =====================================================
    # MODEL DETAILS
    # =====================================================

    st.divider()

    st.subheader("🤖 Machine Learning Model")

    model_col1, model_col2, model_col3, model_col4 = st.columns(4)


    with model_col1:

        st.metric(
            "Algorithm",
            "Logistic Regression"
        )


    with model_col2:

        st.metric(
            "Training Samples",
            "138"
        )


    with model_col3:

        st.metric(
            "Input Features",
            "38"
        )


    with model_col4:

        st.metric(
            "CV Accuracy",
            "72.35%"
        )


    # =====================================================
    # PROJECT ARCHITECTURE
    # =====================================================

    st.subheader("🔄 Project Workflow")

    st.markdown(
        """
        **Dataset**
        ↓

        **Data Cleaning**
        ↓

        **Feature & Target Preparation**
        ↓

        **Model Comparison**
        ↓

        **Cross-Validation**
        ↓

        **Logistic Regression**
        ↓

        **Disease Prediction**
        ↓

        **Disease Information Lookup**
        """
    )


    # =====================================================
    # PROJECT NOTE
    # =====================================================

    st.divider()

    st.caption(
        "This application demonstrates a machine-learning "
        "classification workflow using the provided dataset. "
        "Model probabilities represent the classifier's output "
        "for the selected input features."
    )
