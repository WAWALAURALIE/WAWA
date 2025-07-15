# app_final.py
# Purpose: Streamlit web application for predicting Regret_choix – final, stable version.
# Removed: probability distribution section.

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sys

st.set_page_config(page_title="Regret Choice Prediction", layout="wide")

# --- Version check ---
try:
    import streamlit as st_version
    if not st_version.__version__.startswith("1."):
        st.error(
            f"Streamlit version {st_version.__version__} detected. "
            "This app requires Streamlit 1.x (1.38.0+ recommended). "
            "Update with `pip install --upgrade streamlit`."
        )
        st.stop()
except Exception:
    st.error("Cannot determine Streamlit version.")
    st.stop()

# --- Session state ---
for k in ("prediction_made", "last_prediction"):
    if k not in st.session_state:
        st.session_state[k] = False if k == "prediction_made" else None

# --- Load artefacts ---
@st.cache_resource
def load_artifacts():
    try:
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        model_info = {
            "model_name": getattr(model, "__class__", type(model)).__name__,
            "misclassification_rate": "N/A",
        }
        try:
            with open("model_info.pkl", "rb") as f:
                model_info.update(pickle.load(f))
        except FileNotFoundError:
            pass
        return model, scaler, feature_names, model_info
    except FileNotFoundError as e:
        st.error(f"Missing .pkl file: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading .pkl files: {e}")
        st.stop()

model, scaler, feature_names, model_info = load_artifacts()

# --- UI Header ---
st.title("🎓 Prediction of Study Field Regret")
st.markdown(
    "Fill in all fields accurately and submit the form to see the result."
)

# --- Prediction logic ---
def make_prediction(input_data):
    try:
        input_df = pd.DataFrame([input_data])
        categorical_columns = [
            "Sexe", "Ville_bac", "Etablissement_post_bac", "Ville_post_bac",
            "Profession_parent", "Filiere_post_bac", "Raison_choix_filiere",
            "Option_filiere", "Feeling", "Field_knowledge", "Conseillé_orientation",
            "Influence_choix", "Orientation_platform_acces", "Orientation_forum",
            "Satisfaction_filiere", "Niveau_etude_post_bac", "Diplôme_obtenue",
            "Employabilité", "Reconversion", "Employabilite_nouvelle_filiere",
            "Nouvelle_filiere", "Regret_orientation",
        ]
        input_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
        input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

        numerical_features = ["Age", "Annee_bac"]
        if not all(f in input_encoded.columns for f in numerical_features):
            return None, "Numerical features missing after encoding."

        input_encoded[numerical_features] = scaler.transform(input_encoded[numerical_features])
        prediction = model.predict(input_encoded)[0]
        result = "Regret" if prediction == 1 else "No Regret"
        return {"result": result}, None
    except Exception as e:
        return None, str(e)

# --- Input form ---
with st.form("prediction_form"):
    st.subheader("Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 60, 25)
        sexe = st.selectbox("Sex (Sexe)", ["Masculin", "Feminin"])
        ville_bac = st.text_input("City of Baccalaureate (Ville_bac)", "Abidjan")
    with col2:
        annee_bac = st.number_input("Year of Baccalaureate (Annee_bac)", 2000, 2025, 2020)
        etablissement_post_bac = st.selectbox(
            "Post-Bac Institution (Etablissement_post_bac)",
            ["Universite privee", "Universite publique", "Grandes ecoles"],
        )
        ville_post_bac = st.text_input("City of Post-Bac Studies (Ville_post_bac)", "Abidjan")

    st.subheader("Educational and Professional Details")
    col3, col4 = st.columns(2)
    with col3:
        profession_parent = st.text_input("Parents' Profession (Profession_parent)", "commercant")
        filiere_post_bac = st.text_input("Field of Study (Filiere_post_bac)", "Science economique et gestion")
        raison_choix_filiere = st.text_input("Reason for Choosing Field (Raison_choix_filiere)", "Passion")
        option_filiere = st.selectbox("Field Option (Option_filiere)", ["Scientifique", "Litteraire", "Technologie", "Pratique"])
        feeling = st.selectbox("Feeling about Field (Feeling)", ["Oui", "Non", "Un peu"])
        field_knowledge = st.selectbox("Knowledge of Field (Field_knowledge)", ["Oui", "Non", "Un peu"])
    with col4:
        conseille_orientation = st.selectbox("Received Orientation Advice (Conseillé_orientation)", ["Oui", "Non"])
        influence_choix = st.text_input("Who Influenced Your Choice (Influence_choix)", "Parents")
        orientation_platform_acces = st.selectbox("Accessed Orientation Platform (Orientation_platform_acces)", ["Oui", "Non"])
        orientation_forum = st.selectbox(
            "Orientation Forum Participation (Orientation_forum)",
            ["Oui, j'y ai participe", "Oui mais je n'ai pas participe", "Non"],
        )
        satisfaction_filiere = st.selectbox("Satisfaction with Field (Satisfaction_filiere)", ["Oui", "Non", "Un peu"])
        niveau_etude_post_bac = st.selectbox(
            "Level of Post-Bac Studies (Niveau_etude_post_bac)",
            ["Bac", "Bac+1", "Bac+2", "Bac+3", "Bac+4", "Bac+5", "Bac+8"],
        )

    st.subheader("Employment and Reconversion")
    col5, col6 = st.columns(2)
    with col5:
        diplome_obtenue = st.selectbox("Degree Obtained (Diplôme_obtenue)", ["Oui", "Non"])
        employabilite = st.selectbox("Employability (Employabilité)", ["Oui", "Non"])
        reconversion = st.selectbox("Changed Field (Reconversion)", ["Oui", "Non"])
    with col6:
        employabilite_nouvelle_filiere = st.selectbox(
            "Employability in New Field (Employabilite_nouvelle_filiere)", ["Oui", "Non"]
        )
        nouvelle_filiere = st.text_input("New Field if Changed (Nouvelle_filiere)", "Science economique et gestion")
        regret_orientation = st.text_input("Reason for Regret (Regret_orientation)", "Rien")

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

if submitted:
    text_inputs = [
        ville_bac, ville_post_bac, profession_parent, filiere_post_bac,
        raison_choix_filiere, influence_choix, nouvelle_filiere, regret_orientation,
    ]
    if any(not s.strip() for s in text_inputs):
        st.error("All text fields must be filled with non-empty values.")
    else:
        input_data = {
            "Age": age,
            "Annee_bac": annee_bac,
            "Sexe": sexe,
            "Ville_bac": ville_bac,
            "Etablissement_post_bac": etablissement_post_bac,
            "Ville_post_bac": ville_post_bac,
            "Profession_parent": profession_parent,
            "Filiere_post_bac": filiere_post_bac,
            "Raison_choix_filiere": raison_choix_filiere,
            "Option_filiere": option_filiere,
            "Feeling": feeling,
            "Field_knowledge": field_knowledge,
            "Conseillé_orientation": conseille_orientation,
            "Influence_choix": influence_choix,
            "Orientation_platform_acces": orientation_platform_acces,
            "Orientation_forum": orientation_forum,
            "Satisfaction_filiere": satisfaction_filiere,
            "Niveau_etude_post_bac": niveau_etude_post_bac,
            "Diplôme_obtenue": diplome_obtenue,
            "Employabilité": employabilite,
            "Reconversion": reconversion,
            "Employabilite_nouvelle_filiere": employabilite_nouvelle_filiere,
            "Nouvelle_filiere": nouvelle_filiere,
            "Regret_orientation": regret_orientation,
        }
        res, err = make_prediction(input_data)
        if err:
            st.error(f"Prediction failed: {err}")
        else:
            st.session_state.prediction_made = True
            st.session_state.last_prediction = res

# --- Display result ---
if st.session_state.prediction_made and st.session_state.last_prediction:
    pred = st.session_state.last_prediction
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    if pred["result"] == "Regret":
        st.error(f"**Prediction: {pred['result']}**")
        st.write("The model predicts that the student **will likely regret** their choice of study field.")
    else:
        st.success(f"**Prediction: {pred['result']}**")
        st.write("The model predicts that the student **will likely not regret** their choice of study field.")

    if st.button("🔄 Clear Results"):
        st.session_state.prediction_made = False
        st.session_state.last_prediction = None