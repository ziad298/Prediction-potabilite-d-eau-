import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Water Potability Predictor", page_icon="💧", layout="wide")

MODEL_PATH = "best_model_water.pkl"
SCALER_PATH = "scaler_water.pkl"
DATASET_PATH = "water_potability.csv"

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

DEFAULTS = {
    "ph": 7.0,
    "Hardness": 196.4,
    "Solids": 22014.1,
    "Chloramines": 7.1,
    "Sulfate": 333.8,
    "Conductivity": 426.2,
    "Organic_carbon": 14.3,
    "Trihalomethanes": 66.4,
    "Turbidity": 3.97,
}

RANGES = {
    "ph": (0.0, 14.0, 0.1),
    "Hardness": (0.0, 400.0, 0.1),
    "Solids": (0.0, 70000.0, 1.0),
    "Chloramines": (0.0, 15.0, 0.1),
    "Sulfate": (0.0, 500.0, 0.1),
    "Conductivity": (0.0, 800.0, 0.1),
    "Organic_carbon": (0.0, 30.0, 0.1),
    "Trihalomethanes": (0.0, 150.0, 0.1),
    "Turbidity": (0.0, 8.0, 0.01),
}


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


def predict_binary(model, scaler, input_df):
    x_scaled = scaler.transform(input_df)
    pred = int(model.predict(x_scaled)[0])

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(x_scaled)[0])
        return pred, "decision_score", score

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(x_scaled)[0, 1])
        return pred, "model_probability", proba

    return pred, "class_only", None


def render_prediction_block(pred, score_type, score_value):
    if pred == 1:
        st.success("Decision finale: EAU POTABLE")
    else:
        st.error("Decision finale: EAU NON POTABLE")

    c1, c2, c3 = st.columns(3)
    c1.metric("Classe predite", "Potable" if pred == 1 else "Non potable")
    c2.metric("Code de classe", str(pred))
    c3.metric("Modele", type(model).__name__)

    if score_type == "decision_score" and score_value is not None:
        position = "cote Potable" if score_value >= 0 else "cote Non potable"
        st.info(
            "Position du point par rapport a la frontiere SVM: "
            f"{position} (score={score_value:.4f})."
        )
        st.caption(
            "Interpretation: score > 0 => classe 1, score < 0 => classe 0, "
            "|score| grand => point plus eloigne de la frontiere."
        )
    elif score_type == "model_probability" and score_value is not None:
        st.info(f"Probabilite estimee de la classe 1: {score_value:.2%}")
    else:
        st.info("Ce modele fournit uniquement la classe predite (0/1).")


st.title("Prediction de la potabilite de l'eau")
st.write(
    "Entrez 9 mesures physico-chimiques pour estimer si l'eau est potable (1) ou non potable (0)."
)

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "Fichiers du modele introuvables. Lancez script.py pour generer best_model_water.pkl et scaler_water.pkl."
    )
    st.stop()

try:
    df_data = load_dataset()
except FileNotFoundError:
    df_data = None
    st.warning("Fichier dataset introuvable (water_potability.csv). Le mode test dataset est desactive.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Mode de classification")
    st.caption("Classification binaire simple: decision finale directe du modele (0/1).")
    st.caption("Pour SVM, la position du point est expliquee via le score de frontiere.")

with st.form("prediction_form"):
    st.subheader("Mesures d'entree")
    c1, c2, c3 = st.columns(3)

    with c1:
        ph_min, ph_max, ph_step = RANGES["ph"]
        ph = st.number_input(
            "ph",
            min_value=ph_min,
            max_value=ph_max,
            value=DEFAULTS["ph"],
            step=ph_step,
        )

        hard_min, hard_max, hard_step = RANGES["Hardness"]
        hardness = st.number_input(
            "Hardness",
            min_value=hard_min,
            max_value=hard_max,
            value=DEFAULTS["Hardness"],
            step=hard_step,
        )

        solids_min, solids_max, solids_step = RANGES["Solids"]
        solids = st.number_input(
            "Solids",
            min_value=solids_min,
            max_value=solids_max,
            value=DEFAULTS["Solids"],
            step=solids_step,
        )

    with c2:
        chlor_min, chlor_max, chlor_step = RANGES["Chloramines"]
        chloramines = st.number_input(
            "Chloramines",
            min_value=chlor_min,
            max_value=chlor_max,
            value=DEFAULTS["Chloramines"],
            step=chlor_step,
        )

        sulf_min, sulf_max, sulf_step = RANGES["Sulfate"]
        sulfate = st.number_input(
            "Sulfate",
            min_value=sulf_min,
            max_value=sulf_max,
            value=DEFAULTS["Sulfate"],
            step=sulf_step,
        )

        cond_min, cond_max, cond_step = RANGES["Conductivity"]
        conductivity = st.number_input(
            "Conductivity",
            min_value=cond_min,
            max_value=cond_max,
            value=DEFAULTS["Conductivity"],
            step=cond_step,
        )

    with c3:
        oc_min, oc_max, oc_step = RANGES["Organic_carbon"]
        organic_carbon = st.number_input(
            "Organic_carbon",
            min_value=oc_min,
            max_value=oc_max,
            value=DEFAULTS["Organic_carbon"],
            step=oc_step,
        )

        tri_min, tri_max, tri_step = RANGES["Trihalomethanes"]
        trihalomethanes = st.number_input(
            "Trihalomethanes",
            min_value=tri_min,
            max_value=tri_max,
            value=DEFAULTS["Trihalomethanes"],
            step=tri_step,
        )

        turb_min, turb_max, turb_step = RANGES["Turbidity"]
        turbidity = st.number_input(
            "Turbidity",
            min_value=turb_min,
            max_value=turb_max,
            value=DEFAULTS["Turbidity"],
            step=turb_step,
        )

    submitted = st.form_submit_button("Predire")

if submitted:
    input_data = pd.DataFrame(
        [[
            ph,
            hardness,
            solids,
            chloramines,
            sulfate,
            conductivity,
            organic_carbon,
            trihalomethanes,
            turbidity,
        ]],
        columns=FEATURES,
    )

    pred, score_type, score_value = predict_binary(model, scaler, input_data)
    render_prediction_block(pred, score_type, score_value)

    st.session_state.history.insert(
        0,
        {
            "source": "saisie_manuelle",
            "prediction": pred,
            "score_type": score_type,
            "score_value": round(float(score_value), 4) if score_value is not None else None,
        },
    )

st.subheader("Test rapide depuis le dataset")
if df_data is not None:
    test_cols = st.columns([1, 1, 2])
    with test_cols[0]:
        target_filter = st.selectbox("Classe reelle", options=["toutes", "0", "1"], index=0)
    with test_cols[1]:
        max_idx = len(df_data) - 1
        row_index = st.number_input(
            "Index de ligne",
            min_value=0,
            max_value=max_idx,
            value=251,
            step=1,
        )
    with test_cols[2]:
        run_dataset_test = st.button("Lancer le test dataset")

    if run_dataset_test:
        selected = df_data.copy()
        if target_filter != "toutes":
            selected = selected[selected["Potability"] == int(target_filter)]
            selected = selected.reset_index(drop=True)

        if selected.empty:
            st.warning("Aucune ligne disponible avec ce filtre.")
        else:
            safe_index = int(min(row_index, len(selected) - 1))
            row = selected.iloc[safe_index].copy()
            real_label = int(row["Potability"])

            for col in FEATURES:
                if pd.isna(row[col]):
                    row[col] = df_data[col].median()

            sample_df = pd.DataFrame([row[FEATURES].tolist()], columns=FEATURES)
            pred, score_type, score_value = predict_binary(model, scaler, sample_df)

            st.write(f"Ligne testee du dataset: {safe_index}")
            st.write(f"Classe reelle (Potability): {real_label}")
            render_prediction_block(pred, score_type, score_value)

            if pred == real_label:
                st.success("Resultat: bonne classification pour cette ligne.")
            else:
                st.warning("Resultat: mauvaise classification pour cette ligne.")

            st.session_state.history.insert(
                0,
                {
                    "source": f"ligne_dataset_{safe_index}",
                    "true_label": real_label,
                    "prediction": pred,
                    "score_type": score_type,
                    "score_value": round(float(score_value), 4) if score_value is not None else None,
                },
            )

st.subheader("Historique des predictions")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
    st.download_button(
        "Telecharger l'historique (CSV)",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="historique_predictions.csv",
        mime="text/csv",
    )
else:
    st.caption("Aucune prediction pour le moment.")