# app.py
import streamlit as st
import joblib
import pandas as pd
import altair as alt
from type_encoder import predict_type

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    mlb = joblib.load("mlb.joblib")
    return model, vectorizer, mlb

model, vectorizer, mlb = load_artifacts()

st.title("🔍 Pokémon Type Predictor")

st.markdown(
    """
    Enter a name and get its predicted types based on
    a character-level deep learning model trained on real data!
    """
)

pokemon_name = st.text_input("🧬 Enter a name:")

if st.button("Predict Type") and pokemon_name:
    with st.spinner("Predicting..."):
        types, type_scores = predict_type(pokemon_name, model, vectorizer, mlb)
    
    st.success(f"**Predicted Type(s):** {', '.join(types)}")

    if type_scores:
        # Prepare dataframe, sort descending, limit top 5
        df_scores = pd.DataFrame({
            "Type": list(type_scores.keys()),
            "Confidence": list(type_scores.values())
        }).sort_values("Confidence", ascending=False).head(5)

        st.subheader("📊 Top 5 Type Confidence Scores")

        # Altair bar chart with sorted Y axis and tooltips
        chart = alt.Chart(df_scores).mark_bar().encode(
            x=alt.X('Confidence', title='Confidence', scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('Type', sort='-x', title='Type'),
            tooltip=['Type', alt.Tooltip('Confidence', format='.2f')]
        ).properties(
            width=600,
            height=300
        )

        st.altair_chart(chart, use_container_width=True)
