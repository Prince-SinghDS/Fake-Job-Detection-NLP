import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
shap.initjs()
import time


st.set_page_config(page_title="🕵️ Fake Job Detector", page_icon="🔍", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-color: #0f2027;
        background-image: linear-gradient(to right,#0f2027, #2c5364, #414345, #232526);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

model = joblib.load('model/xgb_final_model.joblib')
explainer = shap.TreeExplainer(model)
vectorizer = joblib.load('model/tfidf_vectorizer.joblib')

st.sidebar.title("About the Project")
st.sidebar.info("""
Detect whether a job posting is real or fake using a trained NLP model.  
Built by **Prince Singh** using TF-IDF + XGBoost.
""")
st.sidebar.markdown("---")
st.sidebar.write("📊 Model: XGBoost")
st.sidebar.write("🔤 Text Vectorizer: TF-IDF")
st.sidebar.write("🔎 Explainability: LIME & SHAP (explained in GitHub repo)")

# Main title
st.title("🕵️ Fake Job Posting Detector")
st.markdown("Check whether a job listing is **Real** or **Fake** using AI 🔍")

# Divider
st.markdown("---")
st.subheader("✍️ Enter Job Details Below")

    
    
if "example" not in st.session_state:
    st.session_state.example = False

def set_example_inputs():
    st.session_state.title = "Data Scientist"
    st.session_state.description = "We are looking for a Data Scientist to analyze large amounts of raw information to find patterns that will help improve our company."
    st.session_state.requirements = "Experience in Python, SQL, machine learning, data visualization. Familiarity with cloud platforms is a plus."

st.button("🎯 Try with Example Job", on_click=set_example_inputs)

    
title = st.text_input("Job Title", key="title", placeholder="e.g., Data Scientist")
description = st.text_area("Job Description", key="description", height=150, placeholder="Describe the job in detail...")
requirements = st.text_area("Job Requirements", key="requirements", height=150, placeholder="List the job requirements...")

col1, col2 = st.columns([5,6])
col1, col_spacer, col2 =st.columns([1,2,1])

with col2:
    def clear_inputs():
        st.session_state.title = ""
        st.session_state.description = ""
        st.session_state.requirements = ""
    st.button("🧹Clear Fields", on_click=clear_inputs)


# Predict Button
with col1:
    if st.button("🫆 Predict"):
        with st.spinner("Analyzing..."):
            time.sleep(1.5)
        if not title or not description or not requirements:
            st.warning("⚠️ Please fill in all the fields before predicting.")
        else:
            text = title + " " + description + " " + requirements
            text_vector = vectorizer.transform([text])
            prediction = model.predict(text_vector)[0]
            probability = model.predict_proba(text_vector)[0][int(prediction)]
            
        if prediction == 1:
            st.success(f"✅ This job posting looks **Real** (Confidence: {probability:.2f})")
        else:
            st.error(f"❌ This job posting looks **Fake** (Confidence: {probability:.2f})")
            
        result_text = f"Prediction: {'Real' if prediction == 1 else 'Fake'}\nConfidence: {probability:.2f}"
        st.download_button("📄 Download Prediction", result_text, file_name="prediction.txt")

       
# ✅ SHAP Waterfall Plot
        shap_values = explainer(text_vector)
        dense_vec = text_vector.toarray()[0]
        feature_names = vectorizer.get_feature_names_out()

        input_features = {
            name: dense_vec[i]
            for i, name in enumerate(feature_names)
            if dense_vec[i] > 0
        }

        shap_exp = shap.Explanation(
            values=shap_values.values[0][dense_vec > 0],
            base_values=shap_values.base_values[0],
            data=dense_vec[dense_vec > 0],
            feature_names=list(input_features.keys())
        )

        st.subheader("🔍 Why did the model make this prediction?")
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_exp, max_display=10, show=False)
        st.pyplot(fig)
            
# Extra details after prediction
        st.markdown(f"🔢 **Words in input**: {len(text.split())}")
        st.progress(int(probability* 100))
        st.caption(f"Model confidence: {probability:.2%}")

        if probability > 0.75:
            st.info("🧠 The model is quite confident in this prediction.")
        elif probability < 0.55:
            st.warning("🤔 The model isn't very confident. Review manually.")

            st.markdown("---")
with st.expander("📊 SHAP Global Explanation (Beeswarm Plot)"):
    st.write("The plot below shows how each feature impacts the model output globally.")
    st.image("visuals/shap_beeswarm_plot.png", caption="SHAP Summary Plot (Beeswarm)", use_column_width=True)
            
# How it works
with st.expander("ℹ️ How this works"):
    st.write("""
    This tool uses a trained XGBoost model that analyzes job title, description, and requirements using TF-IDF features.
    It was trained on a real-world job postings dataset and predicts whether a job post is fake or real.
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Prince Singh| [Linkedin](www.linkedin.com/in/prince-singh-b35209368) | [GitHub Repo](https://github.com/Prince-SinghDS/Fake-Job-Detection-NLP)")