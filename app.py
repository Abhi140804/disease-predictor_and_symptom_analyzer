import streamlit as st
import requests
import json
import datetime
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import uuid
import scipy

OLLAMA_API_URL = "http://localhost:11434/api/generate"
HISTORY_FILE = "analysis_history.json"

# 🧠 Stream response from Mistral
def stream_mistral_response(prompt):
    data = {
        "model": "mistral",
        "prompt": prompt,
        "stream": True
    }
    response_text = ""
    try:
        with requests.post(OLLAMA_API_URL, json=data, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = json.loads(line.decode("utf-8"))
                    response_text += decoded_line.get("response", "")
                    yield decoded_line.get("response", "")
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Ollama: {e}")
        yield None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding Ollama response: {e}")
        yield None

# 🧪 Run symptom analysis
def analyze_symptoms(symptoms, disease):
    prompt = (
        f"Based on the symptoms: {', '.join(symptoms)}, and the predicted disease {disease}.\n"
        "Return ONLY valid JSON like this:\n"
        '{\n'
        '  "medicines": "List of suggested medicines",\n'
        '  "precautions": "List of precautions",\n'
        '  "advice": "Other important steps or suggestions"\n'
        '}\n'
        "No explanation. Only JSON."
    )

    full_text = ""
    for chunk in stream_mistral_response(prompt):
        if chunk is None:
            yield None
            return
        full_text += chunk
        yield full_text

# 💾 Save/load history
def save_history_to_file():
    with open(HISTORY_FILE, "w") as f:
        json.dump(st.session_state.analysis_history, f, indent=2)

def load_history_from_file():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return

# 💄 Dark box UI helper
def dark_box(content, color="#2b2b2b"):
    return f"<div style='background-color:{color};padding:15px;border-radius:10px;color:white;'>{content}</div>"

def format_list(items):
    if not items:
        return "<ul><li>N/A</li></ul>"
    if isinstance(items, str):
        items = [item.strip() for item in items.split(",") if item.strip()]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"

# 🛠 Setup
st.set_page_config(page_title="AI Symptom Analyzer", layout="wide")
st.markdown("""<style>body { background-color: #1c1c1c; }.stApp { color: white; }</style>""", unsafe_allow_html=True)

# 🚀 Header
st.markdown("<h1 style='text-align:center; color:white;'>🧠 AI Symptom Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Powered by Ollama Mistral & Disease Prediction Model</p>", unsafe_allow_html=True)
st.markdown("---")

# 📅 Initialize session state
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = load_history_from_file()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "viewing_history" not in st.session_state:
    st.session_state.viewing_history = False
if "viewed_entry" not in st.session_state:
    st.session_state.viewed_entry = None

# Sidebar for history view
with st.sidebar:
    st.title("📚 Session History")
    if st.session_state.analysis_history:
        for i, entry in enumerate(reversed(st.session_state.analysis_history)):
            label = f"Session on {entry['timestamp']}"
            if st.button(label, key=f"history_{i}"):
                st.session_state.viewing_history = True
                st.session_state.viewed_entry = entry
    else:
        st.info("No history available.")

# ====================================================
# LOAD DATASETS
# ====================================================
training_data = "dataset/Training.csv"
testing_data = "dataset/Testing.csv"

try:
    data = pd.read_csv(training_data).dropna(axis=1)
    test_data = pd.read_csv(testing_data).dropna(axis=1)
except FileNotFoundError:
    st.error("Dataset files not found. Please check the paths.")
    st.stop()

encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

final_rf_model = RandomForestClassifier(random_state=18)
final_rf_model.fit(X, y)

symptoms_list = X.columns.values
symptom_index = {
    " ".join([i.capitalize() for i in value.split("_")]): index
    for index, value in enumerate(symptoms_list)
}

data_dict = {"symptom_index": symptom_index, "predictions_classes": encoder.classes_}

@st.cache_resource
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
    input_data = np.array(input_data).reshape(1, -1)
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    return rf_prediction

# 📥 Input Symptoms
symptoms = []
if not st.session_state.get("viewing_history"):
    cols = st.columns(2)
    symptom_value_range = tuple(["None"] + [" ".join(s.split("_")).title() for s in symptoms_list])
    for i in range(1, 6):
       
            s = st.selectbox(f"Symptom {i}", symptom_value_range, key=f"symptom_{i}")
            if s and s != "None":
                symptoms.append(s)

    # 🧪 Analyze Button
    if st.button("💬 Analyze Symptoms"):
        if len(symptoms) < 2:
            st.error("Please enter at least 2 symptoms.")
        else:
            st.session_state.chat_history = []
            st.session_state.symptom_context = ", ".join(symptoms)
            st.session_state.analysis_json = None
            st.session_state.viewing_history = False

            with st.spinner("Analyzing symptoms..."):
                try:
                    predicted_disease = predictDisease(",".join(symptoms))
                    st.write(f"Predicted Disease: {predicted_disease}")
                    full_response = ""
                    result_box = st.empty()
                    error_occurred = False
                    for chunk in analyze_symptoms(symptoms, predicted_disease):
                        if chunk is None:
                            error_occurred = True
                            break
                        full_response = chunk
                        result_box.code(full_response, language="json")

                    if not error_occurred:
                        try:
                            parsed = json.loads(full_response)
                            st.session_state.analysis_json = parsed
                            new_entry = {
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "symptoms": symptoms,
                                "predicted_disease": predicted_disease,
                                "analysis": parsed,
                                "chat": []
                            }
                            st.session_state.analysis_history.append(new_entry)
                            st.session_state.viewed_entry = new_entry
                            save_history_to_file()
                        except Exception:
                            st.error("❌ Could not parse AI response.")
                except Exception as e:
                    st.error(f"Error during disease prediction: {e}")

# ✅ Display Analysis if Available (Main Flow)
if st.session_state.get("analysis_json") and not st.session_state.get("viewing_history"):
    parsed = st.session_state.analysis_json

    st.markdown("### 💊 Suggested Medicines")
    st.markdown(dark_box(format_list(parsed.get('medicines', 'N/A')), "#2c2c2c"), unsafe_allow_html=True)

    st.markdown("### ⚠️ Precautions")
    st.markdown(dark_box(format_list(parsed.get('precautions', 'N/A')), "#333"), unsafe_allow_html=True)

    st.markdown("### 🏥 Advice")
    st.markdown(dark_box(format_list(parsed.get('advice', 'N/A')), "#2b2b2b"), unsafe_allow_html=True)

# ✅ Display Full History View
if st.session_state.get("viewing_history") and st.session_state.get("viewed_entry"):
    entry = st.session_state.viewed_entry
    st.markdown("## 📄 Detailed Session Report")
    st.markdown(f"### 📅 Date & Time: {entry['timestamp']}")
    st.markdown(f"### 🩺 Predicted Disease: {entry['predicted_disease']}")

    st.markdown("### 🤒 Reported Symptoms")
    st.markdown(dark_box(format_list(entry.get('symptoms',)), "#222"), unsafe_allow_html=True)

    st.markdown("### 💊 Suggested Medicines")
    st.markdown(dark_box(format_list(entry['analysis'].get('medicines', 'N/A')), "#2c2c2c"), unsafe_allow_html=True)

    st.markdown("### ⚠️ Precautions")
    st.markdown(dark_box(format_list(entry['analysis'].get('precautions', 'N/A')), "#333"), unsafe_allow_html=True)

    st.markdown("### 🏥 Advice")
    st.markdown(dark_box(format_list(entry['analysis'].get('advice', 'N/A')), "#2b2b2b"), unsafe_allow_html=True)

    if entry.get("chat"):
        st.markdown("### 💬 Chat History")
        for speaker, msg in entry["chat"]:
            bg = "#444" if speaker == "You" else "#2f2f2f"
            color = "white" if speaker == "You" else "#00FFAA"
            st.markdown(f"<div style='background-color:{bg};padding:10px;border-radius:10px;color:{color};margin-bottom:5px;'><b>{speaker}:</b> {msg}</div>", unsafe_allow_html=True)

# 💬 Chat Section
if st.session_state.get("symptom_context") and not st.session_state.get("viewing_history"):
    st.markdown("---")
    st.markdown("### 💬 Doctor Chat Assistant")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Clear chat button
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

    # Chat input box
    chat_input = st.chat_input("Ask a question about your symptoms...")

    if chat_input:
        st.session_state.chat_history.append(("You", chat_input))

        chat_prompt = (
            f"The user reported these symptoms: {st.session_state.symptom_context}.\n"
            f"Predicted disease: {st.session_state.get('predicted_disease', 'N/A')}.\n"
            f"Medicines: {', '.join(st.session_state.get('analysis', {}).get('medicines', []))}\n"
            f"The user now asks: '{chat_input}'.\n\n"
            "Only respond with a direct, helpful answer to the user's question based on the given disease and medicines.\n"
            "Do not include extra medical disclaimers or general health advice.\n"
            "Be concise and clear. Give solutions in pointwise format."
        )

        st.session_state.chat_history.append(("AI", ""))
        response_index = len(st.session_state.chat_history) - 1

        ai_container = st.empty()
        response_text = ""

        with st.spinner("AI is typing..."):
            for token in stream_mistral_response(chat_prompt):
                if token is None:
                    break
                response_text += token

        st.session_state.chat_history[response_index] = ("AI", response_text)

        # Update chat in history
        if st.session_state.get("analysis_history"):
            st.session_state.analysis_history[-1]["chat"] = st.session_state.chat_history
            save_history_to_file()

    # Display chat history
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)
