import os
import gdown
import zipfile

import streamlit as st
import torch
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModel, pipeline
from medcat.cat import CAT


# --- DOWNLOAD & ENTPACKEN nur einmal ---
def download_file_if_missing(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info(f"Downloading {output_path} ...")
        gdown.download(url, output_path, quiet=False)
    else:
        st.info(f"{output_path} already exists. Skipping download.")

def download_required_files():
    download_file_if_missing(
        "1CDIi4l8Ge_T3W9-aCUyIaO9Ai_mSsrWN",
        "all_diagnoses_embeddings_prettyname_status_type_contextsim.pt"
    )
    download_file_if_missing(
        "1pKV10Btsbkh8LDqAmeWhO31dZJOz1jDb",
        "mc_modelpack_snomed.zip"
    )
    download_file_if_missing(
        "1Yd6nu3RBio0KMKENUGgAWO_BhKU1sgUI",
        "concepts.csv"
    )
    # Entpacken, falls noch nicht entpackt
    modelpack_folder = "mc_modelpack_snomed"
    if not os.path.exists(modelpack_folder) and os.path.exists("mc_modelpack_snomed.zip"):
        with zipfile.ZipFile("mc_modelpack_snomed.zip", 'r') as zip_ref:
            zip_ref.extractall(modelpack_folder)
        st.info("MedCAT Modelpack entpackt.")

# --- CACHING der Modelle/Daten ---
@st.cache_resource(show_spinner=False)
def load_bert_model():
    return AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

@st.cache_resource(show_spinner=False)
def load_bert_tokenizer():
    return AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

@st.cache_resource(show_spinner=False)
def load_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

@st.cache_resource(show_spinner=False)
def load_medcat():
    return CAT.load_model_pack("mc_modelpack_snomed")

@st.cache_data(show_spinner=False)
def load_embeddings():
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    return torch.load("all_diagnoses_embeddings_prettyname_status_type_contextsim.pt", map_location=device)

@st.cache_data(show_spinner=False)
def load_diagnoses_csv():
    return pd.read_csv("concepts.csv")


# --- APP START ---
download_required_files()

st.markdown("""
# 🩺 Patient Similarity Analysis  
*Compare medical patient records using clinical embeddings*  
---
""")

model = load_bert_model()
model.eval()

tokenizer = load_bert_tokenizer()
summarizer = load_summarizer()
cat = load_medcat()
X_all_diagnoses = load_embeddings()
df_diagnoses = load_diagnoses_csv()

# --- LADE MODELLE UND DATEN ---

lookup_diagnoses = pd.Series(
    df_diagnoses["prettyname_status_type_contextsim"].values,
    index=df_diagnoses["SUBJECT_ID"]
).to_dict()


# 🧠 BONUS: Extrahiere saubere Diagnosen aus Text & entferne Klammerzusätze
def extract_pretty_names(text):
    return "; ".join(sorted(set([
        part.strip().split(" status_")[0].split("(")[0].strip()
        for part in text.split(";") if "status_" in part
    ])))

# --- FUNKTIONEN ---
def annotate_with_medcat(text):
    return cat.get_entities(text)

def medcat_entities_to_text(
    medcat_output,
    include_context_similarity=True,
    context_sim_bucketing=True,
):
    """
    Konvertiert MedCAT-Annotationen in eine Repräsentation mit Status, Typ und Kontextähnlichkeit,
    passend zum ClinicalBERT-Embedding-Prozess.

    Args:
        medcat_output (dict): MedCAT-Ausgabe für einen Text.
        include_context_similarity (bool): Kontextähnlichkeit einbeziehen oder nicht.
        context_sim_bucketing (bool): Kontextähnlichkeit in Buckets (hoch/mittel/niedrig) oder als Zahl.

    Returns:
        str: String-Repräsentation der MedCAT-Konzepte mit Zusatzinfos.
    """
    if not medcat_output or not isinstance(medcat_output, dict):
        return ""

    entities = medcat_output.get('entities', {})
    if not entities:
        return ""

    concepts = []
    for ent in entities.values():
        name = (
            ent.get('pretty_name') or
            ent.get('source_value') or
            ent.get('detected_name') or
            ''
        ).lower().strip()
        if not name:
            continue

        parts = [name]

        # Status + Typ
        status = ent.get('meta_anns', {}).get('Status', {}).get('value', '').lower()
        ent_type = ent.get('type', '').lower()
        if status:
            parts.append(f"status_{status}")
        if ent_type:
            parts.append(f"type_{ent_type}")

        # Kontextähnlichkeit
        if include_context_similarity:
            sim = ent.get("context_similarity", 0.0)
            if context_sim_bucketing:
                if sim >= 0.9:
                    parts.append("high_sim")
                elif sim >= 0.6:
                    parts.append("med_sim")
                else:
                    parts.append("low_sim")
            else:
                parts.append(f"context_sim_{round(sim, 2)}")

        concept = " ".join(parts)
        concepts.append(concept)

    return "; ".join(sorted(set(concepts)))

def get_affirmed_concepts(medcat_output):
    """
    Gibt eine Liste der affirmed Diagnosen (nur Name) zurück.
    Nur Konzepte mit affirmed Status und aus erlaubten Quellen.
    """
    if not medcat_output or not isinstance(medcat_output, dict):
        return []

    entities = medcat_output.get('entities', {})
    if not entities:
        return []

    concepts = []
    allowed_sources = {"SNOMEDCT_US", "ICD10CM", "ICD9CM"}
    allowed_types = {"disorder", "disease", "finding"}

    for ent in entities.values():
        name = (ent.get('pretty_name') or '').lower().strip()
        name = name.split("(")[0].strip()  # entfernt Klammer-Ergänzungen
        status = ent.get('meta_anns', {}).get('Status', {}).get('value', '').lower()
        ent_type = ent.get('type', '')
        vocab = ent.get('vocab', '')

        if status != "affirmed" or not name:
            continue

        if ent_type and ent_type.lower() not in allowed_types:
            continue
        if vocab and vocab.upper() not in allowed_sources:
            continue

        concepts.append(name)

    return "; ".join(sorted(set(concepts)))

def compute_f1(set1, set2):
    tp = len(set1 & set2)
    fp = len(set2 - set1)
    fn = len(set1 - set2)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1, precision, recall, tp, fp, fn

def badge_color(value):
    if value > 0.7:
        return "✅"
    elif value > 0.3:
        return "🟡"
    else:
        return "🔴"

def summarize_text(text, max_chunk_length=1000):
    if len(text) <= max_chunk_length:
        return summarizer(text)[0]["summary_text"]
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    summaries = [summarizer(chunk)[0]["summary_text"] for chunk in chunks]
    return " ".join(summaries)

# --- INTERFACE ---
uploaded_file = st.file_uploader("📄 Lade eine Patientenakte (PDF)", type="pdf")

if uploaded_file:
    with st.expander("❓ Wie funktioniert die Analyse?"):
        st.markdown("""
        Diese Anwendung analysiert hochgeladene Patientenakten mithilfe medizinischer KI.  
        Diagnosen werden automatisch erkannt, in ein standardisiertes Format gebracht und mit Fällen aus der MIMIC-III-Datenbank verglichen.  
        Ziel: Ähnliche Fälle finden, um klinische Entscheidungsprozesse zu unterstützen.
        """)

    st.info("🔒 Hinweis: Diese Anwendung dient ausschliesslich der Unterstützung medizinischer Fachpersonen. Die Entscheidungshoheit liegt stets bei der behandelnden Ärztin oder dem Arzt.")

    reader = PdfReader(uploaded_file)
    full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

    st.subheader("📃 Inhalt der Patientenakte")
    st.text_area("Auszug:", full_text[:3000], height=200)

    # --- Zusammenfassung
    with st.expander("📝 Automatische Zusammenfassung durch BART"):
        with st.spinner("Erstelle Zusammenfassung..."):
            summary = summarize_text(full_text)
            st.success(summary)

    # --- MedCAT
    st.markdown("---")
    st.subheader("🧬 Erkannte Diagnosen (affirmed, ICD/SNOMED)")
    with st.spinner("Analysiere medizinische Konzepte..."):
        annotations = annotate_with_medcat(full_text)

        # Für Embedding
        source_diag_str = medcat_entities_to_text(annotations)
        source_diag = set(d.strip().lower() for d in source_diag_str.split(";") if d.strip()) if source_diag_str else set()


        # Für UI
        affirmed_diag_ui = get_affirmed_concepts(annotations)

    if affirmed_diag_ui:
        affirmed_list = [d.strip() for d in affirmed_diag_ui.split(";") if d.strip()]
        st.markdown("\n".join([f"- {d}" for d in sorted(affirmed_list)]))
    else:
        st.info("Keine relevanten Diagnosen erkannt (ICD/SNOMED).")

    # --- ClinicalBERT
    with st.spinner("Berechne Ähnlichkeit..."):
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        with torch.no_grad():
            output = model(**inputs)

        last_hidden = output.last_hidden_state
        attention_mask = inputs['attention_mask']
        cls = last_hidden[:, 0, :]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        mean = torch.sum(last_hidden * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        last_hidden[input_mask_expanded == 0] = -1e9
        maxp = torch.max(last_hidden, 1).values

        embedding = torch.cat((cls, mean, maxp), dim=1) # Kombination CLS, MEAN, MAXP

        similarities = cos_sim(embedding, X_all_diagnoses)[0]
        top_index = torch.argmax(similarities).item()
        top_score = similarities[top_index].item()
    
    st.markdown("---")
    st.subheader("🔍 Ähnlichster Patient aus der Datenbank")
    st.markdown(f"**Patient ID:** `{top_index}`")
    st.markdown(f"**Ähnlichkeitswert (Cosinus -Similarity):** `{top_score:.4f}`")

    # --- Diagnosen des ähnlichsten Patienten
    top_patient_id = df_diagnoses.iloc[top_index]["SUBJECT_ID"]
    top_diag_str = lookup_diagnoses.get(top_patient_id, "")
    top_diag = set([d.strip() for d in top_diag_str.split(";") if d.strip()])
    with st.expander("📋 Diagnosen des ähnlichsten Patienten anzeigen"):
        if top_diag:
            for d in sorted(top_diag):
                st.markdown(f"- {d}")
        else:
            st.info("Keine Diagnosen gespeichert.")


    # --- F1 Score & Vergleich
    st.markdown("---")
    st.markdown("## 📊 Ähnlichkeitsmetriken")

    f1, precision, recall, tp, fp, fn = compute_f1(source_diag, top_diag)

    col1, col2, col3 = st.columns(3)
    col1.metric("Übereinstimmungswert (F1 Score)", f"{f1:.3f}", badge_color(f1))
    col2.metric("Precision", f"{precision:.3f}", badge_color(precision))
    col3.metric("Recall", f"{recall:.3f}", badge_color(recall))

    st.markdown(f"""
    - **True Positives (TP)**: `{tp}`
    - **False Positives (FP)**: `{fp}`
    - **False Negatives (FN)**: `{fn}`
    """)

    if f1 == 0:
        st.warning("⚠️ Keine Überschneidung zwischen erkannter und gespeicherter Diagnose gefunden.")

    with st.expander("ℹ️ Was bedeutet das?"):
        st.markdown("""
        - **Übereinstimmungswert (F1 Score)**: Harmonie zwischen Precision & Recall  
        - **Precision**: Anteil korrekter Diagnosen unter den vorhergesagten  
        - **Recall**: Anteil erkannter Diagnosen von allen tatsächlichen  
        """)



st.markdown("""
---
<small><sup>1</sup> Die MIMIC-III-Datenbank (v1.4) enthält anonymisierte Patientenakten des Beth Israel Deaconess Medical Center und wurde durch das MIT Laboratory for Computational Physiology kuratiert.</small>
""", unsafe_allow_html=True)
