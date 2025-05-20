# 🩺 Patient Similarity Analysis

This Streamlit web app analyzes medical patient records (PDFs) and compares them with cases from the MIMIC-III database using clinical embeddings and medical concepts extracted by MedCAT.

---

## Features

- Upload patient records in PDF format  
- Automatic extraction and annotation of medical diagnoses with MedCAT  
- Calculation of similarity to MIMIC-III patients using Bio_ClinicalBERT embeddings  
- Display of the most similar patient including diagnoses and evaluation metrics (F1, Precision, Recall)  
- Automatic text summarization using a BART model  
- Easy deployment with Streamlit  

---

## Requirements

- Python 3.8 or higher  
- Git (needed to install MedCAT from GitHub)  

---

## Project Work

This web app was developed as part of the project **"Enhancing Diagnosis and Prognosis Through Patient Similarity Analysis"**.  
The project is part of the **Applied Digital Life Sciences** program at the **Zurich University of Applied Sciences (ZHAW)** and was conducted in **2025**.  

The project was carried out by **Ravidu Nakandalage**, **Laurin Seelig** and **Luca Meier**.

The goal is to leverage modern NLP techniques and clinical embeddings to compare medical patient records, improving diagnostic and prognostic support.
