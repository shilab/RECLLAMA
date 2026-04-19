# ReCLLaMA: A Reasoning-Centered LLM Agent for Medical Diagnosis

<p align="center">
A modular neuro-symbolic framework that combines Large Language Models, Biomedical Knowledge Graphs, and Symbolic Reasoning for interpretable medical diagnosis.
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.9%2B-blue">
<img src="https://img.shields.io/badge/Framework-Streamlit-red">
<img src="https://img.shields.io/badge/Domain-Healthcare-green">
<img src="https://img.shields.io/badge/License-Research-lightgrey">
</p>

---

## Overview

ReCLLaMA is a lightweight and interpretable clinical reasoning system that integrates:

- **Knowledge Extraction** from free-text patient narratives using LLMs  
- **Knowledge Alignment** from symptoms/procedures to ICD-9 and biomedical entities  
- **Knowledge Reasoning** via multi-hop symbolic inference over biomedical knowledge graphs  

Unlike black-box diagnosis models, ReCLLaMA provides:

- Transparent reasoning paths  
- Confidence-aware predictions  
- Structured ICD-9 outputs  
- Modular and efficient deployment  

---

## Demo

<p align="center">
<b>Interactive Medical Diagnosis Demo</b>
</p>

[Watch Full Demo Video](./demo.mov)

> If GitHub does not preview `.mov`, download locally and open it.

---

## Project Structure

```text
ReCLLaMA/
├── knowledge_extraction/          # Symptom / entity extraction modules
├── knowledge_alignment/          # ICD-9 mapping + embedding alignment
├── reasoner/                     # Multi-hop KG symbolic reasoning engine
├── Recllama_main.py              # Main Streamlit demo
├── Recllama_ablation.py          # Ablation experiments
├── TruthValue.py                 # Confidence estimation
├── result_evaluation.ipynb       # Evaluation notebook
├── rf_model.pkl                  # Random Forest alignment model
├── node2vec_embeddings.txt       # KG node embeddings
├── ICD9_symptom_mapping.csv      # Symptom-to-ICD9 mapping
├── D_ICD_DIAGNOSES.csv           # ICD9 descriptions
├── patient_descriptions_diagnoses.csv
├── questions.csv
├── demo.mov                      # Demo video


## Installation

Recommended Python version:

- Python 3.9+

Install dependencies:

```bash
pip install pandas numpy scikit-learn torch transformers streamlit
