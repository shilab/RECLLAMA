# ReCLLaMA: A Reasoning-Centered LLM Agent for Medical Diagnosis

## Overview

ReCLLaMA is a lightweight medical diagnosis framework that combines:

1. **Knowledge Extraction** – extract clinical symptoms and entities from patient descriptions  
2. **Knowledge Alignment** – map symptoms and procedures to ICD-9 diagnoses and biomedical evidence  
3. **Reasoning Engine** – perform multi-hop reasoning over a biomedical knowledge graph  
4. **LLM Explanation** – generate interpretable diagnosis reports and rationales

The system is designed for research on trustworthy AI, medical reasoning, and knowledge-enhanced large language models.

---

## Project Structure

```text
ReCLLaMA/
├── knowledge_extraction/        # Symptom/entity extraction modules
├── knowledge_alignment/        # ICD-9 mapping and embedding alignment
├── reasoner/                  # Multi-hop KG reasoning engine
├── Recllama_main.py           # Main demo pipeline
├── Recllama_ablation.py       # Ablation experiments
├── TruthValue.py             # Truth scoring / confidence estimation
├── result_evaluation.ipynb   # Evaluation notebook
├── rf_model.pkl              # Random Forest alignment model
├── node2vec_embeddings.txt   # KG node embeddings
├── ICD9_symptom_mapping.csv  # Symptom to ICD9 mapping
├── D_ICD_DIAGNOSES.csv       # ICD9 descriptions
├── patient_descriptions_diagnoses.csv
├── questions.csv


## Requirements

Recommended Python version:

- Python 3.9+

Install dependencies:

```bash
pip install pandas numpy scikit-learn torch transformers streamlit


Quick Start

Run the main demo with Streamlit:

streamlit run Recllama_main.py

Or run directly with Python:

python Recllama_main.py
