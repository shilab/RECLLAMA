# ReCLLaMA: A Reasoning-Centered LLM Agent for Medical Diagnosis

ReCLLaMA is a modular medical diagnosis framework that combines large language models, biomedical knowledge graphs, and symbolic reasoning to generate interpretable ICD-9 diagnostic predictions with confidence estimates.

The system is designed for research in trustworthy AI for healthcare, knowledge-enhanced LLMs, neuro-symbolic reasoning, and explainable clinical decision support.

---

## Project Structure

```text
ReCLLaMA/
├── knowledge_extraction/          # Symptom and entity extraction modules
├── knowledge_alignment/          # ICD-9 mapping and embedding alignment
├── reasoner/                     # Multi-hop KG symbolic reasoning engine
├── Recllama_main.py              # Main interactive demo pipeline
├── Recllama_ablation.py          # Ablation experiments
├── TruthValue.py                 # Confidence / truth-value estimation
├── result_evaluation.ipynb       # Evaluation notebook
├── rf_model.pkl                  # Random Forest alignment model
├── node2vec_embeddings.txt       # Knowledge graph node embeddings
├── ICD9_symptom_mapping.csv      # Symptom-to-ICD9 mappings
├── D_ICD_DIAGNOSES.csv           # ICD9 code descriptions
├── patient_descriptions_diagnoses.csv
├── questions.csv
├── demo.mov                      # Demo video
Quick Start

Run the main demo with Streamlit:

streamlit run Recllama_main.py

Or run directly with Python:

python Recllama_main.py

## Demo

![Demo Preview](demo.gif)

[Watch Full Demo](./demo.mov)
