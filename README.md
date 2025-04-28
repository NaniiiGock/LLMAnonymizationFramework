
# LLM ANONYMIZATION FRAMEWORK

# Overview

This is the repository accompanying our bachelor thesis "Comparative Analysis and Framework Development of Anonymization Techniques for Large Language Models" containing the code of the framework and scripts to reproduce all our main experiments, plots, and configurations.

# UV setup + spacy

uv init my_project
cd my_project
uv add spacy
uv run python -m spacy download en_core_web_sm

uv sync

# Ollama setup

install ollama 
>>> ollama run llama3.2

# Run

uv run streamlit run streamlit_app.py




# Privacy-Preserving LLM Anonymization & RAG Framework

## Overview
This project provides a modular framework for anonymizing sensitive data before sending it to external large language models (LLMs) such as GPT, Claude, or Gemini. It supports customizable anonymization pipelines, retrieval-augmented generation (RAG) workflows, and evaluation tools for measuring masking quality and retrieval performance.

Organizations bound by data protection laws like GDPR and HIPAA often cannot use external LLMs directly due to data privacy concerns. This framework enables them to use powerful LLMs while maintaining compliance.

---

## Features
- **Interactive anonymization pipeline:** 
  - Pattern-based masking
  - Transformer-based NER (fine-tuned RoBERTa, DistilBERT)
  - LLM-based masking
  - Context-aware masking for culturally sensitive cases
- **RAG retrieval with ChromaDB:** 
  - Flexible configuration (chunk size, overlap, retriever model)
- **RAG retrieval with MongoDB Atlas:**
  - optional Homomorphic encryption of embeddings before populating the database
- **Evaluation metrics:** 
  - Faithfulness, answer relevancy, context precision, context recall, answer correctness
- **Supports English and Ukrainian languages**
- **Open-source modular design:** 
  - Easy integration into external systems
  - Optional UI for building and testing pipelines
---

## Installation
```bash
git clone https://github.com/NaniiiGock/LLMAnonymizationThesis.git
cd LLMAnonymizationThesis

python -m venv .venv
source .venv/bin/activate

pip install uv

uv sync

```

---

## Directory Structure
```text
/configs        # Pipeline configurations
/data           # Input files for training, RAG, and testing
/src            # Source code
/models         # Trained NER models and local LLM configs
/outputs        # Generated outputs and evaluation results
```

---

## Quickstart
1. **Prepare your input text or documents**
2. **Configure the pipeline** via a YAML or UI
3. **Run anonymization and retrieval**
4. **Evaluate output with provided scripts**

Example:
```python
python run_pipeline.py --config configs/default_config.yaml
```

---

## Configuration Options
- **NER model:** Select from pretrained, fine-tuned, or LLM-based
- **Vector DB settings:** (chunk size, overlap, retriever model)
- **Masking strategies:** Regex, categorical, context-aware
- **LLM providers:** GPT-4o, Claude, Gemini, Local LLMs

---

## Training Custom NER Models
Train your own domain-specific models using IOB formatted datasets:
```bash
python train_ner_model.py --train_path data/train.iob --test_path data/test.iob
```

---

## Example Outputs
- Anonymized text samples
- Retrieval examples
- Metric reports saved as CSV

---

## Limitations
- Domain-specific data is often needed for the best results
- LLM-based masking introduces slight latency
- LLM outputs can vary due to probabilistic generation
- Context-aware masking trades off information vs. privacy

---

## Future Work
- Add graph-based pipeline building (no config files)
- Modify LLM prompts from the UI
- Full automation without manual config
- Improve latency for large document processing
- Extend RAG with summarization techniques
- Support additional document formats (e.g., tables, spreadsheets)

---


## Links

- Thesis Document: [In Preparation]