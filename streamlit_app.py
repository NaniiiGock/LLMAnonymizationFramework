import streamlit as st
import yaml
from streamlit_sortables import sort_items
import asyncio
from privacy_pipeline import PrivacyPipeline
import os
import streamlit.components.v1 as components
import re

# =================== HELPER FUNCTIONS ===================

def create_html_with_entities(text, entity_mappings):
    sorted_entities = sorted(entity_mappings.items(), key=lambda x: len(x[0]), reverse=True)

    def replace_entity(match):
        entity_text = match.group(0)
        for old_entity, new_entity in sorted_entities:
            if old_entity in entity_text:
                return f'<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">{old_entity} <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">{new_entity}</span></mark>'
        return entity_text

    html_content = re.sub(r'\b(' + '|'.join(re.escape(key) for key in entity_mappings.keys()) + r')\b', replace_entity, text)
    
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
     <body style="font-size: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; padding: 4rem 2rem; direction: ltr">
      <figure style="margin-bottom: 6rem">
       <div class="entities" style="line-height: 2.5; direction: ltr">
        {html_content}
       </div>
      </figure>
     </body>
    </html>
    """
    return full_html

def load_config():
    try:
        with open(CONFIG_FILE, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error reading config file: {e}")
        return {}
    
def save_config(config):
    with open(CONFIG_FILE, "w") as file:
        yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)
    st.session_state.config = yaml.safe_load(CONFIG_FILE)

def update_processing_order(name, enabled):
    order = config.get("processing", {}).get("order", [])
    if enabled and name not in order:
        order.append(name)
    elif not enabled and name in order:
        order.remove(name)
    config["processing"]["order"] = order

# =================== CONFIG LOAD ===================
CONFIG_FILE = "running_config_eng.yml"

config = load_config()

if 'config' not in st.session_state:
    st.session_state.config = config

st.title("Configuration Editor")
st.sidebar.header("Edit Configuration")

# =================== SIDEBAR ===================

# ===== RETRIEVER =====
with st.sidebar.expander("Retriever", expanded=False):
    retriever_enabled = st.checkbox("Enabled", config.get("retriever", {}).get("enabled", True), key="retriever_enabled")
    update_processing_order("retriever", retriever_enabled)

    vector_store_display = st.selectbox("Select Vector Store:", options=["ChromaDB (Local)", "MongoDB (Cloud)"])
    vector_store_mapping = {"ChromaDB (Local)": "chroma", "MongoDB (Cloud)": "mongo"}
    vector_store = vector_store_mapping[vector_store_display]

    HM_enabled = st.selectbox("Homomorphic encryption: ", options=[True, False])
    chunk_size = st.number_input("Chunk Size", min_value=50, max_value=5000, value=500, step=50)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=50, step=10)
    retrieval_type = st.selectbox("Search Type", options=["similarity", "mmr", "hybrid"], index=0)
    k = st.number_input("Top-K Results", min_value=1, max_value=100, value=5, step=1)

    config["retriever"] = {
        "enabled": retriever_enabled,
        "vector_store": vector_store,
        "he_enabled": HM_enabled,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "retrieval_type": retrieval_type,
        "k": k
    }

# ===== PATTERN PROCESSOR =====
with st.sidebar.expander("Pattern Processor", expanded=False):
    pattern_enabled = st.checkbox("Enabled", config.get("pattern_processor", {}).get("enabled", False), key="pattern_enabled")
    update_processing_order("pattern_processor", pattern_enabled)

    custom_patterns = config.get("pattern_processor", {}).get("custom_patterns", {})

    new_key = st.text_input("New Pattern Key", "")
    new_value = st.text_input("New Pattern Value (Regex)", "")
    if st.button("Add Pattern") and new_key and new_value:
        custom_patterns[new_key] = new_value

    for key, value in list(custom_patterns.items()):
        custom_patterns[key] = st.text_input(f"{key}", value)

    config["pattern_processor"] = {
        "enabled": pattern_enabled,
        "custom_patterns": custom_patterns
    }

# ===== NER PROCESSOR =====
with st.sidebar.expander("NER Processor", expanded=False):
    ner_enabled = st.checkbox("Enabled", config.get("ner_processor", {}).get("enabled", False), "ner_enabled")
    update_processing_order("ner_processor", ner_enabled)

    model = st.selectbox("NER Model", ["en_core_web_sm", "uk_core_news_sm", "roberta_ukr", "bert_eng"],
                         index=["en_core_web_sm", "uk_core_news_sm", "roberta_ukr", "bert_eng"].index(config.get("ner_processor", {}).get("model", "en_core_web_sm")))
    entity_types = config.get("ner_processor", {}).get("entity_types", [])
    new_entity = st.text_input("New Entity Type", "")
    if st.button("Add Entity") and new_entity:
        entity_types.append(new_entity)
    entity_types = st.multiselect("Entity Types", ["PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY"] + entity_types, entity_types)

    config["ner_processor"] = {
        "enabled": ner_enabled,
        "model": model,
        "entity_types": entity_types
    }

# ===== LLM NER Processor =====
with st.sidebar.expander("LLM NER Processor", expanded=False):
    llm_ner_enabled = st.checkbox("Enabled", config.get("llm_ner_processor", {}).get("enabled", False), "llm_ner_enabled")
    update_processing_order("llm_ner_processor", llm_ner_enabled)

    adversarial_mode = st.selectbox("Adversarial Mode", [True, False])
    anonymizer_model = st.selectbox("Anonymizer Model", ["mistral", "llama"])
    reidentifier_model = st.selectbox("Reidentifier Model", ["mistral", "llama"])

    config["llm_ner_processor"] = {
        "enabled": llm_ner_enabled,
        "adversarial_mode": adversarial_mode,
        "anonymizer_model": anonymizer_model,
        "reidentifier_model": reidentifier_model
    }

# ===== CONTEXT PROCESSOR =====
with st.sidebar.expander("Context Processor", expanded=False):
    context_enabled = st.checkbox("Enabled", config.get("context_processor", {}).get("enabled", False), "context_enabled")
    update_processing_order("context_processor", context_enabled)

    context_model = st.selectbox("Context Model", ["mistral", "llama"])
    context_level = st.selectbox("Context Level", [0, 1, 2])

    config["context_processor"] = {
        "enabled": context_enabled,
        "model": context_model,
        "context_level": context_level
    }

# ===== LLM INVOKE =====
with st.sidebar.expander("LLM Invoke", expanded=False):
    llm_invoke_enabled = st.checkbox("Enabled", config.get("llm_invoke", {}).get("enabled", True), "llm_invoke_enabled")
    update_processing_order("llm_invoke", llm_invoke_enabled)

    llm_provider = st.selectbox("Provider", ["openai", "anthropic", "local"],
                                index=["openai", "anthropic", "local"].index(config.get("llm_invoke", {}).get("provider", "openai")))
    llm_model = st.text_input("LLM Model", config.get("llm_invoke", {}).get("model", "gpt-4"))
    temperature = st.slider("Temperature", 0.0, 1.0, config.get("llm_invoke", {}).get("temperature", 0.3))
    max_tokens = st.number_input("Max Tokens", 100, 2000, config.get("llm_invoke", {}).get("max_tokens", 1000))

    config["llm_invoke"] = {
        "enabled": llm_invoke_enabled,
        "provider": llm_provider,
        "model": llm_model,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

# ===== LOGGING =====
with st.sidebar.expander("Logging", expanded=False):
    logging_enabled = st.checkbox("Enable Logging", config.get("logging", {}).get("enabled", True), "logging_enabled")
    logging_level = st.selectbox("Logging Level", ["DEBUG", "INFO", "WARNING", "ERROR"],
                                 index=["DEBUG", "INFO", "WARNING", "ERROR"].index(config.get("logging", {}).get("level", "INFO")))
    config["logging"] = {"enabled": logging_enabled, "level": logging_level}

if st.sidebar.button("Save Config"):
    save_config(config)
    st.sidebar.success("Configuration saved successfully!")

# =================== MAIN PAGE ===================

order = config.get("processing", {}).get("order", [])
order = sort_items(order)
config["processing"]["order"] = order

st.title("Chat with Backend Processing")

uploaded_files = st.file_uploader("Upload Files - only while RETRIEVER step is used", type=None, accept_multiple_files=True, key="file_uploader")
upload_folder = "uploaded_files"
os.makedirs(upload_folder, exist_ok=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Saved file: {uploaded_file.name}")

if st.button("Clear ChromaDB and Uploaded Files"):
    import shutil
    db_path = "./chroma_db"
    upload_folder = "uploaded_files"
    try:
        shutil.rmtree(db_path, ignore_errors=True)
        shutil.rmtree(upload_folder, ignore_errors=True)
        os.makedirs(upload_folder, exist_ok=True)

        if "file_uploader" in st.session_state:
            del st.session_state["file_uploader"]

        st.success("ChromaDB and uploaded files cleared successfully.")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error during cleanup: {str(e)}")

user_input = st.text_area("User Input", height=150)
task_description = st.text_area("Task Description", height=100)

if st.button("Submit"):
    if user_input.strip() == "" or task_description.strip() == "":
        st.error("Both User Input and Task Description are required.")
    else:
        pipeline = PrivacyPipeline(CONFIG_FILE)

        async def process_input():
            results = await pipeline.process_pipeline(user_input, task_description, upload_folder)
            return results

        results = asyncio.run(process_input())

        st.subheader("Anonymized Input")
        html_vis = create_html_with_entities(results.get('anonymized_input', 'No anonymized input available.'), 
                                             results.get('context_mapping', 'No mapping available.'))
        components.html(html_vis, height=400, scrolling=True)

        st.subheader("LLM Response")
        html_vis = create_html_with_entities(results.get('llm_response', 'No response available.'), 
                                            results.get('context_mapping', 'No mapping available.'))
        components.html(html_vis, height=400, scrolling=True)

        st.subheader("Final Output")
        html_vis = create_html_with_entities(results.get('final_output', 'No response available.'), 
                                            results.get('mapping', 'No mapping available.'))
        components.html(html_vis, height=400, scrolling=True)

st.subheader("Updated Configuration")
st.code(yaml.safe_dump(config, default_flow_style=False, sort_keys=False))
save_config(config)
