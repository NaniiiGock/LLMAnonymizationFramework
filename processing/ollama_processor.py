from processing.llama_prompts import *

import requests
import re
import json

class LlamaProvider:
    def __init__(self, config=None, model="llama3.2"):
        self.config = config
        self.url = "http://localhost:11434/api/generate"
        self.model = model

    def call_llama_ollama(self, prompt, model = "llama3.2"):
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature" : 0.1
        }
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.status_code} - {response.text}")
        return response.json()["response"]
    
    def invoke(self, prompt):
        return self.call_llama_ollama(prompt, self.model)
    
    def run(self, text):
        content = prompt + text
        try:
            response = self.call_llama_ollama(content)
            text_match = re.search(r'Processed text:\s*"(.+?)"\s*JSON:', response, re.DOTALL)
            processed_text = text_match.group(1).strip() if text_match else ""
            json_match = re.search(r'JSON:\s*(\{.*?\}|\[.*?\])', response, re.DOTALL)
            json_str = json_match.group(1).strip() if json_match else ""
            mappings = json.loads(json_str)
        except:
            return {
                "masked_text": text,
                "mapping": {},
                "raw_response": text
            }
        return {
            "masked_text": processed_text,
            "mapping": mappings,
            "raw_response": response
        }
    
    def run_with_feedback(self, text, feedback, mapping_text):

        text = f"Here are the previous mappings '{mapping_text}'.\
    Here are the recommendations from reidentifier: {feedback}.\
    Now process the following text: {text}"
        content = prompt_with_feedback + text 
        try:
            response = self.call_llama_ollama(content)
            text_match = re.search(r'Processed text:\s*"(.+?)"\s*JSON:', response, re.DOTALL)
            processed_text = text_match.group(1).strip() if text_match else ""
            json_match = re.search(r'JSON:\s*(\{.*?\}|\[.*?\])', response, re.DOTALL)
            json_str = json_match.group(1).strip() if json_match else ""
            mappings = json.loads(json_str)
        except:
            return "Error"
        return {
            "masked_text": processed_text,
            "mapping": mappings,
            "raw_response": response
        }
    
    def run_analysis(self, mappings, masked_text):
        counter_present = 0
        counter_not_present = 0
        for key, _ in mappings.items():
            if key in masked_text:
                counter_present +=1
            else:
                counter_not_present +=1
        print("COUNTER PRESENT: ", counter_present, "\nCOUNTER_NOT_PRESENT: ", counter_not_present)
        accuracy = counter_present * 100 / len(mappings.items())
        return accuracy
    
    def apply_masking_level(self, text, mapping, level, context_text):
        if level == 0:
            return text

        elif level == 1:
            prompt = (
                f"{prompt_level_1_template}"
                f"Current text:\n{context_text}\n\n"
                f"Current mapping:\n{mapping}\n\n"
            )
            replaced = self.call_llama_ollama(prompt)
            return replaced.strip()

        elif level == 2:
            prompt = (
                "You are an assistant that replaces placeholders in text with realistic, context-appropriate values.\n"
                "Ensure the replacements match tone, location, gender, and writing style.\n\n"
                f"Original (with placeholders):\n{text}\n\n"
                f"Context:\n{context_text}\n\n"
                f"Current mapping:\n{mapping}\n\n"
                "Respond with the fully replaced version of the original text (do not include the placeholder version)."
            )

            replaced = self.call_llama_ollama(prompt)
            return replaced.strip()

        else:
            raise ValueError("Invalid masking level. Use 0, 1, or 2.")
        
    def apply_masking_level_iteratively(self, mappings, level, context_text, margin=10):

        def find_occurrences_with_margin(text, substring, margin=10):
            results = []
            for match in re.finditer(re.escape(substring), text):
                start, end = match.start(), match.end()
                context_start = max(0, start - margin)
                context_end = min(len(text), end + margin)
                context = text[context_start:context_end]
                results.append({
                    'match': match.group(),
                    'start': start,
                    'end': end,
                    'context': context
                })
            return [r["context"] for r in results]

        new_mapping = {}

        for entity, identifier in mappings.items():
            category = identifier.split('_')[0]

            if level == 1:
                prompt_level_1 = prompt_level_1_template_iterative.format(entity=entity, identifier=identifier, category=category)
                new_replacement = self.call_llama_ollama(prompt_level_1)
                new_mapping[entity] = new_replacement

            elif level == 2:
                context = find_occurrences_with_margin(context_text, entity, margin=margin)
                prompt_level_2 = prompt_level_2_template_iterative.format(entity=entity, identifier=identifier, category=category, context=context)
                new_replacement = self.call_llama_ollama(prompt_level_2)
                new_mapping[entity] = new_replacement

        return new_mapping
