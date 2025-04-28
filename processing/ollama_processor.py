prompt = """You are an anonymization assistant.

Your task is to identify and replace all entities in the following text that belong to any of the following categories, using a consistent identifier format. You must detect and anonymize:

- Persons (PER)
- Locations (LOC) — cities, countries, landmarks, addresses
- Organizations (ORG) — companies, institutions, government bodies
- Nationalities or religious/political groups (NORP)
- Dates (DATE)
- Events (EVENT)
- Products (PRODUCT)
- Laws (LAW)
- Languages (LANGUAGE)
- Works of art (WORK_OF_ART)
- Time values (TIME)
- Percentages (PERCENT)
- Money values (MONEY)
- Quantities (QUANTITY)

Each entity must be replaced with a unique identifier in the format:
- PER_1, PER_2, ...
- LOC_1, LOC_2, ...
- ORG_1, ORG_2, ...
- DATE_1, EVENT_1, etc.

Instructions:
- Replace every full and partial mention of the entity with the correct identifier.
  E.g., "Barack Obama" and "Obama" → PER_1
- Reuse the same identifier consistently across the entire text.
- Do **not** include code or explanations.
- Return **only**:
  1. The processed text with all entities replaced.
  2. A JSON list of all replaced entities with this structure:
     {"<original mention>", "<PII-style ID>"}

Example:

Original: 
"Barack Obama was born in Honolulu in 1961. Obama was elected president in 2008."

Response:
Processed text: 
"[PER_1] was born in [LOC_1] in [DATE_1]. [PER_1] was elected president in [DATE_2]."

JSON:
{
  "Barack Obama": "PER_1",
  "Obama": "PER_1",
  "Honolulu": "LOC_1",
  "1961": "DATE_1",
  "2008": "DATE_2"
}


Now process the following text:

"""


prompt_with_feedback = """
You are an anonymization assistant.

Your task is to identify and replace all entities in the following text that belong to any of the following categories, using a consistent identifier format. You must detect and anonymize:

- Persons (PER)
- Locations (LOC) — cities, countries, landmarks, addresses
- Organizations (ORG) — companies, institutions, government bodies
- Nationalities or religious/political groups (NORP)
- Dates (DATE)
- Events (EVENT)
- Products (PRODUCT)
- Laws (LAW)
- Languages (LANGUAGE)
- Works of art (WORK_OF_ART)
- Time values (TIME)
- Percentages (PERCENT)
- Money values (MONEY)
- Quantities (QUANTITY)

Each entity must be replaced with a unique identifier in the format:
- PER_1, PER_2, ...
- LOC_1, LOC_2, ...
- ORG_1, ORG_2, ...
- DATE_1, EVENT_1, etc.

Instructions:
- Replace every full and partial mention of the entity with the correct identifier.
  E.g., "Barack Obama" and "Obama" → PER_1
- Reuse the same identifier consistently across the entire text.
- Do not include code or explanations.
- Return only:
  1. The processed text with all entities replaced.
  2. A JSON list of all replaced entities with this structure:
    {
    "<original mention>", 
    "<PII-style ID>"
    }

Example:

Original: 
"Barack Obama was born in Honolulu in 1961. Obama was elected president in 2008."

Response:
Processed text: 
"[PER_1] was born in [LOC_1] in [DATE_1]. [PER_1] was elected president in [DATE_2]."

JSON:
{
  "Barack Obama": "PER_1",
  "Obama": "PER_1",
  "Honolulu": "LOC_1",
  "1961": "DATE_1",
  "2008": "DATE_2"
}

"""

prompt_level_1_template_iterative = """
You are an anonymization assistant.

Task:
Replace the entity '{entity}' (tagged as {identifier}) with a **realistic but random** replacement of the same type. The replacement does **not** need to match the original context exactly, but must sound plausible and belong to the same category.

Category: {category}
Original: {entity}

Output:
Return only the replacement entity, no explanations.
"""

prompt_level_2_template_iterative = """
You are an anonymization assistant.

Task:
Replace the entity '{entity}' (tagged as {identifier}) with a **contextually appropriate** replacement that fits naturally into the surrounding text. The replacement has to be logically appropriate witin the context, and closely match how the original entity is written.

Category: {category}
Original: {entity}
Context: "{context}"


Output:
Return only the replacement word,  no explanations.
"""

prompt_level_2_template_iterative = """
You are an anonymization assistant.

Task:
Replace the entity '{entity}' (tagged as {identifier}) with a **realistic and logical** replacement of the same type. The replacement need to match the original context exactly, and must belong to the same category.

Category: {category}
Original: {entity}
Context: "{context}"

Output:
Return only the replacement entity, no explanations.
"""

prompt_level_1_template = """
You are an assistant that replaces placeholders in text with similar values.
Replace with the random word that applies to the same detected cathegory.
Respond with the fully replaced version of the original text (do not include the placeholder version).

Instructions:
- Replace every full and partial mention of the entity with the updated identifier
- Reuse the same identifier consistently across the entire text.
- Do **not** include code or explanations.
- Do not ommit the entities, you must replace all of them.
- Return **only**:
  1. The processed text with all entities replaced.
  2. A JSON list of all replaced entities with this structure:
     {"entity": "<original mention>", "updated_entity": "<updated entity>"}

Example:

Original: 
"Barack Obama was born in Honolulu in 1961. Obama was elected president in 2008."

Response:
Processed text: 
"Joseph Biden was born in Scranton in 1942. Joseph was elected president in 2021."

JSON:
[
  {"entity": "Barack Obama", "updated_entity": "Joseph Biden "},
  {"entity": "Obama", "updated_entity": "Joseph"},
  {"entity": "Honolulu", "updated_entity": "Scranton"},
  {"entity": "1961", "updated_entity": "1942"},
  {"entity": "2008", "updated_entity": "2021"}
]

Now process the following text:

"""


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

    

if __name__=="__main__":
    ollama_provider = LlamaProvider()
#     task = """
# Dr. Olena Shevchenko performed a liver transplant on Andriy Kovtun at Kyiv Regional Clinical Hospital on March 14, 2022. The patient, a 43-year-old male from Bila Tserkva, had been on the transplant list for over 9 months.

#     """
#     feedback = "Dr. Olena Shevchenko is still identifiable in the text."
#     mapping= {
#     "Olena Shevchenko": "PER_1",
#     "Andriy Kovtun": "PER_2",
#     "Kyiv Regional Clinical Hospital": "LOC_1",
#     "March 14, 2022": "DATE_3"
#   }
#     result = ollama_provider.run_with_feedback(task, feedback, mapping)
#     print(result)

    # result = ollama_provider.run(task)
    # print("MASKED TEXT")
    # masked_text = result["masked_text"]
    # print(masked_text)
    # mappings = result["mapping"]
    # print("MAPPINGS")
    # print(mappings)


    prompt = """
        I want to mask personally identifiable information (PII) from the text below. Can you help me to undestand, if there is any information not masked left in text, that can be identified by attackers?
        Examples of PII include: full names, organizations, dates, ages, specific locations, or time intervals linked to people.
        Do not include general context, summaries, or text that does not represent PII.
        If you cannot find any personally identifiable information, because everything is masked, tell OK.
        Anonymized Text: "Dr. Dark performed a [PRODUCT_1] on [PER_2] at [LOC_1] Regional Clinical Hospital on [DATE_1]. The patient, a 75-year-old male from London, had been on the [LAW_1] list for over [QUANTITY_1] months."
"""
    result = ollama_provider.invoke(prompt)
    print(result)
    # accuracy = ollama_provider.run_analysis(mappings, masked_text)
    # print("ACCURACY")
    # print(accuracy)
    # text = "In 2020, Sarah Mitchell, the CEO of GlobalTech Innovations, announced a major restructuring within the company. This decision, made after months of deliberations, aimed at addressing the impact of the COVID-19 pandemic on the business. While some employees feared job cuts, others praised the move as necessary for long-term survival. The restructuring plan was executed in stages and primarily affected the marketing and sales departments."
    # masked_text = "In [DATE_0], [PERSON_1], the CEO of [ORG_2], announced a major restructuring within the company. This decision, made after [DATE_3] of deliberations, aimed at addressing the impact of the COVID-19 pandemic on the business. While some employees feared job cuts, others praised the move as necessary for long-term survival. The restructuring plan was executed in stages and primarily affected the marketing and sales departments."
    # question = "What were the main reasons for Sarah Mitchell’s decision to restructure GlobalTech Innovations in 2020?"
    # masked_question = "What were the main reasons for [PERSON_1]’s decision to restructure [ORG_2] in [DATE_0]?"
    # mappings = {'2020': '[DATE_0]', 'Sarah Mitchell': '[PERSON_1]', 'GlobalTech Innovations': '[ORG_2]', 'months': '[DATE_3]'}
    # new_mapping = ollama_provider.apply_masking_level_iteratively(mappings, 1, text)
    # print("NEW MAPPING LEVEL 1:")
    # print(new_mapping)
    # new_mapping = ollama_provider.apply_masking_level_iteratively(mappings, 2, text, margin=300)
    # print("NEW MAPPING LEVEL 2:")
    # print(new_mapping)
