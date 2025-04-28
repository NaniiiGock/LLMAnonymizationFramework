from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class DistilBert:
    def __init__(self):
        self.model_name = "nanigock/distil-bert-conll-2003"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.ner_pipeline = pipeline("ner", 
                            model=self.model,
                            tokenizer=self.tokenizer, 
                            aggregation_strategy="simple")
        
    def mask_entities(self, text):
        results = self.ner_pipeline(text)
        entity_mapping = {}
        masked_text = text
        label_counter = {}
        sorted_results = sorted(results, key=lambda x: x['start'], reverse=True)

        for entity in sorted_results:
            word = entity['word']
            label = entity['entity_group']
            if label not in label_counter:
                label_counter[label] = 1
            else:
                label_counter[label] += 1

            placeholder = f"[{label}_{label_counter[label]}]"
            entity_mapping[word] = placeholder
            masked_text = masked_text[:entity['start']] + placeholder + masked_text[entity['end']:]
        return entity_mapping, masked_text

distilbert = DistilBert()
text = "Barack Obama visited Berlin in 2008."
mapping, masked = distilbert.mask_entities(text)

print("Entity Mapping:")
print(mapping)
print("\nMasked Text:")
print(masked)