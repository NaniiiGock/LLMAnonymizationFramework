
import spacy
from src.models.NER_models.bert.eng_bert_infer import BertEng
from src.models.NER_models.roberta_ukr.ukr_roberta import ROBertaUkr
from src.models.NER_models.distil_bert.distilbert import DistilBert
from spacy import displacy
import os

class NERProcessor:
    def __init__(self, config=None):
        self.model_name = config.get('model', "")

    def preprocess(self, processed_text, pattern_replacements={}, entity_map={}):
        if self.model_name in ["en_core_web_sm", "uk_core_news_sm"]:
            nlp = spacy.load(self.model_name)
            doc = nlp(processed_text)
            # def get_unique_filename(filename):
            #     if os.path.exists(filename):
            #         name, ext = os.path.splitext(filename)
            #         i = 1
            #         while os.path.exists(f"{name}_{i}{ext}"):
            #             i += 1
            #         return f"{name}_{i}{ext}"
            #     else:
            #         return filename

            # filename = "entity_visualization.html"
            # unique_filename = get_unique_filename(filename)

            # with open(unique_filename, "w") as f:
            #     f.write(displacy.render(doc, style="ent", page=True, jupyter=False))
            ner_replacements = {}
            
            for ent in doc.ents:
                if ent.text not in pattern_replacements.values() and ent.text not in entity_map.values(): 
                    replacement = f"[{ent.label_}_{len(entity_map)}]"
                    processed_text = processed_text.replace(ent.text, replacement)
                    entity_map[replacement] = ent.text
                    ner_replacements[ent.text] = replacement
            processed_text = processed_text.replace("\n", "")
            return processed_text, ner_replacements, entity_map
        
        elif self.model_name == "bert_eng":
            bert_infer = BertEng()
            masked_sentence, ner_replacements = bert_infer.mask_entities_in_sentence(processed_text, max_len=128)
            final_replacements = {}
            for entity, cathegory in ner_replacements.items():
                if entity not in pattern_replacements.values() and entity not in entity_map.values():
                    cat_name = cathegory[1:-1].split("_")[0].upper()
                    replacement = f"{cat_name}_{len(entity_map)}"
                    masked_sentence = masked_sentence.replace(cathegory, f"[{replacement}]")
                    entity_map[replacement] = entity
                    final_replacements[entity] = replacement
            return masked_sentence, final_replacements, entity_map

        elif self.model_name == "roberta_ukr":
            roberta_infer = ROBertaUkr()
            masked_sentence, ner_replacements = roberta_infer.run(processed_text)
            for entity, cathegory in ner_replacements.items():
                if entity not in pattern_replacements.values() and entity_map.values():
                    replacement = cathegory.split("_")[0].upper() + "_" + len(entity_map)
                    entity_map[replacement] = entity
                    ner_replacements[entity] = replacement
            return masked_sentence, ner_replacements, entity_map
        
        elif self.model_name == "distil_bert":
            distil_bert_infer = DistilBert()
            masked_sentence, ner_replacements = distil_bert_infer.run(processed_text)
            for entity, cathegory in ner_replacements.items():
                if entity not in pattern_replacements.values() and entity_map.values():
                    replacement = cathegory.split("_")[0].upper() + "_" + len(entity_map)
                    entity_map[replacement] = entity
                    ner_replacements[entity] = replacement
            return masked_sentence, ner_replacements, entity_map

 
    # def replace_entities_with_masks(self, text, mapping):
    #     for entity, mask in mapping.items():
    #         text = text.replace(entity, mask)
    #     return text

    # def replace_entities_with_masks(self, text, mapping):
    # # Sort mappings by length of entity (longest first)
    #     sorted_mappings = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)

    #     for entity, mask in sorted_mappings:
    #         text = text.replace(entity, mask)

    #     return text

    def replace_entities_with_masks(self, text, mapping):
        import re

        sorted_entities = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)
        pattern = r'\b(' + '|'.join(re.escape(entity) for entity, _ in sorted_entities) + r')\b'

        def replace_match(match):
            entity = match.group(0)
            for original, mask in sorted_entities:
                if entity == original:
                    return mask
            return entity

        masked_text = re.sub(pattern, replace_match, text)
        return masked_text

        