
from processing.ollama_processor import LlamaProvider

class ContexAwareMasker:
    def __init__(self, config={}):
        self.mode = config.get("mode", "iteratively")
        self.model = config.get("model", "llama")
        self.level = int(config.get("level", 2))
        self.margin = config.get("margin", 500)

    def mask_iteratively_llama(self, mappings, context_text):
        llama = LlamaProvider()
        new_replacements = llama.apply_masking_level_iteratively(mappings, self.level, context_text, self.margin)
        new_entity_map = {}
        for key, val in new_replacements.items():
            new_entity_map[val] = key
        return new_replacements, new_entity_map

    def replace_entities_with_masks(self, text, mapping):
        for entity, mask in mapping.items():
            text = text.replace(entity, mask)
        return text
    
    def run(self, mappings, context_text):
        if self.mode == "iteratively":
            return self.mask_iteratively_llama(mappings, context_text)
        elif self.mode == "one_shot":
            return None
