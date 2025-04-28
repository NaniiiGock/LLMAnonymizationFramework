import json
from src.processing.ollama_processor import LlamaProvider

class AdversarialLLMNER:
    def __init__(self, model_anonymizer="llama3.2", model_adversarial="llama3.2", max_steps=5):
        self.model_adversarial = model_adversarial
        self.model_anonymizer = model_anonymizer
        self.max_steps = max_steps
        self.initialize_providers()

    def initialize_providers(self):
        self.adversarial_provider = LlamaProvider(self.model_adversarial)
        self.anonymizer_provider = LlamaProvider(self.model_anonymizer)

    def anonymize(self, text, feedback=None, prev_mapping=None):
        feedback_text = f"\nAdvisory Feedback: {feedback}" if feedback else ""
        mapping_text = f"Here is the previous mapping: + {json.dumps(prev_mapping, indent=2)}" if prev_mapping else "None"
        prompt = text + feedback_text  + mapping_text
        result = self.anonymizer_provider.run(prompt)
        return {
            "anonymized_text" : result["masked_text"],
            "mapping" : result["mapping"]
        }
        
    def adversarial_reidentify(self, anonymized_text):
        prompt = f"""
        You are an adversarial agent attempting to recover personally identifiable information (PII) from the anonymized text below.

        Examples of PII include: full names, organizations, dates, ages, specific locations, or time intervals linked to people.

        Do not include general context, summaries, or text that does not represent PII.

        Anonymized Text: "{anonymized_text}"
        Inferred Information:
        """
        response = self.adversarial_provider.invoke(prompt)
        return response

    def refine_anonymization(self, original_text):
        best_result = None
        best_score = float('inf')
        feedback = None
        mapping = None

        for step in range(self.max_steps):
            print(f"\n--- Step {step+1} ---")
            for i in range(3):
                print("trial" , i)
                result = self.anonymize(original_text, feedback=feedback, prev_mapping=mapping)
                anonymized = result["anonymized_text"]
                mapping = result["mapping"]
                if mapping is not {}:
                    break

            reidentified = self.adversarial_reidentify(anonymized)
            original_words = set(original_text.lower().split())
            reidentified_words = set(reidentified.lower().split())
            overlap_score = len(original_words & reidentified_words)
            print("Anonymized:", result)
            print("Reidentified guess:", reidentified)
            print("Overlap Score:", overlap_score)

            if overlap_score < best_score:
                best_score = overlap_score
                best_result = {
                    "step": step + 1,
                    "anonymized_text": anonymized,
                    "mapping": mapping,
                    "reidentified_guess": reidentified,
                    "score": overlap_score
                }

            if best_score == 0:
                break

            feedback = reidentified

        return best_result



# from models.advisory.looped_llm_advisory import AdversarialLLMNER
# import json

# if __name__ == "__main__":
#     input_text = """

# Dr. Olena Shevchenko performed a liver transplant on Andriy Kovtun at Kyiv Regional Clinical Hospital on March 14, 2022. The patient, a 43-year-old male from Bila Tserkva, had been on the transplant list for over 9 months.

# """
#     anonymizer = AdversarialLLMNER()
#     result = anonymizer.refine_anonymization(input_text)
#     print("Best anonymization result:")
#     print(json.dumps(result, indent=2, ensure_ascii=False))
