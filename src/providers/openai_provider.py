import openai 
from dotenv import load_dotenv

load_dotenv()

class OpenAIProvider:
    def __init__(self, config={}):
        self.config = config
        self.model = self.config.get("model", "gpt-4o")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 300)
        
    async def query_llm(self, prompt):
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    # {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying LLM: {str(e)}"