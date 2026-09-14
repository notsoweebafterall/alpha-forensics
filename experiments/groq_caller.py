import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def groq_llm_caller(prompt: str) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content