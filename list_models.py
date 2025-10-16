import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

models = genai.list_models()
for m in models:
    print(f"{m.name}  |  {m.description}")
