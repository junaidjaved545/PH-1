import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=".env")

api_key = os.getenv("OPENAI_API_KEY")

print("KEY LOADED:", api_key is not None)

client = OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-4.1-mini",
    input="Say setup working"
)

print(response.output_text)