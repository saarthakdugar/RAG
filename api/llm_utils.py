# api_server/utils/llm_utils.py
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

def generate_sql(question, schema_info):
    prompt = f"""
You are a helpful assistant that writes SQL Server queries.
Given the following schema:
{schema_info}
Write a SQL query for: "{question}"
SQL:
"""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0
    )
    return response.choices[0].text.strip()