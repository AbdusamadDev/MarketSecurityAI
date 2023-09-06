import os
import openai
openai.organization = "org-HT99JIANWc0HAX77xuDLeyhl"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()
response = openai.Completion.create(
    model="text-davinci-002",  # Or whichever model you want to use
    prompt="Translate the following English text to French: 'Hello, how are you?'",
    max_tokens=60
)

print(response.choices[0].text.strip())
