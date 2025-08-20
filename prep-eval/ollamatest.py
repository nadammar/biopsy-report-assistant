import ollama

# Set up the Ollama model
model = "phi3"  # or whatever model you're using

# Create a prompt
prompt = "Hello, how can I assist you today?"

# Send the prompt to Ollama and get the response
response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])

# Extract and print the content of the response
print(response['message']['content'])
