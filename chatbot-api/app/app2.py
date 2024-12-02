from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS handling library
from transformers import AutoTokenizer, AutoModelForCausalLM  # CausalLM is used for text generation
import torch
import time  # To measure time

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the model and tokenizer
model_path = r"C:\Users\lenovo\Desktop\DevBuddy\chatbot-api\models"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

@app.route('/')
def hello_world():
    return 'Text Generation API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Start timing
        start_time = time.time()
        
        # Get the text from the request
        data = request.get_json()
        text = data['text']

        # Tokenize input text
        inputs = tokenizer(text, return_tensors='pt')

        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,  # Maximum length of generated text
                num_return_sequences=1,  # Number of generated sequences
                temperature=0.7,  # Adjust randomness (lower is more deterministic)
                top_p=0.9,  # Nucleus sampling (top-p sampling)
                do_sample=True  # Enable sampling
            )

        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # End timing
        end_time = time.time()
        time_taken = end_time - start_time

        return jsonify({'generated_text': generated_text, 'time_taken_seconds': time_taken})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
