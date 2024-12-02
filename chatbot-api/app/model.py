# model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os
model_path = r"C:\Users\lenovo\Desktop\DevBuddy\chatbot-api\models"



class ChatModel:
    def __init__(self, model_path=model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize BitsAndBytes configuration
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False
        )
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=self.bnb_config,
            device_map={"": 0}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def generate_response(self, prompt: str) -> str:
        # Format prompt
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        
        DEFAULT_SYSTEM_PROMPT = """You are an expert in software development and technology.
        You will be given a context to answer from, including questions about programming, 
        software tools, algorithms, and best practices."""
        
        formatted_prompt = f"{B_INST}{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}{prompt}{E_INST}"
        
        # Generate response
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.2,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def save_model(self, save_path: str):
        """Save the model and tokenizer"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")