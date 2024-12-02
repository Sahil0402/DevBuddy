# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Optional
import torch

app = FastAPI(title="DevBuddy API", version="1.0.0")

# Pydantic models for request/response
class Question(BaseModel):
    text: str

class Answer(BaseModel):
    response: str

# Global variables
model = None
tokenizer = None
pipe = None
db = None

def load_trained_components():
    """Load the pre-trained model, tokenizer, and vector store"""
    global model, tokenizer, pipe, db
    
    try:
        # Load model and tokenizer
        model_path = r"C:\Users\lenovo\Desktop\DevBuddy\chatbot-api\models"  # Update this path
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": 0}
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.2
        )
        
        # Load vector store
        db = Chroma(
            persist_directory=r"C:\Users\lenovo\Desktop\DevBuddy\chatbot-api\db",  # Update this path
            embedding_function=HuggingFaceEmbeddings()
        )
        
        return True
    except Exception as e:
        print(f"Error loading components: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    if not load_trained_components():
        raise RuntimeError("Failed to initialize application components")

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """Endpoint to ask questions to DevBuddy"""
    try:
        # Get relevant documents
        docs = db.similarity_search(question.text, k=3)
        context = [doc.page_content for doc in docs]
        
        # Create prompt
        prompt = f"""Given the context that has been provided: 
        {' '.join(context)}
        
        Answer the following question: {question.text}"""
        
        # Generate response
        response = pipe(prompt)[0]['generated_text']
        
        return Answer(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "db_loaded": db is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
