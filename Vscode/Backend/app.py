



# backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
import os

# uvicorn app:app --reload --host 0.0.0.0 --port 8000


HF_TOKEN = "YOUR HUGGING FACE TOKEN"
# client = InferenceClient(model="meta-llama/Llama-4-Scout-17B-16E-Instruct", token=HF_TOKEN)


client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN
)


app = FastAPI()

# Allow requests from Expo app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list[list[str]] = []

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    # Prepare messages in HF API format
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": msg}
            for i, msg in enumerate(sum(req.history, []) + [req.message])]
    output = client.chat.completions.create(
        messages=msgs,
        stream=False,
        max_tokens=300,
    )
    reply = output.choices[0].message.content
    return {"reply": reply}






