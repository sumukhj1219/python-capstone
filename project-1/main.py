from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
from groq import Groq
import uvicorn


client = Groq(api_key="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")
SYSTEM_PROMPT = """
You are a helpful AI assistant.
- Be clear and concise.
- Give structured answers when possible.
- If unsure, say you don't know.
"""
                                                                        
app = FastAPI()

chat_store: Dict[str, List[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    response: str
    history: List[Dict[str, str]]



@app.get("/health")
def health():
    return {"status": "All ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    session_id = request.session_id
    user_input = request.prompt

    if session_id not in chat_store:
        chat_store[session_id] = []

    history = chat_store[session_id]

    history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  
            messages=messages
        )

        assistant_response = completion.choices[0].message.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    history.append({"role": "assistant", "content": assistant_response})

    return ChatResponse(
        session_id=session_id,
        response=assistant_response,
        history=history
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
