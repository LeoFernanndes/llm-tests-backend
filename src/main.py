from typing import Literal, List, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_impl.openai.simple_chat import OpenAiMessage, OpenAiSimpleChat
from pydantic import BaseModel


app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Hello": "World"}


@app.get("/contexts")
def list_available_contexts():
    return [
        {
            "context": "high_school_counselor",
            "readable": "High school conselour"
        }
    ]


class OpenAiPayload(BaseModel):
    messages: List[OpenAiMessage]
    context_name: str


@app.post("/openai-messages")
def send_openai_message(openai_payload: OpenAiPayload) -> OpenAiMessage:
    chat = OpenAiSimpleChat()
    chat_response = chat.chat(openai_payload.messages, openai_payload.context_name)
    return OpenAiMessage(sender='ai', content=chat_response.content)
