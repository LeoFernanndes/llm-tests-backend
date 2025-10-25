import os
from pathlib import Path

from typing import List, Literal, Union

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


load_dotenv()


class OpenAiMessage(BaseModel):
    sender: Union[Literal['ai'], Literal['human'], Literal['system']]
    content: str


llm = init_chat_model("gpt-4o-mini", model_provider="openai")
chat = ChatOpenAI(model="gpt-4o")

parent_dir = Path(__file__).parent.parent.resolve()
contexts_dir = os.path.join(parent_dir, 'contexts')


def get_context(name: str) -> BaseMessage:

    available_contexts = {
        'high_school_counselor': 'counselor_context.md'
    }

    with open(os.path.join(contexts_dir, available_contexts[name]), "r") as f:
        context_text = f.read() 

    context = SystemMessage(content=context_text)
    return context


class OpenAiSimpleChat:

    def chat(self, messages: List[OpenAiMessage], context_name: str) -> BaseMessage:

        context = get_context(context_name)

        _messages: List[BaseMessage] = [context]
        for m in messages:
            if m.sender == 'ai':
                _m = AIMessage(content=m.content)
            elif m.sender == 'human':
                _m = AIMessage(content=m.content)
            elif m.sender == 'system':
                _m = AIMessage(content=m.content)

            try:
                _messages.append(_m)
            except ValueError as ve:
                print(ve)
        print(_messages)
        return chat.invoke(_messages)
