import os
from abc import ABC, abstractmethod

from openai import OpenAI
from pydantic import BaseModel
from typing import Type, Dict
from .constants import DEVELOPER, GPT_4o_MINI
from .prompts import OUTPUT_LANGUAGE_PROMPT

class LLMClient(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def completion(self, messages: list[Dict[str,str]], max_tokens: int = 100) -> str:
        pass


class OpenAIClient(LLMClient):
    def __init__(self):
        super().__init__()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


    def completion(self, messages: list[Dict[str,str]], max_tokens: int = 100, temperature: int = 1) -> str:
        try:
            messages.insert(0, {"role": DEVELOPER, "content": OUTPUT_LANGUAGE_PROMPT})
            response = self.client.chat.completions.create(
                model=GPT_4o_MINI,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return ""


    def structured_completion(self, messages: list[Dict[str,str]], response_format: Type[BaseModel], max_tokens: int = 100, temperature: int = 1) -> BaseModel:
        messages.insert(0, {"role": DEVELOPER, "content": OUTPUT_LANGUAGE_PROMPT})
        response = self.client.beta.chat.completions.parse(
            model=GPT_4o_MINI,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )

        return response.choices[0].message.parsed
    
    