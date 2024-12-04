import logging
import requests
import json
import random
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from .prompts import CLASSIFYING_PROMPT, SELECTING_API_PROMPT, GENERATE_VISUALIZATION_PROMPT, BUILD_EXTERNAL_QUERY_PROMPT
from .constants import SYSTEM, USER, GPT_4o_MINI
from .api import APIEndpointRegistry, APIType
from .utils import describe_dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)



def detect_visualization_need(text: str) -> int:
    """
    Determine if the given text suggests a need for visualization.

    Args:
        text (str): Input text to analyze

    Returns:
        int: 1 if visualization is needed, 0 otherwise
    """
    try:
        response = client.chat.completions.create(
            model=GPT_4o_MINI,
            messages=[
                {"role": SYSTEM, "content": CLASSIFYING_PROMPT},
                {
                    "role": USER,
                    "content": f"Classify if this text needs visualization:\n\n{text}",
                },
            ],
            max_tokens=1,
            temperature=0,
        )

        result = response.choices[0].message.content.strip()
        return int(result) if result in ["0", "1"] else 0

    except Exception as e:
        logging.error(f"Error in visualization need detection: {e}")
        return 0


class APISelection(BaseModel):
    url: str


def select_api(prompt: str) -> str:
    """
    Determine what API to query based on the user demand.

    Args:
        text (str): Input text to analyze

    Returns:
        str: API endpoint to query
    """
    try:
        response = client.beta.chat.completions.parse(
            model=GPT_4o_MINI,
            messages=[
                {"role": SYSTEM, "content": SELECTING_API_PROMPT},
                {
                    "role": USER,
                    "content": prompt,
                },
            ],
            max_tokens=20,
            temperature=0,
            response_format=APISelection,
        )

        result = response.choices[0].message.parsed.url
        known_endpoints = APIEndpointRegistry.get_endpoints(APIType.OPEN_METEO)
        for endpoint in known_endpoints:
            if result == endpoint.full_url:
                return result
        return None

    except Exception as e:
        logging.error(f"Error in API selection: {e}")
        return None

def build_external_api_query(api_endpoint: str, prompt: str) -> dict:
    """
    Build the query to request the chosen external external API.

    Args:
        api_endpoint (str): The API endpoint to query

    Returns:
        dict: A dictionary of parameters for the API query
    """
    try:
        system_prompt = str.format(BUILD_EXTERNAL_QUERY_PROMPT, api_endpoint=api_endpoint, api_endpoint_parameters=APIEndpointRegistry()._get_endpoint_parameters(api_endpoint))

        response = client.beta.chat.completions.parse(
            model=GPT_4o_MINI,
            messages=[
                {"role": SYSTEM, "content": system_prompt},
                {
                    "role": USER,
                    "content": prompt,
                },
            ],
            max_tokens=100,
            temperature=0,
            response_format=APISelection
        )

        result = response.choices[0].message.parsed.url
        return result

    except Exception as e:
        logging.error(f"Error in API query parameter building: {e}", exc_info=True)
        return None

class VisualizationType(BaseModel):
    json_definition: str

def generate_visualization(data, prompt) -> dict:
    try:
        visualization_prompt = str.format(GENERATE_VISUALIZATION_PROMPT, data_description=describe_dict(data))
        response = client.beta.chat.completions.parse(
            model=GPT_4o_MINI,
            messages=[
                {"role": SYSTEM, "content": visualization_prompt},
                {
                    "role": USER,
                    "content": f"{prompt}",
                },
            ],
            max_tokens=5000,
            temperature=0,
            response_format=VisualizationType
        )
        
        result = response.choices[0].message.parsed.json_definition
        return result
    except Exception as e:
        logging.error(f"Error in visualization generation: {e}")
        return None


def main():
    with open('mock.json', 'r') as file:
        conversations = json.load(file)

    # Select a random conversation
    conversation = random.choice(conversations)



    logging.info(f"Conversation: {conversation}")

    for message in conversation['messages']:
        if detect_visualization_need(message['message']):
            logging.info(f"Visualization needed for message: {message['message']}")
            
            api_endpoint = select_api(message['message'])
            logging.info(f"Selected API endpoint: {api_endpoint}")
            if api_endpoint:
                query = build_external_api_query(api_endpoint, message['message'])
                logging.info(f"Query {query}")
            
                response = requests.get(query)
                if response.status_code == 200:
                    data = response.json()
                    logging.info(f"Data retrieved: {describe_dict(data)}")

                    return generate_visualization(data, message['message']), data
                else:
                    logging.error(f"Error retrieving data: {response.text}")
