import logging
import json

from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Type
import plotly.graph_objects as go
from fastapi.responses import StreamingResponse

from .prompts import *
from .models import VisualizationNeed, PersonaSelection, NormalizedOpenMeteoData
from .utils import figure_to_json, enhance_plotly_figure, handle_exceptions
from .visualization import visualization_generation_pipeline
from .constants import DEVELOPER, USER, DEVELOPER
from .ai import openai_client, anthropic_client

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s \n\n')

load_dotenv()


@handle_exceptions()
def classify_text(text: str, classification_prompt: str, response_format: Type[BaseModel], max_tokens: int = 20) -> BaseModel:
    """
    Classify the given text based on the provided prompt.

    Args:
        text (str): Input text to analyze
        classification_prompt (str): The prompt to use for classification
        response_format (BaseModel): The Pydantic model to use for the response format
        max_tokens (int): The maximum number of tokens to generate
    Returns:
        BaseModel: The classified result parsed into the specified response format
    """
    response = openai_client.structured_completion(
        messages=[
            {"role": DEVELOPER, "content": classification_prompt},
            {
                "role": USER,
                "content": f"Classify this text:\n\n{text}",
            },
        ],
        response_format=response_format,
        max_tokens=max_tokens,
        temperature=0.8,
    )

    return response



handle_exceptions()
def set_complexity_level(persona: str) -> tuple[str, str]:
    """
    Set the complexity level based on the persona.

    Args:
        persona (str): The persona name

    Returns:
        str: The complexity level prompt
    """
    with open('personas.json', 'r') as file:
        personas = json.load(file)
    
    user_description = next((p['tuning'] for p in personas if p['name'] == persona), None)
    
    if not user_description:
        raise ValueError(f"Persona '{persona}' not found in personas.json")

    complexity_level = classify_text(user_description, COMPLEXITY_MATCHING_PROMPT, PersonaSelection).persona_id

    if complexity_level == 0:
        return LVL0_VIZ_PROMPT, LVL0_EXP_PROMPT
    elif complexity_level == 1:
        return LVL1_VIZ_PROMPT, LVL1_EXP_PROMPT
    elif complexity_level == 2:
        return LVL2_VIZ_PROMPT, LVL2_EXP_PROMPT
    else:
        raise ValueError("Invalid complexity level")


def process_user_message(message: str, persona: str, location: str, chat_id: str, lang: str='en') -> str:
    """
    Process the user message and generate a visualization and explanation.

    Args:
        message (str): The user message
        persona (str): The persona name
        location (str): The area of interest of the user

    Returns:
        tuple[str, str]: The visualization as PLotly JSON and explanation
    """

    visualization_need = classify_text(message, VISUALIZATION_NEED_PROMPT, VisualizationNeed)

    if visualization_need.need_visualization:
        viz_complexity, exp_complexity = set_complexity_level(persona)
        try:
            fig, data = visualization_generation_pipeline(message, persona, location, visualization_need.topic_of_interest, viz_complexity, lang)
            fig = figure_to_json(fig)

            data_description = ""
            for data_point in data:
                data_description += f"{data_point.generate_data_description()}\n\n"

            with open(f"{chat_id}.txt", "w") as file:
                file.write(data_description)
            
            return fig
        except:
            logging.error(f"Error generating visualization:", exc_info=True)
            return "", ""
    else:
        return "", ""

