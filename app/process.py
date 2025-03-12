import logging

from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Type, List

from .prompts import *
from .models import VisualizationNeed, PersonaSelection
from .utils import figure_to_json, handle_exceptions
from .visualization import visualization_generation_pipeline
from .constants import DEVELOPER, USER, DEVELOPER
from .ai import openai_client

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
def set_complexity_level(description: str) -> int:
    """
    Set the complexity level based on the persona.

    Args:
        persona (str): The persona name

    Returns:
        str: The complexity level prompt
    """

    complexity_level = classify_text(description, COMPLEXITY_MATCHING_PROMPT, PersonaSelection).persona_id
    return complexity_level

def get_complexity_level_prompts(complexity_level: int) -> tuple[str,str]:
    """
    Get the complexity level prompt based on the complexity level.

    Args:
        complexity_level (int): The complexity level

    Returns:
        str: The complexity level prompt
    """
    match complexity_level:
        case 0:
            return LVL0_VIZ_PROMPT, LVL0_EXP_PROMPT
        case 1:
            return LVL1_VIZ_PROMPT, LVL1_EXP_PROMPT
        case 2:
            return LVL2_VIZ_PROMPT, LVL2_EXP_PROMPT
        case _:
            return LVL0_VIZ_PROMPT, LVL0_EXP_PROMPT

def generate_visualization(message: str, complexity_level: int, user_description: str, location: str, chat_id: str, scenario: str, topic: str, options: List[str], lang: str='en') -> str:
        viz_complexity, _ = get_complexity_level_prompts(complexity_level)
        try:
            fig, data = visualization_generation_pipeline(message, user_description, location, topic, viz_complexity, scenario, options, lang)
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

def process_user_message(message: str, persona: int, location: str, chat_id: str, lang: str='en') -> str:
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

def process_simple_message(messages: List[any], lang: str='en') -> str:
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
