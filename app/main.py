import logging
import json
import random

from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Type
import plotly.graph_objects as go

from .prompts import *
from .models import VisualizationNeed, PersonaSelection, NormalizedOpenMeteoData
from .utils import figure_to_base64, enhance_plotly_figure, handle_exceptions
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



@handle_exceptions()
def describe_visualization(data: list[NormalizedOpenMeteoData], complexity_level: str, fig: go.Figure) -> str:
    """
    Describe the visualization based on the given data and complexity level.

    Args:
        data (ProcessedData): The processed data to describe
        complexity_level (str): The complexity level of the user
        fig (go.Figure): The generated figure to describe

    Returns:
        str: The description of the visualization
    """
    base64_image = figure_to_base64(fig)
    explanation_plan = anthropic_client.completion(
        messages=[
            {"role": USER, "content": complexity_level},
            {"role": USER, "content": [
                {
                    "type": "text",
                    "text": EXPLANATION_PLAN_PROMPT,
                },
                {
                    "type": "image",
                    "source": { "type": "base64",
                                "data": base64_image,
                                "media_type": "image/png"},
                },
            ]},
        ],
        temperature=0.7,
        max_tokens=300,
    )

    data_description = ""
    for data_point in data:
        data_description += f"{data_point.generate_data_description()}\n\n"

    response = anthropic_client.completion(
        messages=[
            {"role": USER, "content": complexity_level},
            {"role": USER, "content": [
                {
                    "type": "text",
                    "text": EXPLANATION_GENERATION_PROMPT.format(explanation_plan=explanation_plan, data_description=data_description),
                },
                {
                    "type": "image",
                    "source": { "type": "base64",
                                "data": base64_image,
                                "media_type": "image/png"},
                },
            ]},
        ],
        temperature=0.7,
        max_tokens=300,
    )

    return response

def main() -> tuple[go.Figure, str]:
    with open('mock.json', 'r') as file:
        conversations = json.load(file)

    conversation = random.choice(conversations)

    for message in conversation['messages']:
        viz_need = classify_text(message['message'], VISUALIZATION_NEED_PROMPT, VisualizationNeed)

        if viz_need.need_visualization:
            logging.info(f"Needed viz : {message['message']}")
            logging.info(f"Topic of interest : {viz_need.topic_of_interest}")

            viz_complexity, exp_complexity = set_complexity_level(message['persona'])
            try:
                fig, data = visualization_generation_pipeline(message['message'], message['persona'], viz_need.topic_of_interest, viz_complexity)
                fig = enhance_plotly_figure(fig)

                description = describe_visualization(data, exp_complexity, fig)

                return fig, description
            except Exception:
                logging.error(f"Error generating visualization:", exc_info=True)
                return
            finally:
                in_token, out_token = anthropic_client.get_total_tokens()

                print(f"OPENAI spent tokens = {openai_client.get_total_tokens()}")
                print(f"ANTHROPIC spent tokens = {in_token*0.000003:.2f}$, {out_token*0.000015:.2f}$")

